from flask import Flask, request
import json
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, BertModel, AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification, XLMRobertaForSequenceClassification
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from utils import fix_seed, creat_optimizer_and_scheduler, create_dataloader, swa
from retrieval_dataset import RetrievalDataset
import json
from main_retrieval import get_threshold, get_similarity_by_token, get_similarity_by_sentence
app = Flask(__name__)
import time


def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--random_lr', type=float, default=1e-4)
    parser.add_argument('--pretrained_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule_type', type=str, default='poly')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--task_name', type=str, default='base')
    parser.add_argument('--encoder_dir', type=str, default='../user_data/m3e-large')
    parser.add_argument('--use_contras', action='store_true')
    parser.add_argument('--use_classify', action='store_true')
    parser.add_argument('--use_momentum', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--pack_with_cate', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--pair_text', action='store_true')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--loss_func', type=str, default='bce')

    args = parser.parse_args()

    return args

# 初始化加载模型
encoder_dir = "../user_data/m3e-large"
args = args_parser()
device = torch.device(args.device)
start = time.time()
config = AutoConfig.from_pretrained(encoder_dir)
model = BertModel(config)

state_dict = torch.load('../user_data/models/model_1.pt')
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('bert.'):
        k = k.replace('bert.', '')
    new_state_dict[k] = v
msg = model.load_state_dict(new_state_dict, strict=False)
end = time.time()

print(msg)
print('Init model cost time: ', end - start)

model = model.to(device)
model.eval()
model = model.half()


def inference(model, dataset, device, args):
    t1 = time.time()
    with torch.no_grad():
        for batch in dataset:
            # time.sleep(0.5)
            record_ids = batch['record_ids'].numpy().tolist()
            query_input_ids = batch['query_input_ids'].to(device)
            query_masks = batch['query_masks'].to(device).to(torch.float16)
            support_input_ids = batch['support_input_ids'].to(device)
            support_masks = batch['support_masks'].to(device).to(torch.float16)

            # forward
            start = time.time()
            query_output = model(query_input_ids, query_masks, return_dict=True, output_hidden_states=True)
            support_output = model(support_input_ids, support_masks, return_dict=True, output_hidden_states=True)
            end = time.time()
            print('Model forward cost time: ', end - start)

            # 全局相似度
            start = time.time()
            scores = get_similarity_by_sentence(query_output, support_output, query_masks, support_masks, emb_type='mean', idx=-1)
            end = time.time()
            print('Cal score1 cost time: ', end - start)

            # 局部相似度
            start = time.time()
            scores2 = get_similarity_by_token(query_output.last_hidden_state, support_output.last_hidden_state, query_masks, support_masks)[0]
            scores = (scores + scores2) / 2
            end = time.time()
            print('Cal score2 cost time: ', end - start)

            # 计算当前task的阈值
            start = time.time()
            preds = np.zeros(len(record_ids))
            # thrh = max(np.percentile(scores, q=95), 0.48)
            support_embedding = support_output.hidden_states[-1]
            support_embedding = (support_embedding * support_masks.unsqueeze(-1)).sum(1) / support_masks.sum(1, keepdim=True)
            thrh = get_threshold(support_embedding, support_output.last_hidden_state, support_masks,
                                 device) + 0.025 + 0.0125
            # thrh = 0.48
            thrh = max(np.percentile(scores, q=95), thrh)
            end = time.time()
            print('Cal thrh cost time: ', end - start)

            preds[np.array(scores) > thrh] = 1
    t2 = time.time()
    print('Total cost time: ', t2 - t1)
    return record_ids, preds.tolist()


def main(args, model, data):
    fix_seed(args.seed)
    device = torch.device(args.device)

    start = time.time()
    test_dataset = RetrievalDataset(data, 'valid', args.encoder_dir, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
    end = time.time()
    print('Init dataset cost time: ', end - start)

    res = inference(model, test_dataloader.dataset, device, args)
    return res


@app.route('/predict', methods=["POST"])
def testPost():
    request_data = request.get_data()
    if request_data is None or request_data == "":
        return {'code': -1}
    data = [json.loads(request_data)]

    record_ids, pred_labels = main(args, model, data)

    res = {}
    res['code'] = 0
    res['data'] = [{"record_id": rid, "label": int(label)} for rid, label in zip(record_ids, pred_labels)]

    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)

    # with open('../raw_data/test_data.txt', 'r', encoding='utf-8') as fr:
    #     lines = fr.readlines()
    # data = [json.loads(l) for l in lines]
    # predicts = []
    # start = time.time()
    # for item in data:
    #     data = [item]
    #     record_ids, pred_labels = main(args, model, data)
    #
    #     res = {}
    #     res['code'] = 0
    #     res['data'] = [{"record_id": rid, "label": int(label)} for rid, label in zip(record_ids, pred_labels)]
    #     predicts.append(json.dumps(res))
    # end = time.time()
    #
    # labels = []
    # record_ids = []
    # for item in predicts:
    #     sample = json.loads(item)
    #     labels += [d['label'] for d in sample['data']]
    #     record_ids += [d['record_id'] for d in sample['data']]
    #
    # df = pd.DataFrame()
    # df['record_id'] = record_ids
    # df['label'] = labels
    # df['label'] = df['label'].astype(int)
    # df.to_csv('./sub.csv', index=False)
    #
    # print('Cost time: ', end - start)
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, BertModel, AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification, XLMRobertaForSequenceClassification
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import math
from utils import fix_seed, creat_optimizer_and_scheduler, create_dataloader, swa
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, auc, precision_recall_curve
from sentence_transformers import SentenceTransformer
from retrieval_dataset import RetrievalDataset
import json
import jieba
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
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--encoder_dir', type=str, default='')
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


# 基于support计算当前task的阈值
def get_threshold(embedding, last_hidden_states, masks, device):
    # # mean_embedding = torch.mean(embedding, dim=0, keepdim=True)
    # embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    # # mean_embedding = torch.nn.functional.normalize(mean_embedding, p=2, dim=-1)
    # sim1 = torch.matmul(embedding, embedding.T)#.squeeze(1)
    # # sim1 = torch.matmul(embedding, mean_embedding.T).squeeze(1)
    # # print(sim1)
    # # thrh = np.percentile(sim1.detach().clone().cpu().numpy(), q=75)
    # #
    sim2 = get_similarity_by_token(last_hidden_states, last_hidden_states, masks, masks, cal_thrh=True)[1]
    sim2 = torch.tensor(sim2).to(device)
    # # print(sim2.shape, sim1.shape)
    #
    # sim = (sim1 + sim2) / 2
    # thrh = sim.mean().item()
    sim = sim2

    mask = torch.ones_like(sim).to(device)
    thrh = torch.sum(torch.triu(sim, diagonal=1)) / torch.sum(torch.triu(mask, diagonal=1))
    thrh = thrh.item()

    return thrh


def val(model, dataloader, device, args, bge):

    all_task_ids, all_record_ids, all_probs = [], [], []
    all_labels = []
    all_cates = []
    all_preds = []
    step = 0

    with open('./data/ChineseStopWords.txt', 'r') as fr:
        lines = fr.readlines()
    stop_words = [l.strip() for l in lines]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            record_ids = batch['record_ids'].squeeze(0).numpy().tolist()
            task_id = batch['task_id'].squeeze(0).numpy().tolist()[0]
            category_name = batch['category_name'][0]
            labels = batch['labels'].squeeze(0).cpu().numpy().tolist()
            query_input_ids = batch['query_input_ids'].squeeze(0).to(device)
            query_masks = batch['query_masks'].squeeze(0).to(device).to(torch.float)
            support_input_ids = batch['support_input_ids'].squeeze(0).to(device)
            support_masks = batch['support_masks'].squeeze(0).to(device).to(torch.float)
            query_texts = [item[0] for item in batch['query_texts']]
            support_texts = [item[0] for item in batch['support_texts']]

            # forward
            query_output = model(query_input_ids, query_masks, return_dict=True, output_hidden_states=True)
            support_output = model(support_input_ids, support_masks, return_dict=True, output_hidden_states=True)

            # 分别计算全局相似度和局部相似度
            scores = get_similarity_by_sentence(query_output, support_output, query_masks, support_masks, emb_type='mean', idx=-1)
            scores2 = get_similarity_by_token(query_output.last_hidden_state, support_output.last_hidden_state, query_masks, support_masks)[0]
            scores = scores * 0.5 + scores2 * 0.5

            # 计算阈值
            preds = np.zeros(len(record_ids))
            support_embedding = support_output.hidden_states[-1]
            support_embedding = (support_embedding * support_masks.unsqueeze(-1)).sum(1) / support_masks.sum(1, keepdim=True)
            thrh = get_threshold(support_embedding, support_output.last_hidden_state, support_masks, device) + 0.025 + 0.0125
            thrh = max(np.percentile(scores, q=95), thrh)

            preds[np.array(scores) > thrh] = 1

            all_task_ids += [task_id] * len(record_ids)
            all_record_ids += record_ids
            all_probs += scores.tolist()
            all_labels += labels
            all_cates += [category_name] * len(labels)
            all_preds += preds.tolist()

            step += 1

            # if step > 100:
            #     break

    unique_cates = list(set(all_cates))
    f1_dic = {}
    for i, cate in enumerate(all_cates):
        if cate not in f1_dic:
            f1_dic[cate] = {'label': [], 'pred': [], 'prob': []}
        f1_dic[cate]['label'].append(all_labels[i])
        f1_dic[cate]['pred'].append(all_preds[i])
        f1_dic[cate]['prob'].append(all_probs[i])

    avg_f1 = 0
    avg_auprc = 0
    cnt = 0
    for cate in unique_cates:
        cate_f1 = f1_score(f1_dic[cate]['label'], f1_dic[cate]['pred'], zero_division=0)
        cate_precision = precision_score(f1_dic[cate]['label'], f1_dic[cate]['pred'], zero_division=0)
        cate_recall = recall_score(f1_dic[cate]['label'], f1_dic[cate]['pred'], zero_division=0)
        cate_auc = roc_auc_score(f1_dic[cate]['label'], f1_dic[cate]['prob'])
        p, r, thresholds = precision_recall_curve(f1_dic[cate]['label'], f1_dic[cate]['prob'], pos_label=1)
        cate_auprc = auc(r, p)
        print(
            f'validation cate={cate}, avg_f1={cate_f1}, precision={cate_precision}, recall={cate_recall}, '
            f'auc={cate_auc}, auprc={cate_auprc}')

        avg_f1 += cate_f1
        avg_auprc += cate_auprc
        cnt += 1

    avg_f1 /= cnt
    avg_auprc /= cnt
    print('macro f1: ', avg_f1, '; avg auprc: ', avg_auprc)


def inference(model, dataloader, device, args):
    all_task_ids, all_record_ids, all_probs = [], [], []
    all_preds = []
    support_size = 5
    with torch.no_grad():
        for batch in tqdm(dataloader):
            record_ids = batch['record_ids'].squeeze(0).numpy().tolist()
            task_id = batch['task_id'].squeeze(0).numpy().tolist()[0]
            category_name = batch['category_name'][0]
            query_input_ids = batch['query_input_ids'].squeeze(0).to(device)
            query_masks = batch['query_masks'].squeeze(0).to(device).to(torch.float)
            support_input_ids = batch['support_input_ids'].squeeze(0).to(device)
            support_masks = batch['support_masks'].squeeze(0).to(device).to(torch.float)
            # print(query_input_ids.shape)

            # forward
            query_output = model(query_input_ids, query_masks, return_dict=True, output_hidden_states=True)
            support_output = model(support_input_ids, support_masks, return_dict=True, output_hidden_states=True)

            # 计算全局相似度和局部相似度
            scores = get_similarity_by_sentence(query_output, support_output, query_masks, support_masks, emb_type='mean', idx=-1)
            scores2 = get_similarity_by_token(query_output.last_hidden_state, support_output.last_hidden_state, query_masks, support_masks)[0]
            scores = (scores + scores2) / 2

            # 计算当前task的阈值
            preds = np.zeros(len(record_ids))
            support_embedding = support_output.hidden_states[-1]
            support_embedding = (support_embedding * support_masks.unsqueeze(-1)).sum(1) / support_masks.sum(1, keepdim=True)
            thrh = get_threshold(support_embedding, support_output.last_hidden_state, support_masks,
                                 device) + 0.025 + 0.0125
            thrh = max(np.percentile(scores, q=95), thrh)

            preds[np.array(scores) > thrh] = 1

            all_task_ids += [task_id] * len(record_ids)
            all_record_ids += record_ids
            all_probs += scores.tolist()
            all_preds += preds.tolist()

    # all_preds = np.zeros(len(all_probs))
    # all_preds[np.array(all_probs) > 0.8] = 1
    df = pd.DataFrame()
    df['task_id'] = all_task_ids
    df['record_id'] = all_record_ids
    df['label'] = all_preds
    df['label'] = df['label'].astype(int)

    df.to_csv('./output/sub.csv', index=False)


def get_similarity_by_token(query, support, query_masks, support_masks, cal_thrh=False):
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    support = torch.nn.functional.normalize(support, p=2, dim=-1)
    scores = []
    raw_scores = []
    for i in range(len(query)):
        cur_embedding = query[i]
        # cur_word = texts[i]
        cur_mask = query_masks[i].unsqueeze(1).to(torch.float)
        best_sim = 0

        avg_scores = []
        for j in range(len(support)):
            # if cal_thrh and i == j:
            #     continue
            sim = torch.matmul(cur_embedding, support[j].T)
            mask = torch.matmul(cur_mask, support_masks[j].unsqueeze(0).to(torch.float))
            sim *= mask
            sim = (torch.sum(torch.max(sim, dim=1)[0]) / torch.sum(cur_mask)).item()

            avg_scores.append(sim)
        raw_scores.append(avg_scores)
        scores.append(np.mean(avg_scores))
    # print(scores)
    return np.array(scores), np.array(raw_scores)


def get_similarity_by_sentence(query_output, support_output, query_masks, support_masks, emb_type='cls', idx=-1):
    if emb_type == 'pooler':
        query_embedding = query_output.pooler_output
        support_embedding = support_output.pooler_output
    elif emb_type == 'cls':
        query_embedding = query_output.hidden_states[idx][:, 0, :]
        support_embedding = support_output.hidden_states[idx][:, 0, :]
    elif emb_type == 'mean':
        query_embedding = query_output.hidden_states[idx]
        query_embedding = (query_embedding * query_masks.unsqueeze(-1)).sum(1) / query_masks.sum(1, keepdim=True)
        #
        support_embedding = support_output.hidden_states[idx]
        support_embedding = (support_embedding * support_masks.unsqueeze(-1)).sum(1) / support_masks.sum(1, keepdim=True)
    else:
        raise ValueError

    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
    mean_support_embedding = torch.mean(support_embedding, dim=0, keepdim=True)
    mean_support_embedding = torch.nn.functional.normalize(mean_support_embedding, p=2, dim=-1)

    sim = torch.matmul(query_embedding, mean_support_embedding.T).squeeze(1)
    scores = sim.cpu().detach().clone().numpy()

    return scores


def main(args):
    fix_seed(args.seed)
    device = torch.device(args.device)

    with open(args.data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = [json.loads(l) for l in lines]
    test_dataset = RetrievalDataset(data, 'valid', args.encoder_dir, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    model = AutoModel.from_pretrained(args.encoder_dir)

    state_dict = torch.load('../user_data/models/model_1.pt')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            k = k.replace('bert.', '')
        new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)

    model = model.to(device)
    model.eval()
    model = model.half()

    if args.phase == 'test':
        inference(model, test_dataloader, device, args)
    else:
        val(model, test_dataloader, device, args, None)


if __name__ == '__main__':
    args = args_parser()
    main(args)

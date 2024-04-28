from transformers import AutoModelForMaskedLM, BertForMaskedLM, AutoModelForSequenceClassification, BertForSequenceClassification, AutoConfig
import torch
from torch.cuda.amp import GradScaler
from torch import autocast
from utils import fix_seed, creat_optimizer_and_scheduler, create_dataloader, copy_params, momentum_update
import json
from retrieval_dataset import RetrievalDataset, collate_train_single_fn
import torch.utils
from torch.utils.data import DataLoader
import random
import os
from fgm import FGM
import torch.nn.functional as F


# 计算分布之间的KL散度
def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), torch.softmax(q, dim=-1))
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), torch.softmax(p, dim=-1))

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


# 随机以固定比例对文本进行mask
def mask_text(input_ids, tokenizer, vocab_size, device, targets=None, mlm_probability=0.15):
    # 根据probability_matrix中的概率来确定该位置是否mask， masked_indices中为True的就是需要mask
    probability_matrix = torch.ones_like(input_ids, device=device) * mlm_probability
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # padding和文本头部的cls token不需要进行mask
    masked_indices[input_ids == tokenizer.pad_token_id] = False
    masked_indices[input_ids == tokenizer.cls_token_id] = False
    masked_indices[input_ids == tokenizer.sep_token_id] = False

    if targets is not None:
        targets[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool().to(device) & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool().to(
        device) & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
    input_ids[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return input_ids, targets


# mask language modeling 预训练
def pretrain(model, dataloader, optimizer, scheduler, tokenizer, device, args, epoch):
    model.train()
    total_loss, total_contras_loss, total_classify_loss = 0, 0, 0
    scaler = GradScaler()
    print('vocab size: ', tokenizer.vocab_size)
    for step, batch in enumerate(dataloader):
        # labels = batch['labels'].to(device)
        category_ids = batch['category_ids'].to(device)
        input_ids = batch['input_ids'].squeeze(0).to(device)
        type_ids = batch['type_ids'].squeeze(0).to(device)
        masks = batch['masks'].squeeze(0).to(device)

        labels = input_ids.clone()
        masked_input_ids, targets = mask_text(input_ids, tokenizer, tokenizer.vocab_size, device, labels, mlm_probability=0.3)

        if args.use_fp16:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss= model(masked_input_ids, masks, type_ids, labels=targets, return_dict=True).loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(masked_input_ids, masks, type_ids, labels=targets, return_dict=True).loss

            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            # print(model.linear.weight)
            avg_loss = round(total_loss / (step + 1), 6)
            lr = optimizer.param_groups[0].get('lr')
            print(f'train epoch={epoch}, step={step}, lr={lr}, total_loss={avg_loss}')


# classification 多分类微调
def train(model, dataloader, optimizer, scheduler, tokenizer, device, args, epoch, momentum_model):
    model.train()
    total_loss, total_contras_loss, total_classify_loss = 0, 0, 0
    scaler = GradScaler()
    fgm = FGM(model)
    print('vocab size: ', tokenizer.vocab_size)
    for step, batch in enumerate(dataloader):
        category_ids = batch['category_ids'].to(device)
        input_ids = batch['input_ids'].squeeze(0).to(device)
        type_ids = batch['type_ids'].squeeze(0).to(device)
        masks = batch['masks'].squeeze(0).to(device)

        targets = torch.zeros((len(input_ids), 11)).to(device)
        targets = torch.scatter(targets, dim=1, index=category_ids.unsqueeze(1), value=1)

        # 混合精度训练
        if args.use_fp16:
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input_ids, masks, type_ids, labels=category_ids, return_dict=True)
                loss = output.loss

                # 使用rdrop
                if args.use_rdrop:
                    batch_size = len(input_ids)
                    double_input_ids = torch.cat([input_ids, input_ids], dim=0)
                    double_masks = torch.cat([masks, masks], dim=0)
                    double_type_ids = torch.cat([type_ids, type_ids], dim=0)
                    double_category_ids = torch.cat([category_ids, category_ids], dim=0)
                    rdrop_output = model(input_ids, masks, type_ids, return_dict=True)

                    # loss = output.loss
                    # logits = torch.softmax(output.logits, dim=-1)
                    kl_loss = compute_kl_loss(output.logits, rdrop_output.logits)

                    loss += kl_loss
                # else:
                #     loss = model(input_ids, masks, type_ids, labels=category_ids, return_dict=True).loss

            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        else:
            loss = model(input_ids, masks, type_ids, labels=category_ids, return_dict=True).loss
            loss.backward()
            # optimizer.step()

        # 使用对抗训练，在word_embeddings部分加入梯度的扰动
        if args.use_fgm:
            fgm.attack(emb_name='word_embeddings')
            if args.use_fp16:
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss_adv = model(input_ids, masks, type_ids, labels=category_ids, return_dict=True).loss
            else:
                loss_adv = model(input_ids, masks, type_ids, labels=category_ids, return_dict=True).loss

            if args.use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            fgm.restore(emb_name='word_embeddings')

        # 更新动量模型，EMA
        with torch.no_grad():
            momentum_update([model, momentum_model], args.momentum)

        if args.use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            avg_loss = round(total_loss / (step + 1), 6)
            lr = optimizer.param_groups[0].get('lr')
            print(f'train epoch={epoch}, step={step}, lr={lr}, total_loss={avg_loss}')


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
    parser.add_argument('--momentum', type=float, default=0.995)
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
    parser.add_argument('--use_fgm', action='store_true')
    parser.add_argument('--use_rdrop', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--pack_with_cate', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--pair_text', action='store_true')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--loss_func', type=str, default='bce')

    args = parser.parse_args()

    return args


def main(args):
    # init configuration
    fix_seed(args.seed)
    # random.seed(42)
    device = torch.device(args.device)

    # create dataset and dataloader
    with open(args.data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = [json.loads(l) for l in lines]

    category_list = sorted(list(set([d['category_name'] for d in data])))
    random.shuffle(data)
    random.shuffle(category_list)

    ## 根据类别划分训练集和验证集
    valid_category_list = category_list[:int(len(category_list) * args.valid_ratio)]
    print(f'Not train the following categories: {valid_category_list}')
    valid_data = [d for d in data if d['category_name'] in valid_category_list]
    train_data = [d for d in data if d['category_name'] not in valid_category_list]

    print(len(train_data), len(valid_data))

    train_dataset = RetrievalDataset(train_data, 'train', args.encoder_dir, max_len=args.max_len, preprocess=True)
    valid_dataset = RetrievalDataset(valid_data, 'valid', args.encoder_dir, max_len=args.max_len)
    print(f'train data size={len(train_dataset)}, valid data size={len(valid_dataset)}')

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_train_pair_fn, drop_last=args.queue_size > 0)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_train_single_fn, drop_last=args.queue_size > 0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=6)

    # create model
    # model = AutoModelForMaskedLM.from_pretrained(args.encoder_dir)
    config = AutoConfig.from_pretrained(args.encoder_dir)
    config.num_labels = 11
    config.problem_type = "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder_dir, config=config)
    # config = AutoConfig.from_pretrained(args.encoder_dir)
    model = model.to(device)

    # 构建当前模型的副本，用于保存动量模型
    momentum_model = AutoModelForSequenceClassification.from_pretrained(args.encoder_dir, config=config)
    copy_params([model, momentum_model])
    momentum_model = momentum_model.to(device)
    momentum_model.eval()

    optimizer, scheduler = creat_optimizer_and_scheduler(model, args, len(train_dataloader) * args.epochs)
    print(model.num_labels, config.num_labels)

    for epoch in range(args.epochs):
        # mlm 预训练
        # pretrain(model, train_dataloader, optimizer, scheduler, train_dataset.tokenizer, device, args, epoch)

        # 多分类 微调
        train(model, train_dataloader, optimizer, scheduler, train_dataset.tokenizer, device, args, epoch, momentum_model)

        os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
        save_path = os.path.join(args.output_dir, args.task_name, f'model_{epoch}.pt')
        torch.save(model.state_dict(), save_path)

        save_path = os.path.join(args.output_dir, args.task_name, f'model_{epoch}_ema.pt')
        torch.save(momentum_model.state_dict(), save_path)


if __name__ == '__main__':
    args = args_parser()
    main(args)

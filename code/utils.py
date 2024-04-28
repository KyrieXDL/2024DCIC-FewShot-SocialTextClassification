import numpy as np
import torch
import random
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup
import json
# from mydataset.topic_dataset_v2 import TopicDataset, collate_train_fn,collate_train_pair_fn, collate_train_single_fn
# from mydataset.pair_dataset import PairDataset, collate_train_fn
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import GroupKFold
import copy
import os

def fix_seed(seed):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def creat_optimizer_and_scheduler(model, args, total_steps):
    # create optimizer and scheduler
    group_params = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': args.pretrained_lr},
                    {'params': [p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad], 'lr': args.random_lr}]
    optimizer = torch.optim.AdamW(group_params, lr=args.random_lr, weight_decay=args.weight_decay)

    
    if args.schedule_type == 'warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                    num_training_steps=total_steps)
    elif args.schedule_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
            lr_end=0,
            power=1,
        )
    else:
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
        scheduler = None

    return optimizer, scheduler


def create_dataloader(args):
    with open(args.data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = [json.loads(l) for l in lines]

    if args.phase == 'train':
        all_dataloaders = []
        if not args.kfold:
            category_list = sorted(list(set([d['category_name'] for d in data])))
            random.shuffle(data)
            random.shuffle(category_list)

            valid_category_list = category_list[:int(len(category_list) * args.valid_ratio)]
            print(f'Not train the following categories: {valid_category_list}')
            valid_data = [d for d in data if d['category_name'] in valid_category_list]
            train_data = [d for d in data if d['category_name'] not in valid_category_list]

            print(len(train_data), len(valid_data))

            train_dataset = PairDataset(train_data, 'train', args.encoder_dir, max_len=args.max_len,
                                         pair_text=args.pair_text, preprocess=args.preprocess,
                                         pack_with_cate=args.pack_with_cate)
            valid_dataset = PairDataset(valid_data, 'valid', args.encoder_dir, max_len=args.max_len,
                                         pair_text=args.pair_text)
            print(f'train data size={len(train_dataset)}, valid data size={len(valid_dataset)}')

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_train_fn, drop_last=args.queue_size > 0)
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=6)

            all_dataloaders.append((train_dataloader, valid_dataloader))
        else:
            gkf = GroupKFold(n_splits=5)
            random.shuffle(data)
            groups = [d['category_name'] for d in data]
            for train_index, valid_index in gkf.split(data, groups=groups):
                # print(train_data)
                train_data = np.array(data)[train_index]
                valid_data = np.array(data)[valid_index]
                train_cates = list(set([d['category_name'] for d in train_data]))
                valid_cates = list(set([d['category_name'] for d in valid_data]))

                print(train_cates, valid_cates)

                train_dataset = PairDataset(train_data, 'train', args.encoder_dir, max_len=args.max_len,
                                             pair_text=args.pair_text, preprocess=args.preprocess,
                                             pack_with_cate=args.pack_with_cate)
                valid_dataset = PairDataset(valid_data, 'valid', args.encoder_dir, max_len=args.max_len,
                                             pair_text=args.pair_text)
                print(f'train data size={len(train_dataset)}, valid data size={len(valid_dataset)}')

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                              collate_fn=collate_train_fn, drop_last=args.queue_size > 0)
                valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=6)

                all_dataloaders.append((train_dataloader, valid_dataloader))

        return all_dataloaders
    else:
        test_dataset = TopicDataset(data, 'valid', args.encoder_dir, max_len=args.max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

        return test_dataloader


@torch.no_grad()
def momentum_update(model_pair, momentum):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


@torch.no_grad()
def copy_params(model_pair):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data.copy_(param.data)  # initialize
        param_m.requires_grad = False  # not update by gradient


def swa(model, model_dir, swa_start=1, swa_end=100):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = os.listdir(model_dir)
    model_path_list = [os.path.join(model_dir, f) for f in model_path_list if 'ema' not in f and f.endswith('.pt')]
    model_path_list = sorted(model_path_list)
    print(model_path_list)

    assert 0 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:swa_end]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            checkpoint = torch.load(_ckpt, map_location='cpu')

            new_checkpoint = {}
            for k, v in checkpoint.items():
                # print(k)
                if k.startswith('bert.'):
                    k = k.replace('bert.', '')
                new_checkpoint[k] = v

            msg = model.load_state_dict(new_checkpoint, strict=False)
            print(msg)
            # model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    return swa_model
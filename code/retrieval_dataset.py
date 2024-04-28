import time

import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import random
from transformers import AutoTokenizer
import json
import re
import string
import jieba


def end_with_punctuation(text):
    if text[-1] in string.punctuation:
        return True
    else:
        return False


def contain_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


class RetrievalDataset(Dataset):
    def __init__(self, data, mode='train', tokenizer_dir='', max_len=512, pack_with_cate=False, preprocess=False,
                 pair_text=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.cls_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        print(f'cls token = {self.cls_token}, cls token id = {self.cls_token_id}; '
              f'pad_token = {self.pad_token}, pad token id = {self.pad_token_id}; '
              f'sep_token = {self.sep_token}, sep token id = {self.sep_token_id}')
        self.pack_with_cate = pack_with_cate
        self.pair_text = pair_text
        self.category_list = ['环境污染', '医患纠纷', '诈骗曝光', '讨薪维权', '突发火情', '业主维权', '突发交通事故', '教育乱象', '校园霸凌', '拆迁维权']
        self.category_to_id = {cate: idx+1 for idx, cate in enumerate(self.category_list)}
        self.mode = mode
        self.max_len = max_len
        if self.mode == 'train':
            if preprocess:
                self.all_data, self.all_neg_data = self.process_data(data)
                # self.all_data += self.all_neg_data
            else:
                self.all_data = self.raw_data(data)
            self.support_data = self.get_support_data(data)
        else:
            self.all_data = data

    def process_data(self, data):
        all_data, all_text, all_neg_text, all_neg_data = [], [], [], []
        cate_to_desc = {}
        for item in data:
            category = item['category_name']
            desc = item['category_description']
            query_set = item['query_set']
            support_set = item['support_set']

            cate_to_desc[category] = desc

            for sample in query_set + support_set:
                ###
                if int(sample['label']) == 1 and sample['text'] not in all_text:
                    all_text.append(sample['text'])
                    all_data.append({'text': sample['text'],
                                          'category_id': self.category_to_id[category],
                                          'label': 1,
                                          'category': category,
                                          'desc': desc})

        for item in data:
            query_set = item['query_set']
            for sample in query_set:
                if int(sample['label']) == 0 and sample['text'] not in all_neg_text + all_text:
                    all_neg_text.append(sample['text'])
                    all_neg_data.append({'text': sample['text'],
                                          'category_id': 0,
                                          'label': 0,
                                          'category': 'null',
                                          'desc': 'null'})
        print(len(all_data), len(all_neg_data))
        # all_data += all_neg_data

        new_data = []
        if self.pack_with_cate:
            for d in all_data:
                for cate, desc in cate_to_desc.items():
                    d['category'] = cate
                    d['desc'] = desc
                    new_data.append(d)
        else:
            new_data = all_data

        return new_data, all_neg_data

    def raw_data(self, data):
        all_data = []
        for item in data:
            category = item['category_name']
            desc = item['category_description']
            query_set = item['query_set']
            support_set = item['support_set']

            for sample in query_set + support_set:
                ###
                all_data.append({'label': int(sample['label']) if 'label' in sample else 0,
                                 'desc': desc, 'category': category,
                                 'text': sample['text'],
                                 'category_id': self.category_to_id[category]
                                      })

        return all_data

    def get_support_data(self, data):
        support_data = {}
        for item in data:
            category = item['category_name']
            desc = item['category_description']
            query_set = item['query_set']
            support_set = item['support_set']

            if category not in support_data:
                support_data[category] = []

            for sample in query_set + support_set:
                if int(sample['label']) == 1 and sample['text'] not in support_data[category]:
                    support_data[category].append(sample['text'])

        return support_data

    def __len__(self):
        return len(self.all_data)

    def preprocess_text(self, text):
        result = re.findall(r'【标题】：(.*)【正文】：(.*)【封面_OCR】：(.*)【抽帧_OCR】：(.*)【语音转写】：(.*)', text)
        result = [r for r in result[0] if r != '']
        res = ''
        for r in result:
            if contain_chinese(r):
                if end_with_punctuation(r):
                    res += r
                else:
                    res += r + '。'
        res = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', res)
        # res = re.sub(r'[^\w\s,]', '', res)
        return res

    def get_single_sample(self, index):
        if self.mode == 'train':
            sample = self.all_data[index]
            category = sample['category']
            category_id = sample['category_id']
            label = sample['label']
            text = sample['text']
            desc = sample['desc']
            text = self.preprocess_text(text)

            # category_id = self.category_to_id[category]
            text_tokens = self.tokenizer.tokenize(text)
            cate_tokens = self.tokenizer.tokenize(category + '，' + desc)

            if not self.pack_with_cate:
                # [cls] query/support [sep]
                if len(text_tokens) > self.max_len - 2:
                    text_tokens = text_tokens[:(self.max_len - 2)]
                input_tokens = [self.cls_token] + text_tokens + [self.sep_token]
                type_ids = [0] * len(input_tokens)
            else:
                # [cls] cate + desc [sep] query/support [sep]
                if len(text_tokens) > self.max_len - 3 - len(cate_tokens):
                    text_tokens = text_tokens[:(self.max_len - 3 - len(cate_tokens))]
                else:
                    text_tokens = text_tokens
                input_tokens = [self.cls_token] + cate_tokens +[self.sep_token] + text_tokens + [self.sep_token]
                type_ids = [0] * (len(cate_tokens) + 2) + [1] * (len(text_tokens) + 1)

            # input_tokens = [self.cls_token] + text_tokens + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            masks = [1] * len(input_ids)

            item = {}
            item['input_ids'] = torch.LongTensor(input_ids)
            item['type_ids'] = torch.LongTensor(type_ids)
            item['masks'] = torch.LongTensor(masks)
            item['labels'] = label
            item['category_ids'] = category_id

            return item
        else:
            query_texts = [d['text'] for d in self.all_data[index]['query_set']]
            query_labels = [int(d['label']) if 'label' in d else 0 for d in self.all_data[index]['query_set']]
            record_ids = [int(d['record_id']) if 'record_id' in d else 0 for d in self.all_data[index]['query_set']]
            support_texts = [d['text'] for d in self.all_data[index]['support_set']]
            category_name = self.all_data[index]['category_name']
            task_id = self.all_data[index]['task_id']
            desc = self.all_data[index]['category_description']
            instruct = '为这个句子生成表示以用于检索相关文本：'

            # start = time.time()
            query_texts = [self.preprocess_text(text) for text in query_texts]
            support_texts = [self.preprocess_text(text) for text in support_texts]
            # end = time.time()
            # print('Preprocess cost time: ', end - start)

            # start = time.time()
            query_output = self.tokenizer(query_texts, max_length=self.max_len, truncation=True, padding='longest', add_special_tokens=True)
            support_output = self.tokenizer(support_texts, max_length=self.max_len, truncation=True, padding='longest', add_special_tokens=True)
            # end = time.time()
            # print('Tokenize cost time: ', end - start)

            query_input_ids = query_output['input_ids']
            query_masks = query_output['attention_mask']
            support_input_ids = support_output['input_ids']
            support_masks = support_output['attention_mask']

            # query_words = []
            # for text in query_texts:
            #     words = jieba.cut(text.strip().lower(), cut_all=False)
            #     query_words.append(words)
            #
            # support_words = []
            # for text in support_texts:
            #     words = jieba.cut(text.strip().lower(), cut_all=False)
            #     support_words.append(words)


            item = {}
            item['query_input_ids'] = torch.LongTensor(query_input_ids)
            # item['query_type_ids'] = torch.LongTensor(query_type_ids)
            item['query_masks'] = torch.LongTensor(query_masks)
            item['support_input_ids'] = torch.LongTensor(support_input_ids)
            item['support_masks'] = torch.LongTensor(support_masks)
            item['labels'] = torch.LongTensor(query_labels)
            item['record_ids'] = torch.LongTensor(record_ids)
            item['task_id'] = torch.LongTensor([task_id])
            item['category_name'] = category_name
            item['query_texts'] = query_texts
            item['support_texts'] = support_texts

            return item

    def __getitem__(self, index):
        return self.get_single_sample(index)


def collate_train_pair_fn(batch):
    pair_input_ids = [d['pair_input_ids'] for d in batch]
    pair_masks = [d['pair_masks'] for d in batch]
    pair_type_ids = [d['pair_type_ids'] for d in batch]
    labels = [d['labels'] for d in batch]
    category_ids = [d['category_ids'] for d in batch]

    max_sequence_len = max([len(input_id) for input_id in pair_input_ids])

    batch_input_ids, batch_masks, batch_type_ids = [], [], []
    for i in range(len(pair_input_ids)):
        cur_sequence_len = len(pair_input_ids[i])
        # padding input ids
        padding_len = max_sequence_len - cur_sequence_len

        cur_input_id = torch.cat([pair_input_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)
        cur_mask = torch.cat([pair_masks[i], torch.zeros((padding_len, ))], dim=0)
        cur_type_id = torch.cat([pair_type_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)

        batch_input_ids.append(cur_input_id)
        batch_masks.append(cur_mask)
        batch_type_ids.append(cur_type_id)


    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_masks = torch.stack(batch_masks, dim=0)
    batch_type_ids = torch.stack(batch_type_ids, dim=0)
    labels = torch.LongTensor(labels)
    category_ids = torch.LongTensor(category_ids)

    item = {}
    item['pair_input_ids'] = batch_input_ids
    item['pair_masks'] = batch_masks
    item['pair_type_ids'] = batch_type_ids
    item['labels'] = labels
    item['category_ids'] = category_ids

    return item


def collate_train_single_fn(batch):
    input_ids = [d['input_ids'] for d in batch]
    type_ids = [d['type_ids'] for d in batch]
    masks = [d['masks'] for d in batch]
    labels = [d['labels'] for d in batch]
    category_ids = [d['category_ids'] for d in batch]

    max_sequence_len = max([len(input_id) for input_id in input_ids])

    batch_input_ids, batch_masks, batch_type_ids = [], [], []
    for i in range(len(input_ids)):
        cur_sequence_len = len(input_ids[i])
        # padding input ids
        padding_len = max_sequence_len - cur_sequence_len

        cur_input_id = torch.cat([input_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)
        cur_type_id = torch.cat([type_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)
        cur_mask = torch.cat([masks[i], torch.zeros((padding_len, ))], dim=0)

        batch_input_ids.append(cur_input_id)
        batch_masks.append(cur_mask)
        batch_type_ids.append(cur_type_id)


    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_type_ids = torch.stack(batch_type_ids, dim=0)
    batch_masks = torch.stack(batch_masks, dim=0)
    labels = torch.LongTensor(labels)
    category_ids = torch.LongTensor(category_ids)

    item = {}
    item['input_ids'] = batch_input_ids
    item['type_ids'] = batch_type_ids
    item['masks'] = batch_masks
    item['labels'] = labels
    item['category_ids'] = category_ids

    return item


def collate_train_fn(batch):
    input_ids = [d['input_ids'] for d in batch]
    masks = [d['masks'] for d in batch]
    labels = [d['labels'] for d in batch]
    record_ids = [d['record_ids'] for d in batch] if 'record_ids' in batch[0] else []
    category_ids = [d['category_ids'] for d in batch] if 'category_ids' in batch[0] else []
    composed_input_ids = [d['composed_input_ids'] for d in batch]
    composed_masks = [d['composed_masks'] for d in batch]

    max_sequence_len = max([len(input_id) for input_id in input_ids])
    max_composed_sequence_len = max([len(composed_input_id) for composed_input_id in composed_input_ids])

    batch_input_ids, batch_masks = [], []
    batch_composed_input_ids, batch_composed_masks = [], []
    for i in range(len(input_ids)):
        cur_sequence_len = len(input_ids[i])
        # padding input ids
        padding_len = max_sequence_len - cur_sequence_len

        cur_input_id = torch.cat([input_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)
        cur_mask = torch.cat([masks[i], torch.zeros((padding_len, ))], dim=0)

        batch_input_ids.append(cur_input_id)
        batch_masks.append(cur_mask)

        # padding composed input ids
        cur_sequence_len = len(composed_input_ids[i])
        padding_len = max_composed_sequence_len - cur_sequence_len

        cur_composed_input_id = torch.cat([composed_input_ids[i], torch.zeros((padding_len, ), dtype=torch.long)], dim=0)
        cur_composed_mask = torch.cat([composed_masks[i], torch.zeros((padding_len, ))], dim=0)

        batch_composed_input_ids.append(cur_composed_input_id)
        batch_composed_masks.append(cur_composed_mask)

    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_masks = torch.stack(batch_masks, dim=0)
    labels = torch.LongTensor(labels)
    category_ids = torch.LongTensor(category_ids)
    batch_composed_input_ids = torch.stack(batch_composed_input_ids, dim=0)
    batch_composed_masks = torch.stack(batch_composed_masks, dim=0)

    item = {}
    item['input_ids'] = batch_input_ids
    item['masks'] = batch_masks
    item['labels'] = labels
    item['category_ids'] = category_ids
    item['record_ids'] = record_ids
    item['composed_input_ids'] = batch_composed_input_ids
    item['composed_masks'] = batch_composed_masks

    return item


if __name__ == '__main__':
    with open('../data/train_data.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    train_data = [json.loads(l) for l in lines]
    tokenizer = AutoTokenizer.from_pretrained('../pretrained_models/erlangshen-deberta-v2-97m-chinese')
    print(tokenizer.pad_token, tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
    # mydataset = TopicDataset(train_data, 'train', '../pretrained_models/erlangshen-deberta-v2-97m-chinese')
    # print(len(mydataset))

    # mydataset = TopicDataset(train_data, 'train', '../pretrained_models/erlangshen-deberta-v2-97m-chinese', pack_with_cate=False, preprocess=True)
    # print(len(mydataset))
    # dataloader = DataLoader(mydataset, batch_size=2, shuffle=True, collate_fn=collate_train_single_fn)
    #
    # for batch in dataloader:
    #     input_ids = batch['input_ids']
    #     masks = batch['masks']
    #     labels = batch['labels']
    #     print(input_ids.shape, masks.shape, labels.shape)
    #     break

    # mydataset = TopicDataset(train_data, 'train', '../pretrained_models/erlangshen-deberta-v2-97m-chinese',
    #                          pack_with_cate=False, preprocess=True, pair_text=True)
    # print(len(mydataset))
    # dataloader = DataLoader(mydataset, batch_size=10, shuffle=True, collate_fn=collate_train_single_fn)
    #
    # for batch in dataloader:
    #     input_ids = batch['input_ids']
    #     masks = batch['masks']
    #     labels = batch['labels']
    #     print(input_ids.shape, masks.shape, labels.shape)
    #     break

    mydataset = RetrievalDataset(train_data, 'valid', '../pretrained_models/erlangshen-deberta-v2-97m-chinese', pair_text=True)
    dataloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for batch in dataloader:

        query_input_ids = batch['query_input_ids']
        # query_type_ids = batch['query_type_ids']
        query_masks = batch['query_masks']
        # support_input_ids = batch['support_input_ids']
        # support_masks = batch['support_masks']
        labels = batch['labels']
        print(query_input_ids.shape, query_masks.shape, labels.shape)
        query_texts = [item[0] for item in batch['query_texts']]
        support_texts = [item[0] for item in batch['support_texts']]
        print(np.array(query_texts).shape, np.array(support_texts).shape)
        print(support_texts)
        print(query_texts)
        break

    # arr = list(range(16))
    # arr = torch.LongTensor(arr)
    # print(torch.max(arr.view(4, 4), dim=1)[0])
#

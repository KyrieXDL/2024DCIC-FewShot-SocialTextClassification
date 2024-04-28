from flask import Flask, request, jsonify
import json
import time
import pandas as pd

if __name__ == '__main__':
    import requests

    url = "http://0.0.0.0:8090/predict"
    header = {
        "aaaa": "0"
    }

    with open('../raw_data/test_data.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = [json.loads(l) for l in lines]
    predicts = []
    start = time.time()
    for item in data:
        res = requests.post(url=url, headers=header, json=item)
        predicts.append(res.text)

    end = time.time()

    labels = []
    record_ids = []
    for item in predicts:
        sample = json.loads(item)
        labels += [d['label'] for d in sample['data']]
        record_ids += [d['record_id'] for d in sample['data']]

    df = pd.DataFrame()
    df['record_id'] = record_ids
    df['label'] = labels
    df['label'] = df['label'].astype(int)
    df.to_csv('./sub.csv', index=False)

    print('Cost time: ', end - start)
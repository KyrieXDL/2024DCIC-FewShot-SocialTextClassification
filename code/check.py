import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('/home/admin02/projects/competitions/DCIC2024/output/sub658.csv')
    df2 = pd.read_csv('./sub.csv')

    print('Mismatch samples: ', df['label'] != df2['label'])
    print(np.sum(df['label'] != df2['label']))
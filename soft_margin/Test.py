import math

import numpy as np
import pandas as pd


def equal_split(x1,x2):
    # 将 x2 等分为 len(x1)/len(x2) 份

    ratio = math.ceil(len(x1)/math.floor(len(x1)/len(x2)))
    print(ratio)
    return np.array([x1[i:i + ratio] for i in range(0, len(x1), ratio)])

if __name__ == '__main__':
    data = pd.read_excel('00-22data_result.xls')
    pos = data[data['label'] == 1]
    neg = data[data['label'] == 0]
    data = equal_split(neg,pos)
    print(data)

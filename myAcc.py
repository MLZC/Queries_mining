import numpy as np
import pandas as pd


def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)

def ensAcc(results_path, test_set_path):
    res = pd.read_csv(results_path, encoding='utf8')
    res.sort_values('Id', inplace=True)
    res = res[['Age', 'Education', 'Gender']]
    act = pd.read_csv(test_set_path, encoding='utf8')
    act.sort_values('Id', inplace=True)
    act = act[['Age', 'Education', 'Gender']]
    acc = np.sum(np.array(res==act, dtype=np.int64), axis=0)/len(res)
    print("acc-Age:{}\t acc-Education:{}\t acc-Gender{}:".format(acc[0], acc[1], acc[2]))
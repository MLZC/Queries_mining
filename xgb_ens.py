import pandas as pd
import numpy as np
import xgboost as xgb
from data_preprocessing import load_csv
import conf.Education as Education
import conf.Age as Age
import conf.Gender as Gender

import datetime

data_path = './data/'
csv_path = "./data/all_data.csv"
length = 2000


def xgb_acc_score(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds, axis=1)
    return [('acc', np.mean(y_true == y_pred))]


def load_model_data():
    df_lr = pd.read_csv(data_path + 'tfidf_stack.csv')
    df_dm = pd.read_csv(data_path + 'dm_d2v_stack.csv')
    df_dbow = pd.read_csv(data_path + 'dbow_d2v_stack.csv')
    return df_lr, df_dm, df_dbow


def train(length):
    df_lr, df_dm, df_dbow = load_model_data()
    train_data, dic = load_csv(csv_path, length)
    # seed = 10

    TR = int(length / 2 * 0.9)
    df_sub = pd.DataFrame()
    df_sub['Id'] = train_data.iloc[TR:]['Id']
    df = pd.concat([df_lr, df_dbow, df_dm], axis=1)
    print("----" * 5 + "Training xgb-ens start" + "----" * 5)
    print(df.columns)
    for lb in ['Education', 'Age', 'Gender']:
        print("-----" * 5 + lb + "-----" * 5)
        num_class = len(pd.value_counts(dic[lb]))
        X = df.iloc[:TR]
        y = dic[lb][:TR]
        X_te = df.iloc[TR:]
        y_te = dic[lb][TR:]

        esr = 100
        evals = 1
        n_trees = 10

        lb_2_model = eval(lb)
        params = lb_2_model.params
        params['num_class'] = num_class

        dtrain = xgb.DMatrix(X, y)
        dvalid = xgb.DMatrix(X_te, y_te)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                        early_stopping_rounds=esr, verbose_eval=evals)
        df_sub[lb] = np.argmax(bst.predict(dvalid), axis=1) + 1
    df_sub = df_sub[['Age', 'Education', 'Gender', 'Id']]
    df_sub.to_csv(data_path + 'tfidf_dm_dbow_.csv', index=None, header=None, sep=' ')
    print("----" * 5 + "Training xgb-ens finished" + "----" * 5)

if __name__ == '__main__':
    train(length)

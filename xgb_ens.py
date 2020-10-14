import pandas as pd
import numpy as np
import xgboost as xgb
from data_separate import load_csv
import conf.Education as Education
import conf.Age as Age
import conf.Gender as Gender
from myAcc import ensAcc
import datetime

data_path = './data/'
test_set_path = './data/test_data.csv'

def xgb_acc_score(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds, axis=1)
    return [('acc', np.mean(y_true == y_pred))]


def load_model_data():
    df_lr = pd.read_csv(data_path + 'tfidf_stack.csv')
    df_dm = pd.read_csv(data_path + 'dm_d2v_stack.csv')
    df_dbow = pd.read_csv(data_path + 'dbow_d2v_stack.csv')
    return df_lr, df_dm, df_dbow


def train(train_test_data_path, length):
    """
    length: length of train data
    """
    df_lr, df_dm, df_dbow = load_model_data()
    data, dic = load_csv(train_test_data_path)
    # seed = 10

    TR = length
    df_sub = pd.DataFrame()
    df_sub['Id'] = data.iloc[TR:]['Id']
    df = pd.concat([df_lr, df_dbow, df_dm], axis=1)
    print("----" * 5 + "Training xgb-ens start" + "----" * 5)
    print(df.columns)
    for lb in ['Education', 'Age', 'Gender']:
    # for lb in ['Gender']:
        print("-----" * 5 + lb + "-----" * 5)
        num_class = len(pd.value_counts(dic[lb][:length]))
        X = df.iloc[:TR]
        y = dic[lb][:TR]
        X_te = df.iloc[TR:]
        y_te = dic[lb][TR:]
        print('{} train value_counts'.format(lb))
        print(pd.value_counts(dic[lb][:length]))
        print('{} test value_counts'.format(lb))
        print(pd.value_counts(dic[lb][length:]))
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
        df_sub[lb] = np.argmax(bst.predict(dvalid), axis=1)
    df_sub = df_sub[['Age', 'Education', 'Gender', 'Id']]
    df_sub.columns = ['Age', 'Education', 'Gender', 'Id']
    results_path = data_path + 'tfidf_dm_dbow_.csv'
    df_sub.to_csv(results_path, index=None, encoding='utf8')
    ensAcc(results_path, test_set_path)
    print("----" * 5 + "Training xgb-ens finished" + "----" * 5)

if __name__ == '__main__':
    csv_path = "./data/train_test_data.csv"
    length = 1600
    train(csv_path, length)
    # pass

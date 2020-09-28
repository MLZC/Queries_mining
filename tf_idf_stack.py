import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from datetime import datetime
from data_separate import load_csv
from myAcc import myAcc
import codecs

data_path = "./data/"


def train_tfidf_stack(csv_path, length):
    """
    csv_path: train_test_data_path
    length: the length of train_data
    """
    print(datetime.now(), 'training tfidf stack start!')
    train_test_data, dic = load_csv(csv_path)
    # tfv = TfidfVectorizer(tokenizer=Tokenizer(len(train_test_data)), min_df=3, max_df=0.95, sublinear_tf=True)
    # X = tfv.fit_transform(train_test_data['Query'])
    with codecs.open(data_path + 'tfidf_train_test.pkl', 'rb') as f:
        X = pickle.load(f)  # load tfidf matrix from tfidf_train_test.pkl
    df_stack = pd.DataFrame(index=range(len(train_test_data)))
    # -----------------------stacking for education/age/gender------------------
    # ['Education', 'Age','Gender']
    for i in ['Education', 'Age', 'Gender']:
        print(i)
        TR = length
        # print(train_test_data.iloc[:TR][i].value_counts())
        # print(train_test_data.iloc[TR:][i].value_counts())
        num_class = len(pd.value_counts(dic[i]))
        n = 5

        X_tr = X[:TR]
        y_tr = dic[i][:TR]
        X_te = X[TR:]
        y_te = dic[i][TR:]

        stack = np.zeros((X_tr.shape[0], num_class))
        stack_te = np.zeros((X_te.shape[0], num_class))

        for j, (tr, va) in enumerate(KFold(len(y_tr), n_folds=n)):
            print('%s stack:%d/%d' % (str(datetime.now()), j + 1, n))
            # print(train_test_data.iloc[tr][i].value_counts())
            # print(train_test_data.iloc[va][i].value_counts())
            clf = LogisticRegression(C=3)
            clf.fit(X_tr[tr], y_tr[tr])
            y_pred_va = clf.predict_proba(X_tr[va])
            y_pred_te = clf.predict_proba(X_te)
            print('va acc:', myAcc(y_tr[va], y_pred_va))
            print('te acc:', myAcc(y_te, y_pred_te))
            stack[va] += y_pred_va
            stack_te += y_pred_te
        stack_te /= n
        stack_all = np.vstack([stack, stack_te])
        for k in range(stack_all.shape[1]):
            df_stack['tfidf_{}_{}'.format(i, k)] = stack_all[:, k]
    df_stack.to_csv(data_path + 'tfidf_stack.csv', index=None, encoding='utf8')
    print(datetime.now(), 'training tfidf stack done!')


if __name__ == '__main__':
    train_test_data_path = "./data/train_test_data.csv"
    length = 1600
    train_tfidf_stack(train_test_data_path, length)

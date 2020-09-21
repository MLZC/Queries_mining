import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from datetime import datetime
from data_preprocessing import Tokenizer, load_csv
from myAcc import myAcc
import codecs

data_path = "./data/"
csv_path = "./data/all_data.csv"
length = 2000


def train_tfidf_stack(length):
    print(datetime.now(), 'training tfidf stack start!')
    train_data, dic = load_csv(csv_path, length)
    # tfv = TfidfVectorizer(tokenizer=Tokenizer(len(train_data)), min_df=3, max_df=0.95, sublinear_tf=True)
    # X = tfv.fit_transform(train_data['Query'])
    with codecs.open(data_path + 'tfidf.pkl', 'rb') as f:
        X = pickle.load(f)  # Loading X from 'tfidf.pkl'
    df_stack = pd.DataFrame(index=range(len(train_data)))
    # -----------------------stacking for education/age/gender------------------
    for i in ['Education', 'Age', 'Gender']:
        print(i)
        TR = int(length / 2 * 0.9)
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
    train_tfidf_stack(length)

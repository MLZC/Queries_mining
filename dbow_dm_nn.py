from myAcc import myAcc
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import KFold
from gensim.models import Doc2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from data_preprocessing import load_csv

data_path = "./data/"
csv_path = "./data/all_data.csv"
length = 2000


def train_dbow_dm_nn(feat: str, length: int):
    """
    :param feat: str ['dbow_d2v'.'dm_d2v']
    :return: none
    """
    print(datetime.now(), 'training ' + feat + ' stack start!')
    train_data, dic = load_csv(csv_path, length)
    model = Doc2Vec.load(data_path + feat + '.model')
    doc_vec = np.array([model.docvecs[i] for i in range(int(length / 2))])
    df_stack = pd.DataFrame(index=range(len(train_data)))
    TR = int(length / 2 * 0.9)
    n = 5
    X_tr = doc_vec[:TR]
    X_te = doc_vec[TR:]
    for i, lb in enumerate(['Education', 'Age', 'Gender']):
        num_class = len(pd.value_counts(dic[lb]))
        y_tr = dic[lb][:TR]
        y_te = dic[lb][TR:]

        stack = np.zeros((X_tr.shape[0], num_class))
        stack_te = np.zeros((X_te.shape[0], num_class))

        for k, (tr, va) in enumerate(KFold(len(y_tr), n_folds=n)):
            print('{} stack:{}/{} {}'.format(datetime.now(), k + 1, n, lb))
            nb_classes = num_class
            X_train = X_tr[tr]
            y_train = y_tr[tr].astype(np.int)
            X_test = X_te
            y_test = y_te.astype(np.int)

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            Y_test = np_utils.to_categorical(y_test, nb_classes)

            model = Sequential()
            model.add(Dense(300, input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.1))
            model.add(Activation('tanh'))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adadelta',
                          metrics=['accuracy'])

            history = model.fit(X_train, Y_train, shuffle=True,
                                batch_size=128, nb_epoch=35,
                                verbose=2, validation_data=(X_test, Y_test))
            y_pred_va = model.predict_proba(X_tr[va])
            y_pred_te = model.predict_proba(X_te)
            print('va acc:', myAcc(y_tr[va], y_pred_va))
            print('te acc:', myAcc(y_te, y_pred_te))
            stack[va] += y_pred_va
            stack_te += y_pred_te
        stack_te /= n
        stack_all = np.vstack([stack, stack_te])
        for l in range(stack_all.shape[1]):
            df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
    df_stack.to_csv(data_path + feat + '_stack.csv', encoding='utf8', index=None)
    print(datetime.now(), 'training ' + feat + ' stack done!')


if __name__ == '__main__':
    train_dbow_dm_nn('dbow_d2v', length)
    train_dbow_dm_nn('dm_d2v', length)

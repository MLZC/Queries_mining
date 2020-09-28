import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from collections import namedtuple
from gensim.models import Doc2Vec
from datetime import datetime
from sklearn.model_selection import cross_val_score
import pickle
from data_separate import load_csv

data_path = "./data/"


def train(train_data_path, test_data_path):
    """
    return: len(train_data), train_test_data_path
    """
    start = datetime.now()
    print(start, "Start training dm and dbow model!")
    add_number_to_query(train_data_path)
    train_data, _ = load_csv(train_data_path)
    X_train = tokenize_data(train_data)
    tfidf_matrix_dump(X_train, 'tfidf_train.pkl')

    new_train_data_path = fill_train_data(train_data, X_train)
    length, train_test_data_path, train_test_data = concat_train_test(new_train_data_path, test_data_path)
    add_number_to_query(train_test_data_path)
    X_train_test = tokenize_data(train_test_data)
    tfidf_matrix_dump(X_train_test, 'tfidf_train_test.pkl')

    train_doc2vec(train_test_data_path, length, 'dbow')
    train_doc2vec(train_test_data_path, length, 'dm')
    end = datetime.now()
    print(end, "Train dm and dbow model down! Duration time: {}s ".format((end - start).seconds))
    return length, train_test_data_path


def add_number_to_query(csv_path):
    """
    :param csv_path: train_data_path
    :return: the total number of train_data.
    """
    print("----" * 5 + "Add number to query: Start" + "----" * 5)
    train_data, _ = load_csv(csv_path)
    f = codecs.open(csv_path[:-4] + '_num.txt', 'w', encoding='utf8')
    for i, queries in enumerate(train_data.iloc[:len(train_data)]['Query']):
        words = []
        for query in queries.split('\t'):
            words.extend(list(jieba.cut(query)))
        f.write('_*{} {}'.format(i, ' '.join(words)))
    f.close()
    print("----" * 5 + "Add number to query: Done" + "----" * 5)
    return len(train_data)


def fill_train_data(train_data, X):
    """
    train_data: train_data
    X: Tf-idf-weighted document-term matrix. [n_samples, n_features]
    return: new_train_data_path
    """
    print("Fill train data start", datetime.now())
    for i, j in [('Age', 0), ('Education', 1), ('Gender', 2)]:
        _, _ = fill_missing_value(i, j, train_data, X)
    new_train_data_path = data_path + "new_train_data.csv"
    train_data.to_csv(new_train_data_path, index=None, encoding='utf8')
    print("Fill train data done, the new train_data save at {}".format(new_train_data_path))
    return new_train_data_path


def fill_missing_value(category: str, idx: int, train_data, X):
    """
    Only for Train data.
    The value 0 means is a missing value.
    category: ['Age','Education','Gender']
    idx: [0,1,2]
    X: Tf-idf-weighted document-term matrix. [n_samples, n_features]
    """
    normal_data_idx = np.where(train_data[category] != -1)[0]
    missing_data_idx = np.where(train_data[category] == -1)[0]
    # C: float, default=1.0 Inverse of regularization strength;
    # smaller values specify stronger regularization.
    c = 1
    if idx != 1:
        c = 2
    train_data.iloc[missing_data_idx, idx] = LogisticRegression(C=c).fit(X[normal_data_idx],
                                                                         train_data.iloc[normal_data_idx, idx]).predict(
        X[missing_data_idx])
    return normal_data_idx, missing_data_idx


def concat_train_test(train: str, test: str):
    """
    parameter: train, test - path of data
    return: len(train),train_test_data_path, train_test_data
    """
    train_data, _ = load_csv(train)
    test_data, _ = load_csv(test)
    train_test_data = pd.concat([train_data, test_data])
    train_test_data_path = data_path + 'train_test_data.csv'
    train_test_data.to_csv(train_test_data_path, index=None, encoding='utf8')
    return len(train_data), train_test_data_path, train_test_data


def tokenize_data(train_data):
    """
    train_data: dataFrame
    return: X: [n_samples, n_features] matrix
    """
    tfv = TfidfVectorizer(tokenizer=Tokenizer(len(train_data)), min_df=3, max_df=0.95, sublinear_tf=True)
    print("------" * 5 + "   Tokenize Query   " + "------" * 5)
    X = tfv.fit_transform(train_data['Query'])
    print("------" * 5 + " Tokenize  Finished " + "------" * 5)
    print("Vocabulary length: ", len(tfv.vocabulary_))
    # X: Tf-idf-weighted document-term matrix. [n_samples, n_features]
    print("The shape of Tf-idf-weighted document-term matrix:", X.shape)
    return X


def tfidf_matrix_dump(X, var_name):
    """
    X: tfidf matrix
    var_name: file name without path
    return: path+filename
    """
    with codecs.open(data_path + var_name, 'wb') as f:
        pickle.dump(X, f)  # save time for next reading this matrix


def train_doc2vec(csv_path, length, model_type='dbow'):
    """
    :param csv_path: path of train data with .csv format
    :param model_type: 'dbow' or 'dm'
    :param length: the length of train data
    :return: dbow_d2v.model in ./data/
    """
    d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025,
                  min_alpha=0.025)
    epoch = 2
    if model_type == 'dm':
        epoch = 10
        d2v = Doc2Vec(dm=1, size=300, negative=5, hs=0, min_count=3, window=10, sample=1e-5, workers=8, alpha=0.05,
                      min_alpha=0.025)
    doc_list = DocList(csv_path[:-4] + '_num.txt')
    d2v.build_vocab(doc_list)
    _, dic = load_csv(csv_path)
    print(datetime.now(), model_type + ' model training!')
    for i in range(epoch):
        print(datetime.now(), 'pass: {}/{}'.format(i + 1, epoch))
        doc_list = DocList(csv_path[:-4] + '_num.txt')
        d2v.train(doc_list)
        X_d2v = np.array([d2v.docvecs[i] for i in range(length)])
        for j in ["Education", 'Age', 'Gender']:
            scores = cross_val_score(LogisticRegression(C=3), X_d2v, dic[j][:length], cv=5)
            print(model_type, j, scores, np.mean(scores))
    d2v.save(data_path + model_type + '_d2v.model')
    print(datetime.now(), model_type + ' model save done!')


class Tokenizer(object):
    def __init__(self, length):
        self.n = 0
        self.length = length

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        self.n += 1
        if self.n % int(self.length / 10) == 0:
            print(self.n)
        return tokens


class DocList(object):
    def __init__(self, f):
        self.f = f
        self.SentimentDocument = namedtuple('SentimentDocument', 'words tags')

    def __iter__(self):
        for _, line in enumerate(codecs.open(self.f, encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            yield self.SentimentDocument(words, tags)


if __name__ == "__main__":
    # train_data_path = './data/train_data.csv'
    # test_data_path = './data/test_data.csv'
    # start = datetime.now()
    # print("Start training dm and dbow model!")
    # add_number_to_query(train_data_path)
    # train_data, _ = load_csv(train_data_path)
    # X_train = tokenize_data(train_data)
    # tfidf_matrix_dump(X_train, 'tfidf_train.pkl')
    #
    # new_train_data_path = fill_train_data(train_data, X_train)
    # length, train_test_data_path, train_test_data = concat_train_test(new_train_data_path, test_data_path)
    # add_number_to_query(train_test_data_path)
    # X_train_test = tokenize_data(train_test_data)
    # tfidf_matrix_dump(X_train_test, 'tfidf_train_test.pkl')
    #
    # train_doc2vec(train_test_data_path, length, 'dbow')
    # train_doc2vec(train_test_data_path, length, 'dm')
    # end = datetime.now()
    # print("Train dm and dbow model down! Duration time: {}s ".format((end - start).seconds))
    pass

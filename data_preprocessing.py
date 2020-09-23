import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import subprocess
from collections import namedtuple
from gensim.models import Doc2Vec
from datetime import datetime
from sklearn.model_selection import cross_val_score
import pickle


data_path = "./data/"


def to_csv(flag):
    """
    :param flag: 1--mini batch 0--whole set
    :return: path of csv
    """
    """1. Loading data"""
    train_set = "user_tag_query.10W.TRAIN"
    test_set = "user_tag_query.10W.TEST"
    data_set_path = data_path + train_set
    data_loader = Loader(flag)
    train_data, n = data_loader.data_loader(data_set_path)
    ## This step is very important
    for lb in ['Education', 'Age', 'Gender']:
        train_data[lb] = train_data[lb] - 1
        print(train_data.iloc[:n][lb].value_counts())
    data_set_path = data_path + test_set
    test_data, _ = data_loader.data_loader(data_set_path, False)
    # Loader.count_values(train_data)
    '''2. tokenize data'''
    tfv = TfidfVectorizer(tokenizer=Tokenizer(len(train_data)), min_df=3, max_df=0.95, sublinear_tf=True)
    print("------" * 5 + "   Tokenize Query   " + "------" * 5)
    X = tfv.fit_transform(train_data['Query'])
    with codecs.open(data_path + 'tfidf.pkl', 'wb') as f:
        pickle.dump(X, f)  # save time for next reading this matrix
    print("------" * 5 + " Tokenize  Finished " + "------" * 5)
    print("Vocabulary length: ", len(tfv.vocabulary_))
    # X: Tf-idf-weighted document-term matrix. [n_samples, n_features]
    print("The shape of Tf-idf-weighted document-term matrix:", X.shape)
    """3. fill NA data"""
    print("----" * 6 + " Fill NA Data " + "----" * 6)
    for i, j in [('Age', 0), ('Education', 1), ('Gender', 2)]:
        fill_data(i, j, train_data, X)
    # Loader.count_values(train_data)
    """4. concat train and test data"""
    all_data = pd.concat([train_data, test_data]).fillna(0)
    print(all_data.shape)
    all_data.to_csv(data_path + 'all_data.csv', index=None, encoding='utf8')
    return data_path + "all_data.csv"


def add_number_to_query(csv_path):
    """
    :param csv_path:
    :return: the total number of data - include train(50%) and test(50%).
    """
    print("----" * 5 + "add number to query: Start" + "----" * 5)
    all_data = pd.read_csv(csv_path, encoding='utf8')
    f = codecs.open('all_data_num.txt', 'w', encoding='utf8')
    for i, queries in enumerate(all_data.iloc[:int(len(all_data) / 2)]['Query']):
        words = []
        for query in queries.split('\t'):
            words.extend(list(jieba.cut(query)))
        f.write('_*{} {}'.format(i, ' '.join(words)))
    f.close()
    print("----" * 5 + "add number to query: Done" + "----" * 5)
    return len(all_data)


def fill_data(category: str, idx: int, train_data, X):
    """
    Only for Train data.
    The value 0 means is a missing value.
    category: ['Age','Education','Gender']
    idx: [0,1,2]
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


def count_values(data):
    for i in ['Age', 'Education', 'Gender']:
        print(data.iloc[:1000][i].value_counts())


def run_cmd(cmd: str):
    """
    :param cmd: str, command
    :return: ==0 -- subprocess exit normally
    """
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    for t, line in enumerate(iter(process.stdout.readline, b'')):
        line = line.decode('utf8').rstrip()
        print(line)
    process.communicate()
    return process.returncode


def load_csv(csv_path, n):
    """
    :param csv_path:
    :param n: total number lines of csv half train half test
    :return: train_data, dic
    """
    train_data = pd.read_csv(csv_path, encoding='utf8', nrows=int(n / 2))
    dic = {}
    for i in ['Education', 'Age', 'Gender']:
        dic[i] = np.array(train_data[i])
    return train_data, dic


def train_doc2vec(csv_path, n, model_type='dbow'):
    """
    :param csv_path: path of train data with .csv format
    :param n: total number of dataset train and test each half
    :param model_type: 'dbow' or 'dm'
    :return: dbow_d2v.model in ./data/
    """
    d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025,
                  min_alpha=0.025)
    epoch = 2
    if model_type == 'dm':
        epoch = 10
        d2v = Doc2Vec(dm=1, size=300, negative=5, hs=0, min_count=3, window=10, sample=1e-5, workers=8, alpha=0.05,
                      min_alpha=0.025)
    doc_list = DocList('all_data_num.txt')
    d2v.build_vocab(doc_list)
    train_data, dic = load_csv(csv_path, n)
    for i in range(epoch):
        print(datetime.now(), 'pass:', i)
        # run_cmd('shuf all_data_num.txt > all_data_num_shuf.txt')
        doc_list = DocList('all_data_num.txt')
        d2v.train(doc_list)
        X_d2v = np.array([d2v.docvecs[i] for i in range(int(n / 2))])
        for j in ["Education", 'Age', 'Gender']:
            scores = cross_val_score(LogisticRegression(C=3), X_d2v, dic[j], cv=5)
            print(model_type, j, scores, np.mean(scores))
    d2v.save(data_path + model_type + '_d2v.model')
    print(datetime.now(), 'save done')


def processing(flag):
    """
    :param flag: 1--mini_batch 0--whole set
    :return: the total number lines in csv half train and half test
    """
    # 1. origin data to csv
    all_data_path = to_csv(flag)
    # 2. add number to query
    n = add_number_to_query(all_data_path)
    # 3. train doc2vec model
    train_doc2vec(all_data_path, n, 'dbow')
    train_doc2vec(all_data_path, n, 'dm')
    print("----" * 5 + "Model dbow and dm Training completed" + "----" * 5)
    print("------" * 5 + "Saving in" + data_path + "------" * 5)
    return n


class Loader(object):
    def __init__(self, mini_dataSets=True):
        """
        mini_dataSets: 
            True - take the first 1000 lines from the origin dataSet
            False - Using the whole datasets
        """
        self.mini_dataSets = mini_dataSets

    def data_loader(self, data_set_path, Is_train_data=True):
        """
        data_set_path: full path of data
        """
        data = []
        print("----" * 5 + " opening " + data_set_path + "----" * 5)
        with open(data_set_path, encoding='GB18030') as f:
            for i, line in enumerate(f):
                if self.mini_dataSets:
                    if i > 999:
                        break
                segs = line.split('\t')
                row = {'Id': segs[0]}
                if Is_train_data:
                    row['Age'] = int(segs[1])
                    row['Gender'] = int(segs[2])
                    row['Education'] = int(segs[3])
                    row['Query'] = '\t'.join(segs[4:])
                else:
                    row['Query'] = '\t'.join(segs[1:])
                data.append(row)
        df_data = pd.DataFrame(data)
        print("---" * 5 + " loading " + data_set_path + " completed" + "---" * 5)
        return df_data, i


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
        for i, line in enumerate(codecs.open(self.f, encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            yield self.SentimentDocument(words, tags)


if __name__ == "__main__":
    start = datetime.now()
    n = processing(0)  # 1--mini batch 0--whole set
    end = datetime.now()
    print("Duration time: ", (end - start).seconds)
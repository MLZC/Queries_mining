import numpy as np
import pandas as pd
from datetime import datetime

data_path = "./data/"


def separate(flag):
    """
    flag: 1-mini batch 0-whole set
    return: train_data_path, test_data_path
    """
    start = datetime.now()
    print(start, "Start loading data and save as csv!")
    origin_data_path = origin_to_csv(flag)  # 1- mini batch(2000), 0-whole set(10w)
    origin_data, _ = load_csv(origin_data_path)
    train_data_path, test_data_path = separate_data(origin_data)
    end = datetime.now()
    print(end, "Save done! Duration time: {}s ".format((end - start).seconds))
    return train_data_path, test_data_path


def origin_to_csv(flag):
    """
    :param flag: 1--mini batch 0--whole set
    :return: path of csv
    """
    """1. Loading data"""
    data_name = "user_tag_query.10W.TRAIN"
    data_set_path = data_path + data_name
    data_loader = Loader(flag)
    origin_data, _ = data_loader.data_loader(data_set_path)
    # This step is very important
    for lb in ['Education', 'Age', 'Gender']:
        origin_data[lb] = origin_data[lb] - 1
        print(origin_data.iloc[:len(origin_data)][lb].value_counts())
    origin_data.to_csv(data_path + 'origin_data.csv', index=None, encoding='utf8')
    origin_data_path = data_path + "origin_data.csv"
    print("Origin data to csv done! csv_path={}".format(origin_data_path))
    return origin_data_path


def separate_data(all_data):
    """
    :param all_data: dataFrame
    return: train_data_path, test_data_path
    """
    print("Separate data to train and test: start!")
    all_normal_data_idx = \
        np.where((all_data['Age'] != -1) & (all_data['Education'] != -1) & (all_data['Gender'] != -1))[0]
    all_na_value_idx = np.where((all_data['Age'] == -1) & (all_data['Education'] == -1) & (all_data['Gender'] == -1))[0]
    # difference set
    temp = np.setdiff1d(np.arange(len(all_data)), all_normal_data_idx, assume_unique=True)
    # if a data miss these three value at same time, then it's a non-value data
    normal_data_idx = np.setdiff1d(temp, all_na_value_idx, assume_unique=True)
    # 80% train 20% test
    t = int(len(all_normal_data_idx) - len(all_data) * 0.2)
    test_data_idx = all_normal_data_idx[t:]
    train_data_idx = np.sort(np.hstack((normal_data_idx, all_normal_data_idx[:t])))
    test_data = all_data.iloc[test_data_idx]
    train_data = all_data.iloc[train_data_idx]
    test_data_path = data_path + 'test_data.csv'
    test_data.to_csv(test_data_path, index=None, encoding='utf8')
    print("test_data:{}, save at {} ".format(test_data.shape, test_data_path))
    train_data_path = data_path + 'train_data.csv'
    train_data.to_csv(train_data_path, index=None, encoding='utf8')
    print("test_data:{}, save at {} ".format(train_data.shape, train_data_path))
    print("Separate data to train and test: done!")
    return train_data_path, test_data_path


def load_csv(csv_path):
    """
    :param csv_path: csv_path
    :return: train_data: dataFrame, dic: dictionary
    """
    train_data = pd.read_csv(csv_path, encoding='utf8')
    dic = {}
    for i in ['Education', 'Age', 'Gender']:
        dic[i] = np.array(train_data[i])
    return train_data, dic


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
                    if i > 1999:
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
        df_data = pd.DataFrame(data)[['Age','Education','Gender','Id','Query']]
        print("---" * 5 + " loading " + data_set_path + " completed" + "---" * 5)
        return df_data, i


if __name__ == "__main__":
    start = datetime.now()
    print("Start loading data and save as csv!")
    csv_path = origin_to_csv(1)  # 1- mini batch(2000), 0-whole set(10w)
    data, _ = load_csv(csv_path)
    _, _ = separate_data(data)
    end = datetime.now()
    print("Save done! Duration time: {}s ".format((end - start).seconds))
    # pass

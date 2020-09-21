import data_preprocessing
import tf_idf_stack
import dbow_dm_nn
import xgb_ens
import datetime

if __name__ == '__main__':
    start = datetime.datetime.now()
    # 1. data preprocessing
    n = data_preprocessing.processing(1)  # 1 mini batch,0 whole set
    # 2. tf-idf slcaking
    tf_idf_stack.train_tfidf_stack(n)
    # 3. train dbow dm nn
    dbow_dm_nn.train_dbow_dm_nn('dbow_d2v', n)
    dbow_dm_nn.train_dbow_dm_nn('dm_d2v', n)
    # 4. train xgboost ensemble
    xgb_ens.train(n)
    end = datetime.datetime.now()
    print("Duration time: ", (end - start).seconds)

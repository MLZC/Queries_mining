import data_separate
import train_dm_dbow
import tf_idf_stack
import dbow_dm_nn
import xgb_ens
import datetime
import sys

if __name__ == '__main__':
    start = datetime.datetime.now()
    flag = int(sys.argv[1])
    if flag:
        print("Run on the mini dataset")
    else:
        print("Run on the whole set")
    # 1. data separating
    train_data_path, test_data_path, length = data_separate.separate(flag)  # 1 mini batch,0 whole set
    # 2. train dm dbow and fill NA data in train_data
    _, train_test_data_path = train_dm_dbow.train(train_data_path, test_data_path)
    # 3. tf-idf slcaking
    tf_idf_stack.train_tfidf_stack(train_test_data_path, length)
    # 4. train dbow dm nn
    dbow_dm_nn.train_dbow_dm_nn('dbow_d2v', length)
    dbow_dm_nn.train_dbow_dm_nn('dm_d2v', length)
    # 5. train xgboost ensemble
    xgb_ens.train(train_test_data_path, length)
    end = datetime.datetime.now()
    print("Duration time: ", (end - start).seconds)
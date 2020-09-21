import  numpy as np
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)
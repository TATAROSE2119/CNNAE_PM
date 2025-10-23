from scipy.io import loadmat
import numpy as np

def tep_data_load(mat_path,var_name,):
    data=loadmat(mat_path)
    X=data[var_name]
    print("原始形状：", X.shape)
    #筛选data中的变量
    cols_to_delete = [45, 49] + list(range(52, X.shape[1]))
    X_new = np.delete(X, cols_to_delete, axis=1)
    print("删除后形状：", X_new.shape)

    return X_new


if __name__ == "__main__":
    X=tep_data_load('TE_data/M1/m1d00.mat','m1d00')

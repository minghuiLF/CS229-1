import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    dfx = pd.read_table('data/logistic_x.txt',delim_whitespace=True, header=None)
    dfy = pd.read_table('data/logistic_y.txt', header=None)
    dfy.astype(int)
    x = dfx.values
    y = dfy.values
    m = len(x)
    plt.title('Original Data')
    for i in range(m):
        if y[i] == 1:
            plt.scatter(x[i][0], x[i][1], color='red')
        else:
            plt.scatter(x[i][0], x[i][1], color='blue')
    plt.show()
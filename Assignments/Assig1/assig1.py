import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def LogisticRegression():
    '''
    Perform Logistic Regression with Newton's Method
    '''

    def fun1(y_k, theta, x_k):
        '''
        :param y_k:
        :param theta:
        :param x_k:
        :return: calculate the bottom part of the formula
        '''
        return 1.0 + math.exp(y_k * theta * x_k.T)
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

    theta = np.zeros(3)

    # working part
    delta = 1e9
    iter = 0
    while delta > 1e-6:
        iter += 1
        # calculate the value
        der1 = np.mat(np.zeros(3))
        der2 = np.mat(np.zeros((3,3)))
        tmp_theta = np.mat(theta)

        for k in range(m):
            if y[k] == 1:   y_k = 1.0
            else:   y_k = -1.0
            x_k = np.mat(np.append(np.array(1),x[k]))
            bottom = fun1(y_k,tmp_theta,x_k)
            der1 += -y_k*x_k/bottom
            der2 += (math.exp(y_k * tmp_theta * x_k.T) / bottom**2)*np.mat(x_k.T*x_k)
        der1 /= m
        der2 /= m
        der2 = der2.I
        delta = np.abs(np.sum(der2*der1.T))
        tmp_theta = (tmp_theta.T - der2*der1.T).T
        theta = tmp_theta.getA()[0]

        # draw the line
        x1 = np.arange(0, 8, 0.1)
        plt.title('After %d Iterations' % (iter))
        for i in range(m):
            if y[i] == 1:
                plt.scatter(x[i][0], x[i][1], color='red')
            else:
                plt.scatter(x[i][0], x[i][1], color='blue')
        plt.plot(x1, -(theta[0] + theta[1]*x1) / theta[2] )
        plt.show()
        print('%d Iteration: theta changed by %.6f'%(iter,delta))
    print('Converged After %d Iterations'%(iter))
    print(theta)

if __name__ == '__main__':
    pass

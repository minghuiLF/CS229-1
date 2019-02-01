import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def LogisticRegression():
    """
    Perform Logistic Regression with Newton's Method
    """

    def fun1(y_k, theta, x_k):
        """
        :param y_k:
        :param theta:
        :param x_k:
        :return: calculate the bottom part of the formula
        """
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

def Regression4Quasar():
    """
    solution to Regression for denoising quasar spectra
    """

    def problem_b():
        """
        perform linear regression for denoising quasar spectra
        """
        # read the data
        df_train = pd.read_table('data/quasar_train.csv', sep=',')
        df_test = pd.read_table('data/quasar_test.csv', sep=',', header=None)
        train_col = df_train.columns.values.astype(float).astype(int)
        train_x = df_train.values.astype(float)
        test_col = df_test.columns.values.astype(float).astype(int)
        test_x = df_test.values.astype(float)

        # problem b-i
        plt.figure(1)
        plt.title('Linear Regression Based on Sample #1')
        X = np.stack([np.ones(train_x[0].shape), train_col]).T
        Y = train_x[0]
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        plt.plot(train_col, train_x[0], label='Sample')
        plt.plot(train_col, train_col.dot(theta[1]) + theta[0], label='Regression')
        plt.legend()
        plt.show()

        # problem b-ii
        plt.figure(1)
        plt.title('Locally Weighted Linear Regression Based on Sample #1')
        plt.plot(train_col, train_x[0], label='Sample')
        regression = []
        X = np.stack([np.ones(train_x[0].shape), train_col]).T
        Y = train_x[0]
        tau = 5
        for t, eval_x in X:
            W = np.diag(np.exp(-(eval_x - X[:, 1]) ** 2 / (2 * tau ** 2)))
            theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
            regression.append(theta[1] * eval_x + theta[0])
        plt.plot(train_col, regression, label='Regression')
        plt.legend()
        plt.show()

        # problem b-iii
        plt.figure(1)
        plt.title('Locally Weighted Linear Regression Based on Sample #1')
        # plt.plot(train_col, train_x[0], label='Sample')

        X = np.stack([np.ones(train_x[0].shape), train_col]).T
        Y = train_x[0]
        regression = []
        taus = [1, 10, 100, 1000]
        for i in range(len(taus)):
            regression.append(np.ones(len(train_col)))
        for i in range(len(X)):
            eval_x = X[i][1]
            for j in range(len(taus)):
                tau = taus[j]
                W = np.diag(np.exp(-(eval_x - X[:, 1]) ** 2 / (2 * tau ** 2)))
                theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
                regression[j][i] = theta[1] * eval_x + theta[0]
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        axes = axes.ravel()
        for i in range(len(taus)):
            axes[i].set_title('tau = %d' % (taus[i]))
            axes[i].plot(train_col, train_x[0])
            axes[i].plot(train_col, regression[i])
        plt.show()

    def problem_c():
        """
        perform functional regression for denoising quasar spectra
        """
        # read the data
        df_train = pd.read_table('data/quasar_train.csv', sep=',')
        df_test = pd.read_table('data/quasar_test.csv', sep=',', header=None)
        train_col = df_train.columns.values.astype(float).astype(int)
        train_x = df_train.values.astype(float)
        test_col = df_test.columns.values.astype(float).astype(int)
        test_x = df_test.values.astype(float)

        # smoothing the data first
        s_train = []
        s_test = []
        thetas = []
        for i in range(train_x.shape[0]):
            regression = []
            tmp_thetas = []
            X = np.stack([np.ones(train_x[i].shape), train_col]).T
            Y = train_x[i]
            tau = 5
            for t, eval_x in X:
                W = np.diag(np.exp(-(eval_x - X[:, 1]) ** 2 / (2 * tau ** 2)))
                theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
                regression.append(theta[1] * eval_x + theta[0])
                tmp_thetas.append(theta)
            thetas.append(tmp_thetas)
            s_train.append(regression)
        for i in range(test_x.shape[0]):
            regression = []
            X = np.stack([np.ones(test_x[i].shape), test_col]).T
            Y = test_x[i]
            tau = 5
            for t, eval_x in X:
                W = np.diag(np.exp(-(eval_x - X[:, 1]) ** 2 / (2 * tau ** 2)))
                theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
                regression.append(theta[1] * eval_x + theta[0])
            s_test.append(regression)
        s_train = np.array(s_train)
        s_test = np.array(s_test)


        # problem c-ii

        def ker(t):
            return max(1-t,0)

        k = 3
        estimation = []
        errors = []
        train_left, train_right = np.split(s_train,[150],axis=1)
        for i in range(train_right.shape[0]):
            row = train_right[i]
            dist = ((train_right-row)**2).sum(axis=1)
            h = dist.max()

            neighbors = dist.argsort()[:k]

            bottom = 0
            for j in range(len(neighbors)):
                bottom += ker(dist[neighbors[j]]/h)
            left_hat = np.zeros(len(train_left[0]))
            for j in range(len(neighbors)):
                left_hat += ker(dist[neighbors[j]]/h)*train_left[neighbors[j]]
            left_hat /= bottom
            estimation.append(left_hat)
            error = np.sum((train_left[i] - left_hat) ** 2)
            errors.append(error)
        errors = np.array(errors)
        estimation = np.array(estimation)
        print('The average error is %.6f'%errors.mean())

        # plot some examples
        examples = [1,50,100,150]
        fig, axes = plt.subplots(2,2,figsize=(10,6))
        for i in range(len(examples)):
            idx = examples[i]
            ax = axes.ravel()[i]
            ax.plot(train_col, s_train[idx], label='smoothed')
            ax.plot(train_col,
                    np.hstack([estimation[idx],
                    np.zeros(max(0,np.abs(train_col.shape[0]-estimation[idx].shape[0])))]),
                    label='estimation')
            ax.legend()
            ax.set_title('Training Set Sample #%d' % (idx))
        plt.show()

        # problem c-iii
        estimation = []
        errors = []
        test_left, test_right = np.split(s_test,[150],axis=1)
        for i in range(test_right.shape[0]):
            row = test_right[i]
            dist = ((train_right-row)**2).sum(axis=1)
            h = dist.max()

            neighbors = dist.argsort()[:k]

            bottom = 0
            for j in range(len(neighbors)):
                bottom += ker(dist[neighbors[j]]/h)
            left_hat = np.zeros(len(train_left[0]))
            for j in range(len(neighbors)):
                left_hat += ker(dist[neighbors[j]]/h)*train_left[neighbors[j]]
            left_hat /= bottom
            estimation.append(left_hat)
            error = np.sum((test_left[i] - left_hat) ** 2)
            errors.append(error)
        errors = np.array(errors)
        estimation = np.array(estimation)
        print('The average error is %.6f'%errors.mean())

        # plot some examples
        examples = [1,6,10,15]
        fig, axes = plt.subplots(2,2,figsize=(10,6))
        for i in range(len(examples)):
            idx = examples[i]
            ax = axes.ravel()[i]
            ax.plot(test_col, s_test[idx], label='smoothed')
            ax.plot(test_col,
                    np.hstack([estimation[idx],
                    np.zeros(max(0,np.abs(test_col.shape[0]-estimation[idx].shape[0])))]),
                    label='estimation')
            ax.legend()
            ax.set_title('Test Set Sample #%d' % (idx))
        plt.show()
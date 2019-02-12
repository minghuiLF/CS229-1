import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################

    num_nspam = (category==0).sum()
    num_spam = (category==1).sum()
    phi_yeq0 = np.ones(N)
    phi_yeq0_d = N
    phi_yeq1 = np.ones(N)
    phi_yeq1_d = N

    for i in range(matrix.shape[0]):
        if category[i] == 0:
            phi_yeq0 += matrix[i]
            phi_yeq0_d += matrix[i].sum()
        else:
            phi_yeq1 += matrix[i]
            phi_yeq1_d += matrix[i].sum()
    phi_yeq0 /= phi_yeq0_d
    phi_yeq1 /= phi_yeq1_d

    state['phi_ye0'] = phi_yeq0
    state['phi_ye1'] = phi_yeq1
    state['phi_y'] = num_spam/(num_nspam+num_spam)
    state['num_vocab'] = N
    print(state)
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################

    for i in range(matrix.shape[0]):
        phi_yeq0 = 0
        phi_yeq1 = 0
        for j in range(state['num_vocab']):
            phi_yeq0 += np.log(state['phi_ye0'][j])*matrix[i][j]
        for j in range(state['num_vocab']):
            phi_yeq1 += np.log(state['phi_ye1'][j])*matrix[i][j]

        prob_yeq0 = 1.0/(1.0+np.exp(phi_yeq1+np.log(state['phi_y'])-phi_yeq0-np.log(1.0-state['phi_y'])))
        if prob_yeq0 <= 0.5: output[i] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error


def print_5most_indicative_token(state,tokenlist):
    print('The 5 tokens who are most indicative of the SPAM class :')
    logp = np.log(state['phi_ye1']) - np.log(state['phi_ye0'])
    idx = np.argsort(logp)

    for i in range(5):
        print('No. %d: %s - %.4f' % (i + 1, tokenlist[idx[-i - 1]], logp[idx[-i - 1]]))
    print('')

def main():
    x = [50,100,200,400,800,1400]
    y = []
    for xx in x:
        trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN'+'.'+str(xx))
        testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state)

        error = evaluate(output, testCategory)
        #print_5most_indicative_token(state,tokenlist)
        y.append(error)
    # draw(x,y)

def draw(x,y):
    plt.title('Test Error via Training-set Size')
    plt.xlabel('Training-set Size')
    plt.ylabel('Error')
    plt.plot(x, y)
    plt.plot(x, y, 'ro')
    for i in range(len(x)):
        if i != 2:
            plt.text(x[i] + 0.1, y[i] + 0.0003, '%.4f' % (y[i]))
        else:
            plt.text(x[i] + 10, y[i] - 0.0003, '%.4f' % (y[i]))
    plt.show()



if __name__ == '__main__':
    main()

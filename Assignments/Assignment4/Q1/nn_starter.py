import numpy as np
import matplotlib.pyplot as plt

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    maxn = np.max(x,axis=1,keepdims=True)
    numerator = np.exp(x-maxn)
    denominator = np.sum(numerator,axis=1,keepdims=True)
    s = numerator/denominator
    ### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = 1/(1+np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h = sigmoid(data.dot(W1)+b1)
    y = softmax(h.dot(W2)+b2)
    cost = np.multiply(-np.log(y),labels).sum()/y.shape[0]

    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params,lamb=0.0):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, y, cost = forward_prop(data,labels,params)
    y = y-labels
    gradW2 = h.T.dot(y)
    gradb2 = np.sum(y,axis=0,keepdims=True)

    h = np.multiply(h*(1-h),y.dot(W2.T))
    gradW1 = data.T.dot(h)
    gradb1 = np.sum(h,axis=0,keepdims=True)


    if lamb > 0:
        gradW2 += 2.0*lamb*W2
        gradW1 += 2.0*lamb*W1
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1/data.shape[0]
    grad['W2'] = gradW2/data.shape[0]
    grad['b1'] = gradb1/data.shape[0]
    grad['b2'] = gradb2/data.shape[0]



    return grad

def update_params(grad,params,learning_rate):
    """
    update the parameters according to the grads
    """
    params['W1'] -= learning_rate*grad['W1']
    params['W2'] -= learning_rate*grad['W2']
    params['b1'] -= learning_rate * grad['b1']
    params['b2'] -= learning_rate * grad['b2']





def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}


    ### YOUR CODE HERE
    batch_size = 1000
    output_dim = trainLabels.shape[1]
    num_epochs = 30

    params['W1'] = np.random.standard_normal((n,num_hidden))
    params['b1'] = np.zeros((1,num_hidden))
    params['W2'] = np.random.standard_normal((num_hidden,output_dim))
    params['b2'] = np.zeros((1, output_dim))

    num_iter = np.int(m/batch_size)

    train_loss = []
    train_acc = []
    dev_loss = []
    dev_acc = []

    epoch = 1
    while epoch <= num_epochs:

        for i in range(num_iter):
            data = trainData[i*batch_size:(i+1)*batch_size]
            labels = trainLabels[i*batch_size:(i+1)*batch_size]
            grad = backward_prop(data,labels,params,lamb=0.25)
            update_params(grad,params,learning_rate)

        h, y, tr_cost = forward_prop(trainData,trainLabels,params)
        train_loss.append(tr_cost)
        tr_acc = compute_accuracy(y,trainLabels)
        train_acc.append(tr_acc)
        h, y, cost = forward_prop(devData,devLabels,params)
        dev_loss.append(cost)
        acc = compute_accuracy(y,devLabels)
        dev_acc.append(acc)

        print('Epoch %d. training loss: %.2f acc: %.2f'%(epoch,tr_cost,tr_acc))
        epoch += 1

    plt.title('Loss Change')
    plt.ylabel('CE Loss')
    plt.xlabel('Epoch')
    plt.plot(np.arange(1,epoch,1),train_loss,label='training-set')
    plt.plot(np.arange(1,epoch,1),dev_loss,label='dev-set')
    plt.legend()
    plt.show()

    plt.title('Accuracy Change')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(1, epoch, 1), train_acc, label='training-set')
    plt.plot(np.arange(1, epoch, 1), dev_acc, label='dev-set')
    plt.legend()
    plt.show()

    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    params = nn_train(trainData, trainLabels, devData, devLabels)
    np.save('params_v2',params)

    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print('Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()

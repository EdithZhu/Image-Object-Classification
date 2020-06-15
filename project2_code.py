import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm


def pegasos(X_train, y_train, X_test, y_test, B, lambda_):
    w = np.zeros(X_train.shape[1])  # initialize w
    err_train = []
    err_test = []
    for iteration in range(100):
        index = np.random.randint(0, len(X_train), B)
        X_batch = X_train[index]
        y_batch = y_train[index]
        index_new = [i for i in range(B) if y_batch[i] * w @ X_batch[i] < 1]
        X_batch_new = X_batch[index_new]
        y_batch_new = y_batch[index_new]
        coeffi_1 = 1 / ((iteration + 1) * lambda_)
        delta_t = lambda_ * w - coeffi_1 / B * y_batch_new @ X_batch_new
        w_ = w - coeffi_1 * delta_t
        w = min(1,1/math.sqrt(lambda_)/math.sqrt(sum(w_**2)))*w_
        y_train_pred = np.sign(X_train@w)
        y_test_pred = np.sign(X_test@w)
        err_train.append(np.count_nonzero(y_train_pred!=y_train)/len(y_train))
        err_test.append(np.count_nonzero(y_test_pred!=y_test)/len(y_test))
    return err_train,err_test,w

def error_graph_together(train_error,test_error):
    plt.plot(np.arange(len(train_error)),train_error, label='training error')
    plt.plot(np.arange(len(test_error)),test_error, label='test error')
    plt.legend(loc = 'best')
    plt.xlabel('iteration')
    plt.ylabel('error percentage')
    plt.title("error vs. the number of iterations")
    plt.show()


def adagrad(X_train, y_train, X_test, y_test, lambda_, eta, B, iterations):
    train_error = []
    test_error = []
    w = csr_matrix([0] * X_train.shape[1])
    G = csr_matrix([1] * X_train.shape[1])
    for iteration in range(iterations):
        sum = csr_matrix([0] * X_train.shape[1])
        indexes = random.sample(range(X_train.shape[0]), B)
        for index in indexes:
            multiply_matrix = w.dot(X_train[index, :].T) * y_train[index]
            if multiply_matrix < 1:
                sum = sum + X_train[index, :] * y_train[index]
        gradient = lambda_ * w - sum / B
        gradient = csr_matrix(gradient)
        g_sqrt = np.sqrt(G + csr_matrix.multiply(gradient, gradient))
        w_ = w - csr_matrix.multiply(csr_matrix(csr_matrix([eta] * X_train.shape[1]) / g_sqrt), gradient)
        res = csr_matrix.multiply(g_sqrt, w_)
        w = min(1.0, 1.0 / math.sqrt(lambda_) / norm(res)) * w_
        if iteration % 10 == 0:
            train_error.append(compute_error(X_train, y_train, w))
            test_error.append(compute_error(X_test, y_test, w))

    return w, train_error, test_error

def compute_error(X_train, y_train, w):
    w_ = w.T.toarray()
    y_pred = X_train.dot(w_).reshape(1, -1).tolist()[0]
    res = np.array(y_pred)*np.array(y_train)
    error = len(np.where(res < 0)[0]) / len(res)
    return error

def error_graph1_together(train_error,test_error):
    plt.plot(np.arange(0,10*len(train_error),10),train_error, label='training error')
    plt.plot(np.arange(0,10*len(test_error),10),test_error, label='test error')
    plt.legend(loc = 'best')
    plt.xlabel('iteration')
    plt.ylabel('error percentage')
    plt.title("error vs. the number of iterations")
    plt.show()


def pegasos1(X_train,y_train, X_test, y_test,B,lambda_,X_test1,y_test1):
    w = np.zeros(X_train.shape[1])
    err_test = [] #store the testerror {2,5,7}
    err_test1 = [] #store the training error {2,5,7}
    y_test_lst = [] #store labels of test images {2,5,7}
    y_test1_lst = [] #store labels of training images {2,5,7}
    for iteration in range(100):
        index = np.random.randint(0, len(X_train), B)
        X_batch = X_train[index]
        y_batch = y_train[index]
        index_new = [i for i in range(B) if y_batch[i] * w @ X_batch[i] < 1]
        X_batch_new = X_batch[index_new]
        y_batch_new = y_batch[index_new]
        coeffi_1 = 1 / ((iteration + 1) * lambda_)
        delta_t = lambda_ * w - coeffi_1 / B * y_batch_new @ X_batch_new
        w_ = w - coeffi_1 * delta_t
        w = min(1,1/math.sqrt(lambda_)/math.sqrt(sum(w_**2)))*w_
        y_test_pred = np.sign(X_test@w)
        y_test1_pred = np.sign(X_test1@w)
        err_test.append(np.count_nonzero(y_test_pred!=y_test)/len(y_test))
        err_test1.append(np.count_nonzero(y_test1_pred!=y_test1)/len(y_test1))
        y_test_lst.append(y_test_pred)
        y_test1_lst.append(y_test1_pred)
    return err_test1,err_test,w,y_test1_lst,y_test_lst



if __name__ == '__main__':
    #************************************* problem 1 *******************************
    train = np.load('train.npy')
    train_labels = np.load('train_labels.npy')
    test = np.load('test.npy')
    test_labels = np.load('test_labels.npy')

    train3 = []
    train3_labels = []
    test3 = []
    test3_labels = []
    for i in range(len(train_labels)):
        if train_labels[i] == 2 or train_labels[i] == 5 or train_labels[i] == 7:
            train3.append(train[i])
            train3_labels.append(train_labels[i])
    train3 = np.array(train3)
    for i in range(len(test_labels)):
        if test_labels[i] == 2 or test_labels[i] == 5 or test_labels[i] == 7:
            test3.append(test[i])
            test3_labels.append(test_labels[i])
    test3 = np.array(test3)

    random_train3 = []
    for i in range(10):
        index = random.randint(0, len(train3))
        random_train3.append(index)
    # print('indexes of 10 random images: ' + str(random_train3))

    random_test3 = []
    for i in range(10):
        index = random.randint(0, len(test3))
        random_test3.append(index)
    # print('indexes of 10 random images: ' + str(random_test3))

    random_train3_labels = []
    random_test3_labels = []
    for i in range(10):
        index_train = random_train3[i]
        index_test = random_test3[i]
        random_train3_labels.append(train3_labels[index_train])
        random_test3_labels.append(test3_labels[index_test])
    #print('labels of 10 random images(train data set): ' + str(random_train3_labels))
    #print('labels of 10 random images(test data set): ' + str(random_test3_labels))
    fig, ax = plt.subplots(2, 10, figsize=(20, 5), subplot_kw=None, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i in range(10):
        index_train = random_train3[i]
        index_test = random_test3[i]
        ax[0, i].imshow(np.reshape(train3[index_train, :], (28, 28)), cmap='Greys')
        ax[0, i].set_title('label = ' + str(random_train3_labels[i]))
        ax[1, i].imshow(np.reshape(test3[index_test, :], (28, 28)), cmap='Greys')
        ax[1, i].set_title('label = ' + str(random_test3_labels[i]))
    ax[0, 0].set_ylabel('train data set')
    ax[1, 0].set_ylabel('test data set')
    plt.show()

    # ********************************* problem 2 ******************************
    # ******************** preprocess: involving labels {2,5} ******************
    # training data
    X_train = np.vstack((train[train_labels == 2], train[train_labels == 5]))
    y_train = np.hstack((train_labels[train_labels == 2], train_labels[train_labels == 5]))
    y_train[y_train == 2] = 1
    y_train[y_train == 5] = -1
    X_train = X_train.reshape(len(X_train), 28 * 28)
    # test data
    X_test = np.vstack((test[test_labels == 2], test[test_labels == 5]))
    y_test = np.hstack((test_labels[test_labels == 2], test_labels[test_labels == 5]))
    y_test[y_test == 2] = 1
    y_test[y_test == 5] = -1
    X_test = X_test.reshape(len(X_test), 28 * 28)
    # present the amount of training and test images
    print('length of train data set{2,5}: ' + str(len(X_train)))
    print('length of test data set{2,5}: ' + str(len(X_test)))

    # ******************** problem 2: present the results ******************
    start = time.clock()
    err_train, err_test, w = pegasos(X_train, y_train, X_test, y_test, 50, 2)
    elapsed = (time.clock() - start) / 60
    print("minute elapsed for problem 2: " + str(elapsed))
    # error_graph(err_train, err_test)
    print('accuracy of training data is: ' + str(1 - err_train[-1]))
    print('accuracy of testing data is: ' + str(1 - err_test[-1]))

    error_graph_together(err_train, err_test)

    # ********************************* problem 3 ******************************
    # ****************************** present the results ***********************
    batch_size = 50
    T = 100
    lambda_ = 0.0001
    # eta = 1/np.sqrt(T)
    eta = 0.01

    start = time.clock()
    w, train_err, test_err = adagrad(X_train, y_train, X_test, y_test, lambda_, eta, batch_size, T)
    elapsed = (time.clock() - start) / 60
    print("minute elapsed for problem 3: " + str(elapsed))

    error_graph1_together(train_err, test_err)
    print('accuracy of training data is: ' + str(1 - train_err[-1]))
    print('accuracy of testing data is: ' + str(1 - test_err[-1]))


    #************************************* problem 5 *************************************
    #*********************************** train 3 classifiers *****************************
    start = time.clock()
    tri_batch = 50
    tri_lambda = 2
    # ********************classifier 1 {2=1,5=-1}*******************************
    X1_train = np.vstack((train[train_labels == 2], train[train_labels == 5]))
    y1_train = np.hstack((train_labels[train_labels == 2], train_labels[train_labels == 5]))
    y1_train[y1_train == 2] = 1
    y1_train[y1_train == 5] = -1
    X1_test = np.vstack((test[test_labels == 2], test[test_labels == 5], test[test_labels == 7]))
    y1_test = np.hstack((test_labels[test_labels == 2], test_labels[test_labels == 5], test_labels[test_labels == 7]))
    y1_test[y1_test == 2] = 1
    y1_test[y1_test == 5] = -1
    X1_test1 = np.vstack((train[train_labels == 2], train[train_labels == 5], train[train_labels == 7]))
    y1_test1 = np.hstack(
        (train_labels[train_labels == 2], train_labels[train_labels == 5], train_labels[train_labels == 7]))
    y1_test1[y1_test1 == 2] = 1
    y1_test1[y1_test1 == 5] = -1
    X1_train = X1_train.reshape(len(X1_train), 28 * 28)
    X1_test = X1_test.reshape(len(X1_test), 28 * 28)
    X1_test1 = X1_test1.reshape(len(X1_test1), 28 * 28)
    err1_train, err1_test, w1, y1_train_lst, y1_test_lst = pegasos1(X1_train, y1_train, X1_test, y1_test, tri_batch,
                                                                    tri_lambda, X1_test1, y1_test1)
    # print(y1_train_lst[0])#100 array = 100 iteration，each array has 12000 data
    # plots(err1_train, err1_test)

    # ********************classifier 2 {2=1,7=-1}*******************************
    X2_train = np.vstack((train[train_labels == 2], train[train_labels == 7]))
    y2_train = np.hstack((train_labels[train_labels == 2], train_labels[train_labels == 7]))
    y2_train[y2_train == 2] = 1
    y2_train[y2_train == 7] = -1
    X2_test = np.vstack((test[test_labels == 2], test[test_labels == 5], test[test_labels == 7]))
    y2_test = np.hstack((test_labels[test_labels == 2], test_labels[test_labels == 5], test_labels[test_labels == 7]))
    y2_test[y2_test == 2] = 1
    y2_test[y2_test == 7] = -1
    X2_test1 = np.vstack((train[train_labels == 2], train[train_labels == 5], train[train_labels == 7]))
    y2_test1 = np.hstack(
        (train_labels[train_labels == 2], train_labels[train_labels == 5], train_labels[train_labels == 7]))
    y2_test1[y2_test1 == 2] = 1
    y2_test1[y2_test1 == 7] = -1
    X2_train = X2_train.reshape(len(X2_train), 28 * 28)
    X2_test = X2_test.reshape(len(X2_test), 28 * 28)
    X2_test1 = X2_test1.reshape(len(X2_test1), 28 * 28)
    err2_train, err2_test, w2, y2_train_lst, y2_test_lst = pegasos1(X2_train, y2_train, X2_test, y2_test, tri_batch,
                                                                    tri_lambda, X2_test1, y2_test1)
    # print(y_train_lst)
    # plots(err2_train,err2_test)
    # print(y2_test_lst)

    # ********************classifier 3 {5=1,7=-1}*******************************
    X3_train = np.vstack((train[train_labels == 5], train[train_labels == 7]))
    y3_train = np.hstack((train_labels[train_labels == 5], train_labels[train_labels == 7]))
    y3_train[y3_train == 5] = 1
    y3_train[y3_train == 7] = -1
    X3_test = np.vstack((test[test_labels == 2], test[test_labels == 5], test[test_labels == 7]))
    y3_test = np.hstack((test_labels[test_labels == 2], test_labels[test_labels == 5], test_labels[test_labels == 7]))
    y3_test[y3_test == 5] = 1
    y3_test[y3_test == 7] = -1
    X3_test1 = np.vstack((train[train_labels == 2], train[train_labels == 5], train[train_labels == 7]))
    y3_test1 = np.hstack(
        (train_labels[train_labels == 2], train_labels[train_labels == 5], train_labels[train_labels == 7]))
    y3_test1[y3_test1 == 5] = 1
    y3_test1[y3_test1 == 7] = -1
    X3_train = X3_train.reshape(len(X3_train), 28 * 28)
    X3_test = X3_test.reshape(len(X3_test), 28 * 28)
    X3_test1 = X3_test1.reshape(len(X3_test1), 28 * 28)
    err3_train, err3_test, w3, y3_train_lst, y3_test_lst = pegasos1(X3_train, y3_train, X3_test, y3_test, tri_batch,
                                                                    tri_lambda, X3_test1, y3_test1)
    # print(y3_test_lst)
    # plots(err3_train,err3_test)

    err_tri_train = []
    err_tri_test = []
    iterations = 100
    tri_train_labels = np.hstack(
        (train_labels[train_labels == 2], train_labels[train_labels == 5], train_labels[train_labels == 7]))
    tri_test_labels = np.hstack(
        (test_labels[test_labels == 2], test_labels[test_labels == 5], test_labels[test_labels == 7]))
    for iter in range(iterations):
        count_train = 0
        count_test = 0
        data18000 = []
        data3000 = []
        final_train_labels = []
        final_test_labels = []
        for i in range(18000):
            data18000.append([y1_train_lst[iter][i], y2_train_lst[iter][i], y3_train_lst[iter][i]])
        for i in range(18000):
            if data18000[i][0] == 1 and data18000[i][1] == 1:
                final_train_labels.append(2)
            elif data18000[i][0] == -1 and data18000[i][2] == 1:
                final_train_labels.append(5)
            elif data18000[i][1] == -1 and data18000[i][2] == -1:
                final_train_labels.append(7)
            else:
                final_train_labels.append(9)
        for i in range(18000):
            if final_train_labels[i] == tri_train_labels[i]:
                count_train = count_train + 1
        err_tri_train.append(1 - (count_train / 18000))

        for j in range(3000):
            data3000.append([y1_test_lst[iter][j], y2_test_lst[iter][j], y3_test_lst[iter][j]])
        for j in range(3000):
            if data3000[j][0] == 1 and data3000[j][1] == 1:
                final_test_labels.append(2)
            elif data3000[j][0] == -1 and data3000[j][2] == 1:
                final_test_labels.append(5)
            elif data3000[j][1] == -1 and data3000[j][2] == -1:
                final_test_labels.append(7)
            else:
                final_test_labels.append(9)
        for j in range(3000):
            if final_test_labels[j] == tri_test_labels[j]:
                count_test = count_test + 1
        err_tri_test.append(1 - (count_test / 3000))


    elapsed = (time.clock() - start) / 60
    print("minute elapsed for problem 5: " + str(elapsed))

    print('accuracy of training data is: ' + str(1 - err_tri_train[-1]))
    print('accuracy of testing data is: ' + str(1 - err_tri_test[-1]))

    error_graph_together(err_tri_train, err_tri_test)



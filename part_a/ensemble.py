#imports
from operator import add
import sys
sys.path.append('/Users/Syed-Talha/CSC311_Project/starter_code') #or whatever directory

from utils import *
import knn 
import item_response
import neural_network 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.sparse
import torch
from torch.autograd import Variable
from sklearn.impute import KNNImputer

def get_data(path):
    '''
    Helper function to get data from the CSVs and load up the features and labels
    '''
    df = pd.read_csv(path)
    data = df[['question_id', 'user_id']]
    labels = df['is_correct']
    return data, labels

def bootstrap(train_data, ratio=1.0):
    N = len(train_data['is_correct'])
    sample_size = round(N*ratio)

    boot_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    for i in range(sample_size):
        choice = random.randrange(0, N)
        boot_data['user_id'].append(train_data['user_id'][choice])
        boot_data['question_id'].append(train_data['question_id'][choice])
        boot_data['is_correct'].append(train_data['is_correct'][choice])

    return boot_data

def bootstrap_matrix(data, matrix, add_zero = False, ratio=1.0):
    #note: the add_zero argument will optionally return a zero matrix, needed for the autoencoder training
    N = len(data['is_correct'])
    sample_size = round(N*ratio)

    N_users = matrix.shape[0]
    N_questions = matrix.shape[1]

    boot_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    boot_matrix = scipy.sparse.csc_matrix((N_users,N_questions))
    boot_matrix[:] = np.nan

    for i in range(N):
        choice = random.randrange(0, N)
        user = data['user_id'][choice]
        question = data['question_id'][choice]
        correctness = data['is_correct'][choice]

        boot_data['user_id'].append(user)
        boot_data['question_id'].append(question)
        boot_data['is_correct'].append(correctness)

        boot_matrix[user, question] = correctness
        boot_matrix = boot_matrix
    
    boot_matrix = boot_matrix.toarray()

    if not add_zero:
        return boot_data, boot_matrix
    else:
        #for neural nets
        zero_boot_matrix = boot_matrix.copy()
        zero_boot_matrix[np.isnan(boot_matrix)] = 0
        zero_boot_matrix = torch.FloatTensor(zero_boot_matrix)
        boot_matrix = torch.FloatTensor(boot_matrix)
        return boot_data, boot_matrix, zero_boot_matrix

def ensemble(train_data, train_matrix, data, theta, beta, k, model):
    '''
    Given three trained models, and the testing/validation data, return the ensemble
    predictions and their accuracy

    data: the test or validation data, in dict or df format
    train_data: bootstrapped data for IRT
    theta: optimized parameter for IRT
    beta: optimized parameter for IRT
    k: optimized parameter for KNN
    model: a trained autoencoder
    '''

    irt_pred = irt_prediction(data, theta, beta) #list 7086
    knn_pred = knn_prediction(data, train_matrix, k) #list 7086
    nn_pred = nn_prediction(data, train_matrix, model) #list 7086

    ensemble_pred = []
    ensemble_acc = 0
    
    for i in range(len(data["user_id"])):
        pred = (irt_pred[i] + knn_pred[i] + nn_pred[i]) > 1.5
        ensemble_pred.append(pred)
        if pred == data['is_correct'][i]:
            ensemble_acc += 1
    ensemble_acc /=  len(data['user_id'])

    return ensemble_pred, ensemble_acc

def irt_prediction(data, theta, beta):
    ''' 
    Given the validation or test data, return the IRT prediction

    theta, beta: parameters from the trained IRT model
    i: a loop index
    '''

    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        pred.append(int(p_a >= 0.5))
    return pred

def nn_prediction(data, train_matrix, model):
    '''
    Given data, return the trained AutoEncoder prediction

    model: an torch.nn.Module object of the trained neural network
    '''
    model.eval()
    nn_pred = []

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)

    for i, u in enumerate(data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][data["question_id"][i]].item() 
        nn_pred.append(int(guess > 0.5))
    return nn_pred

def knn_prediction(data, matrix, k, threshold=0.5):
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)

    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if mat[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions


def main(): 
    np.random.seed(0)

    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    sparse_matrix = load_train_sparse("../data").toarray()


    ################################################
    ###   Training Code
    ################################################


    #MODEL 1: KNN
    #best hyperparameters: {k = 11}
    boot_data_knn, boot_matrix_knn = bootstrap_matrix(train_data, sparse_matrix)
    #note: I convert the csc matrix to an array inside the function

    k_vals = [1, 6, 11, 16, 21, 26]
    accuracies = []

    for k in k_vals:
        accuracies.append(knn.knn_impute_by_user(boot_matrix_knn, val_data, k))
    
    '''
    fig, ax = plt.subplots()
    ax.plot(k_vals, accuracies, 'ro-')
    ax.set_xlabel('k Value')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracies for Different Values of k')
    plt.show()
    '''

    #best value was k* = 11, now compute test accuracy
    k_best = k_vals[accuracies.index(max(accuracies))]
    print("best k:", k_best)
    val_acc_knn = knn.knn_impute_by_user(boot_matrix_knn, val_data, k_best)
    print("KNN VALIDATION ACCURACY:", val_acc_knn)
    test_acc_knn = knn.knn_impute_by_user(boot_matrix_knn, test_data, k_best)
    print("KNN TEST ACCURACY:", test_acc_knn)

    #MODEL 2: ITEM RESPONSE THEORY
    #best hyperparameters: {lr = 0.01, iterations = 15}
    boot_data_irt = bootstrap(train_data, ratio=1.0)

    lr = 0.01
    iterations = 15

    iterations_list = list(range(1, iterations + 1))

    # Train the IRT model
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = item_response.irt(boot_data_irt, val_data, lr, iterations)

    #plt.plot(iterations_list, train_lld_lst, 'red', label="Training Log-likelihood")

    plt.plot(iterations_list, val_acc_lst, 'blue', label="Validation Accuracy")
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood as a Function of the Number of Iterations')
    plt.legend(loc="upper right")
    plt.show()


    # part (c)
    final_val_acc = val_acc_lst[-1]
    test_acc_IRT = item_response.evaluate(test_data, theta, beta)

    print("ITEM RESPONSE THEORY VALIDATION ACCURACY:", final_val_acc)
    print("ITEM RESPONSE THEORY TEST ACCURACY:", test_acc_IRT)


    #MODEL 3: AUTOENCODERS
    #best hyperparameters: {k = 200, lambda = 0.001, lr = 0.01}
    boot_data_NN, boot_matrix_NN, zero_boot_matrix_NN = bootstrap_matrix(train_data, sparse_matrix, add_zero=True)
    

    #Set model hyperparameters.
    '''
    ks = [10, 20, 30, 50, 100]

    for k in ks:
        model = neural_network.AutoEncoder(boot_matrix_NN.shape[1], k)
        print(k)

    # Set optimization hyperparameters.
        lr = 0.02
        num_epoch = 50
        lamb = 0
        neural_network.train(model, lr, lamb, boot_data_NN, boot_matrix_NN, zero_boot_matrix_NN,
            val_data, num_epoch)

    #The optimal k* is 200 with 0.01 learning rate and 20 epochs 
    epoch = [*range(0,20,1)]
    lr = 0.01
    lamb = 0
    model = neural_network.AutoEncoder(boot_matrix_NN.shape[1], 200)

    accuracies = neural_network.train(model, lr, lamb, boot_data_NN, boot_matrix_NN, zero_boot_matrix_NN, val_data, 20)
    
    print(neural_network.evaluate(model, zero_boot_matrix_NN, test_data))
    # #plot
    fig, ax = plt.subplots()
    ax.plot(epoch, accuracies, 'ro-')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracies vs Epoch Number')
    ax.set_xticks(np.arange(0, 20, 1.0))
    plt.show() 


    #tuning lambda
    lambdas = [0.001, 0.01, 0.1, 1]
    for lamb in lambdas:
        model = neural_network.AutoEncoder(boot_matrix_NN.shape[1], 20)
        print(lamb)

    # Set optimization hyperparameters.
        lr = 0.01
        num_epoch = 50
        neural_network.train(model, lr, lamb, boot_data_NN, boot_matrix_NN, zero_boot_matrix_NN, val_data, num_epoch)
    '''

    #The optimal lambda is 0.01
    epoch = [*range(0,50,1)]
    lr = 0.02
    lamb = 0.001
    model = neural_network.AutoEncoder(boot_matrix_NN.shape[1], k=20)
    accuracies = neural_network.train(model, lr, lamb, boot_data_NN, boot_matrix_NN, zero_boot_matrix_NN,
          val_data, 50)
    
    val_acc_NN = neural_network.evaluate(model, zero_boot_matrix_NN, val_data)
    print("NEURAL NET VALIDATION ACCURACY:", val_acc_NN)
    test_acc_NN = neural_network.evaluate(model, zero_boot_matrix_NN, test_data)
    print("NEURAL NET TEST ACCURACY:", test_acc_NN)

    print(ensemble(train_data, sparse_matrix, val_data, theta, beta, k_best, model))
    print(ensemble(train_data, sparse_matrix, test_data, theta, beta, k_best, model))

    return

if __name__ == '__main__':
    main()
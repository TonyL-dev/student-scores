#from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sys
sys.path.append('/Users/Syed-Talha/CSC311_Project/starter_code')
from utils import *


def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, p=0):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.dropout = nn.Dropout(p=p)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = torch.sigmoid(self.g(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, train_matrix, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    train_acc =  []
    val_acc = []
    train_losses = []
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    #criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]

    for epoch in range(0, num_epoch):
        model.train()
        train_loss = 0.

        for user_id in range(num_student):
            regularizer = (lamb/2)*model.get_weight_norm()
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0) 
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            #regularized vs unregularized loss 
            loss = torch.sum((output - target) ** 2.)
            #loss = criterion(output, target)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_losses.append(train_loss)
        train_accuracy = evaluate(model, zero_train_data, train_data)
        valid_accuracy = evaluate(model, zero_train_data, valid_data)
        train_acc.append(train_accuracy)
        val_acc.append(valid_accuracy)
        print("Epoch: {} \tTrain Loss: {:.6f}\t "
              "Train Acc: {}\tValid Acc: {}".format(epoch+1, train_loss, train_acc[-1], val_acc[-1]))
    return train_acc, val_acc, train_losses
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(base_path='/Users/Syed-Talha/CSC311_Project/starter_code/data')
    train_data = load_train_csv("../data")

    k = 10
    lr = 0.02
    num_epochs = 50
    lamb = 0.001
    train_accs = []
    val_accs = []
    train_losses = []
    ps = list(np.linspace(0, 1, 21))

    for p in ps:
        print(p)
        model = AutoEncoder(train_matrix.shape[1], k, p)
        train_acc, val_acc, train_loss = train(model, lr, lamb, train_data, train_matrix, zero_train_matrix, valid_data, num_epochs)
        train_accs.append(train_acc[-1])
        val_accs.append(val_acc[-1])
        train_losses.append(train_loss)
    
    fig, ax = plt.subplots()
    ax.plot(ps, train_accs, 'ro-', label='train')
    ax.plot(ps, val_accs, 'bo-', label='valid')
    ax.set_xlabel('Dropout Probability p')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies for Various Dropout Probabilities')
    ax.legend()
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(ps, [x1 - x2 for (x1, x2) in zip(train_accs, val_accs)], 'ro-')
    ax1.set_xlabel('Dropout Probability p')
    ax1.set_ylim(0, 0.3)
    ax1.set_ylabel('Generalization Gap')
    ax1.set_title('Generalization Gaps for Various Dropout Probabilities')
    plt.show()    

    print(train_accs)




    '''
    k = 10
    lr = 0.02
    num_epochs = 50
    lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], k)

    train_acc, val_acc, train_loss = train(model, lr, lamb, train_data, train_matrix, zero_train_matrix, valid_data, num_epochs)

    epochs = list(range(1,num_epochs+1))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(epochs, train_acc, 'ro-')
    ax[0].plot(epochs, val_acc, 'bo-')
    ax[0].set_xlabel('Epoch Number')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Accuracies vs Epoch Number')
    ax[0].legend(['train', 'valid'])

    ax[1].plot(epochs, train_loss, 'ro-')
    ax[1].set_xlabel('Epoch Number')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss vs Epoch Number')
    plt.show() 
    '''

    # fig, ax = plt.subplots()
    # ax.plot(epoch, accuracies, 'ro-')
    # ax.set_xlabel('Epoch Number')
    # ax.set_ylabel('Training Loss')
    # ax.set_title('Training Loss vs Epoch Number')
    # ax.set_xticks(np.arange(0, 20, 1.0))
    # plt.show()  

    # Set model hyperparameters.
    # ks = [10, 50, 100, 200, 500]

    # for k in ks:
    #   model = AutoEncoder(train_matrix.shape[1], k)
    #   print(k)

    # # Set optimization hyperparameters.
    #   lr = 0.01
    #   num_epoch = 20
    #   lamb = 0
    #   train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)

    #The optimal k* is 200 with 0.01 learning rate and 20 epochs 
    # epoch = [*range(0,20,1)]
    # lr = 0.01
    # lamb = 0
    # model = AutoEncoder(train_matrix.shape[1], 200)

    # accuracies = train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, 20)
    
    # print(evaluate(model, zero_train_matrix, test_data))
    # #plot
    # fig, ax = plt.subplots()
    # ax.plot(epoch, accuracies, 'ro-')
    # ax.set_xlabel('Epoch Number')
    # ax.set_ylabel('Validation Accuracy')
    # ax.set_title('Validation Accuracies vs Epoch Number')
    # ax.set_xticks(np.arange(0, 20, 1.0))
    # plt.show()  

    # fig, ax = plt.subplots()
    # ax.plot(epoch, accuracies, 'ro-')
    # ax.set_xlabel('Epoch Number')
    # ax.set_ylabel('Training Loss')
    # ax.set_title('Training Loss vs Epoch Number')
    # ax.set_xticks(np.arange(0, 20, 1.0))
    # plt.show()  

    #tuning lambda
    # lambdas = [0.001, 0.01, 0.1, 1]
    # for lamb in lambdas:
    #   model = AutoEncoder(train_matrix.shape[1], 200)

    # # Set optimization hyperparameters.
    #   lr = 0.01
    #   num_epoch = 20
    #   train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)
    
    #The optimal lambda is 0.001
    '''
    epoch = [*range(0,50,1)]
    lr = 0.01
    lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], 200)
    '''

    #accuracies = train(model, lr, lamb, train_matrix, zero_train_matrix,
    #      valid_data, 20)
    
    #print(evaluate(model, zero_train_matrix, test_data))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

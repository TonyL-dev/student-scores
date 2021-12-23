from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

#########################################################################
#NOTE: Change the path to the directory with utils.py in it when running#
#########################################################################

import sys
sys.path.append('/Users/Syed-Talha/CSC311_Project/starter_code')

from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.transpose()).transpose()
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_vals = [1, 6, 11, 16, 21, 26]
    accuracies = []

    for k in k_vals:
        accuracies.append(knn_impute_by_user(sparse_matrix, val_data, k))
    
    fig, ax = plt.subplots()
    ax.plot(k_vals, accuracies, 'ro-')
    ax.set_xlabel('k Value')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracies for Different Values of k')
    plt.show()

    #best value was k* = 11, now compute test accuracy
    k_best = k_vals[accuracies.index(max(accuracies))]
    print("best k:", k_best)
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_best)
    print("Test Accuracy is:", test_acc)

    ##item based collaborative filtering
    for k in k_vals:
        accuracies.append(knn_impute_by_item(sparse_matrix, val_data, k))
    
    fig, ax = plt.subplots()
    ax.plot(k_vals, accuracies, 'ro-')
    ax.set_xlabel('k Value')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracies for Different Values of k')
    plt.show()

    #best value was k* = 21, now compute test accuracy
    k_best = 21
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_best)
    print("Test Accuracy is:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

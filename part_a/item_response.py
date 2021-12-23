from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    c = data["is_correct"]
    u = data["user_id"]
    q = data["question_id"]

    log_lklihood = 0

    # Just need 1 loop

    # u contains the i's
    # q contains the j's
    for idx in range(len(c)):
        i = u[idx]
        j = q[idx]
        c_ij = c[idx]

        log_lklihood += c_ij * theta[i] - c_ij * beta[j] - np.logaddexp(0, theta[i] - beta[j])

    log_lklihood = log_lklihood / len(c)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    c = data["is_correct"]
    u = data["user_id"]
    q = data["question_id"]
    # Just 1 for loop
    for idx in range(len(c)):
        i = u[idx]
        j = q[idx]
        c_ij = c[idx]

        # Use derivatives of neg log-likelihood
        theta[i] = theta[i] - lr * (-1 * c_ij + sigmoid(theta[i] - beta[j]))

        beta[j] = beta[j] - lr * (c_ij - sigmoid(theta[i] - beta[j]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    # Initialize theta and beta randomly
    theta_len = len(set(data["user_id"]))
    beta_len = len(set(data["question_id"]))

    np.random.seed(1) # Used to determine best hyperparameters
    theta = np.random.rand(theta_len, 1)
    beta = np.random.rand(beta_len, 1)

    val_acc_lst = []
    lld_lst = []  # This line was added by the Student
    val_lld_lst = []  # This line was added by the Student

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        lld_lst.append(neg_lld)  # This line was added by the Student
        val_lld_lst.append(val_neg_lld)  # This line was added by the Student
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("/Users/jordanbotines/Desktop/CSC311/HW4 Code/data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("/Users/jordanbotines/Desktop/CSC311/HW4 Code/data")
    val_data = load_valid_csv("/Users/jordanbotines/Desktop/CSC311/HW4 Code/data")
    test_data = load_public_test_csv("/Users/jordanbotines/Desktop/CSC311/HW4 Code/data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    lr = 0.01
    iterations = 27

    iterations_list = list(range(1, iterations + 1))

    # Train the IRT model
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = irt(train_data, val_data, lr, iterations)

    plt.plot(iterations_list, train_lld_lst, 'red', label="Training Log-likelihood")
    plt.plot(iterations_list, val_lld_lst, 'blue', label="Validation Log-likelihood")
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood as a Function of the Number of Iterations')
    plt.legend(loc="upper right")
    plt.show()

    # part (c)
    final_val_acc = val_acc_lst[-1]
    test_acc = evaluate(test_data, theta, beta)

    print("Final Validation Accuracy:")
    print(final_val_acc)
    print("Final Test Accuracy:")
    print(test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # We pick 3 questions
    j1 = train_data["question_id"][0]
    j2 = train_data["question_id"][500]
    j3 = train_data["question_id"][1000]

    q1_correct_prob = []
    q2_correct_prob = []
    q3_correct_prob = []

    theta_sorted = np.sort(theta, axis=None)

    for i in range(len(theta_sorted)):
        q1_correct_prob.append(sigmoid(theta_sorted[i] - beta[j1]))
        q2_correct_prob.append(sigmoid(theta_sorted[i] - beta[j2]))
        q3_correct_prob.append(sigmoid(theta_sorted[i] - beta[j3]))

    plt.plot(theta_sorted, q1_correct_prob, 'red', label="Question 1")
    plt.plot(theta_sorted, q2_correct_prob, 'blue', label="Question 2")
    plt.plot(theta_sorted, q3_correct_prob, 'darkorange', label="Question 3")
    plt.xlabel('Theta')
    plt.ylabel('Probability of the Correct Response')
    plt.title('Probability of the Correct Response Given a Question j')
    plt.legend(loc="lower right")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

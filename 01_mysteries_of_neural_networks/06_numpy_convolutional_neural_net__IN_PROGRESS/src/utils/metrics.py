import numpy as np


def convert_prob_into_class(probs: np.array) -> np.array:
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat: np.array, Y: np.array) -> float:
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def multi_class_cross_entropy_loss(Y_hat, Y, eps=1e-12) -> float:
    m = Y_hat.shape[1]
    Y_hat = np.clip(Y_hat, eps, 1. - eps)
    loss = - np.sum(Y * np.log(Y_hat)) / m
    return loss
import numpy as np


def preprocess(train_img, train_mask, test_img, test_mask):
    X_train = np.array(train_img, dtype='float') / 255
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std

    num_classes = len(np.unique(train_mask))
    y_train = np.eye(num_classes)[np.array(train_mask)]

    train_elements = round(len(X_train) * 0.8)
    X_val = X_train[train_elements:]
    y_val = y_train[train_elements:]
    X_train = X_train[:train_elements]
    y_train = y_train[:train_elements]

    X_test = np.array(test_img, dtype='float') / 255
    X_test = (X_test - X_mean) / X_std
    y_test = np.eye(num_classes)[np.array(test_mask)]

    return X_train, y_train, X_val, y_val, X_test, y_test

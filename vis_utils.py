"""This module makes some visualization."""

import matplotlib.pyplot as plt


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss)
    if val_loss is not None:
        plt.plot(epochs, val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epochs, acc)
    if val_acc is not None:
        plt.plot(epochs, val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

import numpy as np
from matplotlib import pyplot as plt

def plot_loss(train_loss, valid_loss, save_filename, loss_prefix=""):
    '''
    Plots training and validation loss over the course of training
    '''
    epochs = np.arange(1, len(valid_loss)+1)
    steps = np.arange(1, len(train_loss)+1)
    steps_per_epoch = steps[-1] / epochs[-1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.plot((steps / steps_per_epoch), train_loss, label="Training")
    ax.plot(epochs, valid_loss, label="Validation")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Epochs trained")
    ax.set_ylabel(loss_prefix+" Loss")
    
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_filename, dpi='figure')
    plt.close()
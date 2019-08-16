import numpy as np
import matplotlib.pyplot as plt

def error_rate(predictions, labels):
    """Return the error rate"""
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]
    return 100.0 * (1  - float(correct) / float(total))

def confusions(predictions, labels):
    "Return the confusions"
    confusions = np.zeros([10, 10], np.float32)
    bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    return confusions

def plot_confusions(grid, ax = None):
    
    ax = plt.subplot(111) if ax is None else ax
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(False)
    ax.set_xticks(np.arange(grid.shape[0]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.imshow(grid, interpolation='nearest');
    
    for i, cas in enumerate(grid):
        for j, count in enumerate(cas):
            if count > 0:
                xoff = .07 * len(str(count))
                plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')

    return ax

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

def preprocess(data, colInds=None):
    """ 
    Handle extracting / sorting gene columns in data, and normalizing
    data by doing log(1 + data)
    """
    data = np.atleast_2d(data)
    if colInds is None:
        colInds = np.where(np.sum(data, axis=0) != 0.0)[0]
    return np.log1p(data[:, colInds])


def visualize_data(data, labels, ax=None, cbar_width=0.1):
    """ Utility to neatly plot data matrix with labels """
    ax = plt.subplot(111) if ax is None else ax
    l_max, l_min, d_max, d_min = labels.max(), labels.min(), data.max(), data.min()
    labels = (labels - l_min) * (d_max - d_min) / (l_max - l_min) + d_min
    cbar = np.repeat(labels[:, None], int(data.shape[1] * cbar_width) , axis=1)
    ax.imshow(np.hstack((data, cbar)))
    ax.set_xlabel('Gene index (colorbar at end is label)')
    ax.set_ylabel('Cell index')
    return ax

def error_rate(predictions, labels):
    """Return the error rate (fraction of samples misclassified)"""
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]
    return (1  - float(correct) / float(total))

def confusions(predictions, labels):
    """Return the confusions matrix"""
    confusions = np.zeros([10, 10], np.float32)
    bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    return confusions

def perf_metric(predictions, labels, metric='precision'):
    """ 
    A generic function to return a performance metric given the 
    predictions and labels.

    Supported metrics:
        - accuracy
        - error
        - precision (per class)
        - recall (per class)
    """
    conf = confusions(predictions, labels)
    if metric == 'precision':
        return np.diag(conf) / np.sum(conf, axis=1)
    elif metric == 'recall':
        return np.diag(conf) / np.sum(conf, axis=0)
    elif metric == 'accuracy':
        return np.sum(np.diag(conf)) / np.sum(conf)
    elif metric == 'error':
        return 1 - (np.sum(np.diag(conf)) / np.sum(conf))

def plot_confusions(grid, ax = None):
    """ Utility to neatly plot confusions matrix. """
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

def jensen_shannon(ps, weights=None):
    """ Given a list of probability distributions ps, this will calculate
    the jensen_shannon divergence a.k.a. the information radius, an extension
    of the idea of KL divergence to multiple distributions. 
    
    Arguments:
        ps - a list of probability distributions, if all are not the same length
             then the short ones will be padded with 0s. Will be normalized
             (each distribution sums to 1) if not already.
        weights - a len(ps) array of weights, must sum to 1. Can leave as None
                  to weight all distributions equally
    """
    if weights is None:
        weights = np.full(len(ps), 1 / len(ps))
    tot = np.max([len(p) for p in ps])
    ps = np.array([np.concatenate((p, np.zeros(tot - p.shape[0]))) for p in ps])
    ps = ps / np.sum(ps, axis=1)[:, None]
    t1 = scipy.stats.entropy(np.sum(ps * weights[:, None], axis=0))
    t2 = np.dot(weights, [scipy.stats.entropy(p) for p in ps])
    return t1 - t2

def gene_divergence(gene_vals, labels):
    """ 
    Given the counts of a single gene across all cells, and the
    labels of the cells, this calculates the jensen_shannon divergence
    for the distributions of the count of this gene in each cell class.
    """
    counts = [gene_vals[labels == l] for l in np.unique(labels)]
    distributions = [np.bincount(val) for val in counts]
    return jensen_shannon(distributions)



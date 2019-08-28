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

def jensen_shannon(ps, weights=None):
    if weights is None:
        weights = np.full(len(ps), 1 / len(ps))
    tot = np.max([len(p) for p in ps])
    ps = np.array([np.concatenate((p, np.zeros(tot - p.shape[0]))) for p in ps])
    ps = ps / np.sum(ps, axis=1)[:, None]
    t1 = scipy.stats.entropy(np.sum(ps * weights[:, None], axis=0))
    t2 = np.dot(weights, [scipy.stats.entropy(p) for p in ps])
    return t1 - t2

def gene_relevance_metric(gene_vals, labels):
    counts = [gene_vals[labels == l] for l in np.unique(labels)]
    distributions = [np.bincount(val) for val in counts]
    return jensen_shannon(distributions)

#%timeit gene_relevance_metric(data[:, 0].astype(int), labels.squeeze().astype(int))
#
#data_f = data.astype(int)
#labels_f = labels.squeeze().astype(int)
#gene_scores = np.array([gene_relevance_metric(data_f[:, i], labels_f) for i in tnrange(data_f.shape[1])])
#
#gene_scores
#
#inds = np.argsort(gene_scores)[::-1]
#
#data_shuffled = data[:, inds]
#data_shuffled = np.log1p(data_shuffled)
#
## Visualize data with labels
#labels_scaled = labels * (data_shuffled.max() - data_shuffled.min()) / labels.max() - data_shuffled.min()
#plt.imshow(np.hstack((data_shuffled, np.repeat(labels_scaled, data.shape[1] //  10, axis=1))))
#plt.xlabel("Gene index (Colorbar at end is label)")
#plt.ylabel("Cell index");

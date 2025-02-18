from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.util.stats import cov
import matplotlib

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True,label='',alpha=1.,ax=None,artists=None):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2
    ax = ax if ax else plt.gca()

    # TODO use artists!

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if artists is None:
        point = ax.scatter([mu[0]],[mu[1]],marker='D',color=color,s=4,alpha=alpha) \
                if centermarker else None
        line, = ax.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',
                linewidth=2,color=color,label=label,alpha=alpha)
    else:
        line, point = artists
        if centermarker:
            point.set_offsets(np.atleast_2d(mu))
        line.set_xdata(ellipse[0,:] + mu[0])
        line.set_ydata(ellipse[1,:] + mu[1])
        line.set_alpha(alpha)
        line.set_color(color)

    return line, point


def plot_gaussian_projection(mu, lmbda, vecs, **kwargs):
    '''
    Plots a ndim gaussian projected onto 2D vecs, where vecs is a matrix whose two columns
    are the subset of some orthonomral basis (e.g. from PCA on samples).
    '''
    return plot_gaussian_2D(project_data(mu,vecs),project_ellipsoid(lmbda,vecs),**kwargs)


def pca_project_data(data,num_components=2):
    # convenience combination of the next two functions
    return project_data(data,pca(data,num_components=num_components))


def pca(data,num_components=2):
    U,s,Vh = np.linalg.svd(cov(data))
    return Vh.T[:,:num_components]


def project_data(data,vecs):
    return np.dot(data,vecs)


def project_ellipsoid(ellipsoid,vecs):
    # vecs is a matrix whose columns are a subset of an orthonormal basis
    # ellipsoid is a pos def matrix
    return np.dot(vecs,np.dot(ellipsoid,vecs.T))


def subplot_gridsize(num):
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))

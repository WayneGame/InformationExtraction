import seaborn as sns
from matplotlib.colors import Normalize
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 100, "scatter.edgecolors": "k"},
)
cache = {}


def plot_support_vectors(X, y, clf, title=None):
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure()
    plt.clf()

    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=90,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors="k")

    plt.axis("tight")

    x_min, x_max = X[:, 0].min() * 0.90, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 0.90, X[:, 1].max() * 1.1

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(
        XX,
        YY,
        Z,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        levels=[-1, 0, 1],
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    if title:
        plt.title(title)
    plt.show()


class MidpointNormalize(Normalize):
    """Source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html"""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def visualize_gridsearch(
    gs, range_a, range_b, xlabel, ylabel, title, logx=False, fignum="121", cbar=False
):
    scores = gs.cv_results_["mean_test_score"].reshape(len(range_a), len(range_b))
    plt.subplot(fignum)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(vmin=0.1, midpoint=0.95),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if cbar:
        plt.colorbar()
    plt.yticks(
        np.arange(len(range_a)), ["$2^{" + str(int(x)) + "}$" for x in np.log2(range_a)]
    )
    if logx:
        plt.xticks(
            np.arange(len(range_b)),
            ["$2^{" + str(int(x)) + "}$" for x in np.log2(range_b)],
            rotation=45,
        )
    else:
        plt.xticks(np.arange(len(range_b)), range_b, rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Plot a set of digits
def plot_digits(data, title):
    fig, axes = plt.subplots(
        4,
        10,
        figsize=(10, 4),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8), cmap="binary", interpolation="nearest", clim=(0, 16)
        )

    plt.suptitle(title)
    plt.show()


def plot_decision_boundary(model, X, y):
    """Create a decision boundary plot that shows the predicted label for each point."""
    h = 0.05  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    k = model.n_neighbors
    if k in cache:
        Z = cache[k]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        cache[k] = Z

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    plt.title("2D Iris KNN Decision Boundaries ($k=%s$)" % k)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
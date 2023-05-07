import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

matplotlib.use('Agg')
import seaborn as sns


def loss_plot(loss, name):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(name)


def tsne_plot(labels, embeddings, perplexity, n_components=2, n_iter=2048):
    """Creates and TSNE model and plots it

    Args:
        labels (Tensor): [stocks]
        embeddings (Tensor): [stocks, features]
    """

    tsne_model = TSNE(perplexity=perplexity,
                      n_components=n_components,
                      n_iter=n_iter,
                      init='pca',
                      random_state=19)
    new_values = tsne_model.fit_transform(embeddings)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('tsne.png', bbox_inches='tight')
    # plt.show()
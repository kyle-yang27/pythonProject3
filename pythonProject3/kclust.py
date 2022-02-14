"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #4: K-means Clustering
Steve Braich
Assignment Description:
https://web.cecs.pdx.edu/~singh/courses/winter21/dl/a4w21.pdf
k-means clustering: From the MNIST data set, pick 100 samples from each of the 10 classes.
Take all these 1,000 images and run them through a k-means clustering algorithm (k=10). You
can use the scikit-learn library.
Sources:
   scikit-learn 2.3.2 K-means
   https://scikit-learn.org/stable/modules/clustering.html#k-means
   A demo of K-Means clustering on the handwritten digits data
   https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
   K-Means Clustering with Scikit-Learn
   https://stackabuse.com/k-means-clustering-with-scikit-learn/
   Transpose Convolutions and Autoencoders
   https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
"""

import numpy as np
from time import time
from sklearn import metrics
from sklearn import utils
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Type of Machine: {device}")

class Cluster():

    def __init__(self, data=None, labels=None) -> None:
        if data is None:
            self.bind_data()
        else:
            self.data = data
            self.labels = labels
            self.n_samples = 10000
            self.n_features = 64
            self.n_digits = 10
            print(f"# digits: {self.n_digits}; # samples: {self.n_samples}; # features {self.n_features}")

    def bind_data(self) -> None:

        data, labels = load_digits(return_X_y=True)
        # X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        # data, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        labels = labels.astype(float)
        # mnist = fetch_openml(data_id=554, return_X_y=True, as_frame=False)
        # data, labels = mnist.data, mnist.target.astype(int)

        self.data = []
        self.labels = []
        for index in range(0, 10):
            label_indices = np.argwhere(labels == index)[:100]
            self.data.extend(data[label_indices.ravel()])
            self.labels.extend(np.full(shape=100, fill_value=index))

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        self.data, self.labels = utils.shuffle(self.data, self.labels)

        # 1000, 64
        (self.n_samples, self.n_features) = self.data.shape
        # 10
        self.n_digits = np.unique(self.labels).size

        print(f"# digits: {self.n_digits}; # samples: {self.n_samples}; # features {self.n_features}")

    def bench_k_means(self, kmeans, name, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.
        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        name : str
            Name given to the strategy. It will be used to show the results in a
            table.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        t0 = time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        fit_time = time() - t0
        results = [name, fit_time, estimator[-1].inertia_]

        # Define the metrics which require only the true labels and estimator
        # labels
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # The silhouette score requires the full dataset
        results += [
            metrics.silhouette_score(data, estimator[-1].labels_,
                                     metric="euclidean", sample_size=300, )
        ]

        # Show the results
        formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                            "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
        print(formatter_result.format(*results))

        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        #cm = confusion_matrix(truth, k_labels_matched)
        cm = confusion_matrix(labels, estimator[-1].labels_)

        # Plot confusion matrix
        plt.imshow(X=cm, interpolation='none', cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, z, ha='center', va='center')
        plt.title(f"{name}")
        plt.xlabel("kmeans label")
        plt.ylabel("truth label")
        plt.show()

    def evaluate(self):
        # %%
        # Run the benchmark
        # -----------------
        #
        # We will compare three approaches:
        #
        # * an initialization using `kmeans++`. This method is stochastic and we will
        #   run the initialization 4 times;
        # * a random initialization. This method is stochastic as well and we will run
        #   the initialization 4 times;
        # * an initialization based on a :class:`~sklearn.decomposition.PCA`
        #   projection. Indeed, we will use the components of the
        #   :class:`~sklearn.decomposition.PCA` to initialize KMeans. This method is
        #   deterministic and a single initialization suffice.

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        kmeans = KMeans(init="k-means++", n_clusters=self.n_digits, n_init=4, random_state=0)
        self.bench_k_means(kmeans=kmeans, name="k-means++", data=self.data, labels=self.labels)

        kmeans = KMeans(init="random", n_clusters=self.n_digits, n_init=4, random_state=0)
        self.bench_k_means(kmeans=kmeans, name="random", data=self.data, labels=self.labels)

        pca = PCA(n_components=self.n_digits).fit(self.data)
        kmeans = KMeans(init=pca.components_, n_clusters=self.n_digits, n_init=1)
        self.bench_k_means(kmeans=kmeans, name="PCA-based", data=self.data, labels=self.labels)

        print(82 * '_')

    def visualize(self):
        # %%
        # Visualize the results on PCA-reduced data
        # -----------------------------------------
        #
        # :class:`~sklearn.decomposition.PCA` allows to project the data from the
        # original 64-dimensional space into a lower dimensional space. Subsequently,
        # we can use :class:`~sklearn.decomposition.PCA` to project into a
        # 2-dimensional space and plot the data and the clusters in this new space.

        reduced_data = PCA(n_components=2).fit_transform(self.data)
        kmeans = KMeans(init="k-means++", n_clusters=self.n_digits, n_init=4)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
                  "Centroids are marked with white cross")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

class Autoencoder(nn.Module):
    def __init__(self):
        self.bind_data()

        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def bind_data(self, data=None, labels=None) -> None:

        # data, labels = load_digits(return_X_y=True)
        mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

        mnist_data = list(mnist_data)

        # self.data = mnist_data
        data, labels = list(map(list, zip(*mnist_data)))
        data = np.array(data)
        labels = np.array(labels)

        self.data = []
        self.labels = []
        for index in range(0, 10):
            label_indices = np.argwhere(labels == index)[:1000]
            self.data.extend(data[label_indices.ravel()])
            self.labels.extend(np.full(shape=1000, fill_value=index))

        self.data, self.labels = utils.shuffle(self.data, self.labels)

        self.n_samples = 10000
        self.n_features = 64
        self.n_digits = 10

    def fit(self, model, num_epochs=5, batch_size=64, learning_rate=1e-3):
        self.epochs = num_epochs

        model.to(device)
        torch.manual_seed(42)
        criterion = nn.MSELoss()  # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # <--

        train_loader = torch.utils.data.DataLoader(self.mnist_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        outputs = []
        for epoch in range(num_epochs):
            for data in train_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # img = data

                recon = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))

            # added labels to this below?
            outputs.append((epoch, images, recon), )
        return outputs

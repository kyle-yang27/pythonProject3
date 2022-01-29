from keras.datasets import mnist
import numpy as np
import tensorflow
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Load the data to be used for the Plot and Kmeans clustering
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_data = train_X
train_labels = train_y

# Print the shape of the dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# Assign classes that fit datapoints with labels 0-9
label_0 = train_data[train_labels==0]
label_1 = train_data[train_labels==1]
label_2 = train_data[train_labels==2]
label_3 = train_data[train_labels==3]
label_4 = train_data[train_labels==4]
label_5 = train_data[train_labels==5]
label_6 = train_data[train_labels==6]
label_7 = train_data[train_labels==7]
label_8 = train_data[train_labels==8]
label_9 = train_data[train_labels==9]

#Apply KMeans clustering onto classes of all labels, and calculate the centroid out of 10 clusters.
kmeans_0 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_0)
sample_0 = kmeans_0.cluster_centers_
kmeans_1 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_1)
sample_1 = kmeans_1.cluster_centers_
kmeans_2 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_2)
sample_2 = kmeans_2.cluster_centers_
kmeans_3 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_3)
sample_3 = kmeans_3.cluster_centers_
kmeans_4 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_4)
sample_4 = kmeans_4.cluster_centers_
kmeans_5 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_5)
sample_5 = kmeans_5.cluster_centers_
kmeans_6 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_6)
sample_6 = kmeans_6.cluster_centers_
kmeans_7 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_7)
sample_7 = kmeans_7.cluster_centers_
kmeans_8 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_8)
sample_8 = kmeans_8.cluster_centers_
kmeans_9 = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(label_9)
sample_9 = kmeans_9.cluster_centers_

#Concatenate centroids into a single array, as prototypes
proto_sample = np.concatenate((sample_0,sample_1,
                               sample_2,sample_3,
                               sample_4,sample_5,
                               sample_6,sample_7,
                               sample_8,sample_9
                              ), axis = 0
                             )

# Generate labels for the prototypes of the sample.
sample_0_labels = np.full((10,), 0)
sample_1_labels = np.full((10,), 1)
sample_2_labels = np.full((10,), 2)
sample_3_labels = np.full((10,), 3)
sample_4_labels = np.full((10,), 4)
sample_5_labels = np.full((10,), 5)
sample_6_labels = np.full((10,), 6)
sample_7_labels = np.full((10,), 7)
sample_8_labels = np.full((10,), 8)
sample_9_labels = np.full((10,), 9)

#Contatenate the labels of the sample into a single array
proto_labels = np.concatenate((sample_0_labels,sample_1_labels,
                               sample_2_labels,sample_3_labels,
                               sample_4_labels,sample_5_labels,
                               sample_6_labels,sample_7_labels,
                               sample_8_labels,sample_9_labels),
                              axis = 0
                             )

# Define a function that randomly chooses an M number of training points as prototypes for the KNN classification
#fof MNIST
def rand_prototypes(M):
    """
    rand_prototypes() takes in a parameter M and returns M. M is an int.
    M is the number of data points to be samples.
    The function rand_prototypes returns numpy arrays train_data and train_labels of M randomly chosen
    datapoints and labels.
    """

    indices = np.random.choice(len(train_labels), M, replace=False)
    return train_data[indices, :], train_labels[indices]

#Define a function "comparison()" that compares the output of the random selection classification with the
#Kmeans classification
#Initiailize the function
@interact_manual(M=(10000, 6000, 1000), rounds=(1, 10))
def comparison(M, rounds):
    """
    comparison() takes in a value M and rounds and returns the mean-squared error of both randomly selected prototypes
    and KMeans clustered prototypes.
    Calculations are done #rounds times to improve randomness in the study.
    Both input parameters are ints.
    """
    print("Comparing your prototype selection method to random prototype selection...")
    rand_err, rand_std = mean_squared_error(rand_prototypes, M, rounds)
    my_err, my_std = mean_squared_error(my_prototypes, M, rounds)

    print;
    print("Number of prototypes:", M)
    print("Number of trials:", rounds)
    print("Error for random prototypes:", rand_err)
    print("Error for your prototypes:", my_err);
    print
    if rand_err < my_err:
        print("The randomly selected prototypes have lower error.")
    else:
        print("The KMeans clustered prototypes have lower error.")

# Plotting the images from the MNIST dataset
for i in range(9):
    pyplot.subplot(330 + 1 + i)
pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
#K-means clustering

import numpy as np
import matplotlib.pyplot as plt

#the same old dataset!
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()
data = scale(digits.data)

#In case you wanna take a look at the images
def print_digits(images, y, max_n = 10):
    #set up the figure size in inches
    fig = plt.figure(figsize = (12, 12))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    i=0
    while i < max_n and i < images.shape[0]:
        #plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i+1, xticks = [], yticks = [])
        p.imshow(images[i], cmap = plt.cm.bone)
        #label the image with the target value
        p.text(0, 14, str(y[i]))
        i = i+1
        
print_digits(digits.images, digits.target, max_n = 10)

#now let's seperate the training set and the testing set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size = 0.25, random_state = 42)
n_samples, n_features = x_train.shape
n_digits = len(np.unique(y_train))
labels = y_train

#In case u don't know what K-means does:
#1. randomn selection of cluster centers
#2. Find the nearest cluster center for each data point, assignment
#3. Compute new cluster centers, iterations

from sklearn import cluster
clf = Cluster.KMeans(init = 'kmeans++', n_custers = 10, randomn_state = 42)
clf.fit(x_train)

#to know what the clustering has resulted in:
print_digits(images_train, clf.labels_, max_n=10)

#IMPORTANT - the cluster number has got nothing to do with the numerical value of the digit

#Let's get on with the unsupervised prediction
y_pred = clf.predict(x_test)

#Let's have a look at the predictions

def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred == cluster_number]
    y_pred = y_pred[y_pred == cluster_number]
    print_digits(images, y_pred, max_n = 10)

for i in range(10):
    print_cluster(images_test, y_pred, i)


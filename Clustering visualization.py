#Run the K-means and K-means evaluation scripts before running this.

#2D visualization of the K-means clustering

#We'll again use PCA for this job

from sklearn import decomposition
pca = decomposition.PCA(n_components = 2).fit(x_train)
reduced_x_train = pca.transform(x_train)

#step size of the mesh

h = .01

#point in the mesh [x_min, x_max]x[y_min, y_max]

x_min, x_max = reduced_x_train[:, 0].min() + 1, reduced_x_train[:, 0].max() - 1
y_min, y_max = reduced_x_train[:, 1].min() + 1, reduced_x_train[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
kmeans = cluster.KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10)
kmeans.fit(reduced_x_train)
z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

#put the result into a color pot

z = z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(z, interpolation = 'nearest', extent = (xx.min(), xx.max(), yy.min(), yy.max()), cmap = plt.cm.Paired, aspect = 'auto', origin = 'lower')
plt.plot(reduced_x_train[:, 0], reduced_x_train[:, 1], 'k.', markersize = 2)

#plot the centroids as a white X

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = '.', s = 169, linewidths = 3, color = 'w', zorder = 10)
plt.title('K-means clustering on digits dataset')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
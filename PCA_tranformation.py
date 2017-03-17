#dataset consisting handwritten digits is imported from scikit learn dataset library

from sklearn.datasets import load_digits
digits = load_digits() #dataframe created
x_digits, y_digits = digits.data, digits.target #prime features of the dataframe are stored in two vectors

#just in case you wanna take a look at all the components of the dataframe
print(digits.keys())

import matplotlib.pyplot as plt #plotting library introduced

#now we try to visualize the instances of digits
n_row, n_col = 2, 5

def print_digits(images, y, max_n=10):
    #set up the figure size in inches
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    i=0
    while i < max_n and i < images.shape[0]:
        p = fig.add_subplot(n_row, n_col, i+1, xticks = [], yticks = [])
        p.imshow(images[i], cmap = plt.cm.bone, interpolation = 'nearest')
        #label the images with the target value
        p.text(0, -1, str(y[i]))
        i=i+1
        
#call the defined function to visualize
print_digits(digits.images, digits.target, max_n = 10)

#now define a funtion that will plot a scatter with the 2D points that will be obtained after the PCA transformation

from sklearn.decomposition import PCA
estimator = PCA(n_components = 10)
x_pca = estimator.fit_transform(x_digits)

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'grey']
    #use this array to impart colors to different legends of the scatter
    for i in range(len(colors)):
        px = x_pca[:,0][y_digits == i]
        py = x_pca[:,1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

#now use the plotting function to visualize
plot_pca_scatter()

#Now we'll take the principal components from the estimator by accesing the'components' attribute
#And we'll plot the newly formed components in the same shape as the original data

def print_pca_components(images, n_col, n_row):
    plt.figure(figsize = (2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(comp.reshape((8, 8)), interpolation = 'nearest')
        plt.text(0, -1, str(i+1) + '-component')
        plt.xticks(())
        plt.yticks(())
        
#now visualize the dimensionally reduced forms of the labels

print_pca_components(estimator, n_col, n_row)   


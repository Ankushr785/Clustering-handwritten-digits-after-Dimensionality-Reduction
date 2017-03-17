#Run the K-means.py script first to get anything out of the following code!
    
#K-means evaluation
    
from sklearn import metrics
print("Adjusted rand score: {: .2}".format(metrics.adjusted_rand_score(y_test, y_pred)))

#adjusted RAND index is a measure for accuracy, which doesn't change even if we change class names.

#want a confusion matrix as well?

print(metrics.confusion_matrix(y_test, y_pred))
 
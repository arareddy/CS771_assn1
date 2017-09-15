from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

def predict(Xtr, Ytr, Xts, metric=None):

    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros((Xts.shape[0], 1))

    k = 10; #Value of k by tuning
    M = Xtr; # This matrix is useful for intermediate computation
    Mahalanobis_norm = np.zeros(Xtr.shape[0]); # This matrix stores the Mahalanobis norm for the difference between one test point and all training points.
    indices_closest = np.zeros(k); # This vector stores the indices of the k nearest neighbours in the training data.
    vote = np.zeros(3); #This function is used for finding majority label among the k-nearest neighbours.
    
    for i in range(Xts.shape[0]):
        M = M - Xtest[i,0:]; #By broadcasting, we get a matrix in which every row is row-i'th training point
        G_temp = np.matmul(M,M.T);
        Mahalanobis_norm = np.matmul(M,G_temp); # Vector consisting of Mahalanobis norm from all the training points.
        indices_closest = np.argpartition(Euclidean_norm, k)[:k];   
        for j in range(0,k): 
            vote[int(Ytr[indices_closest[j]])-1]+=1 # -1 is required because the labels are 1,2,3 and not 0,1,2
        Yts[i] = np.argmax(vote)+1;
    
    return Yts

def main(): 

    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]
    testdatafile = sys.argv[2]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];

    # The testing file is in libSVM format too
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray();
    # The test labels are useless for prediction. They are only used for evaluation

    # Load the learned metric
    metric = np.load("model.npy")

    ### Do soemthing (if required) ###

    Yts = predict(Xtr, Ytr, Xts)
    #Yts = predict(Xtr, Ytr, Xts, metric)

    # Save predictions to a file
	# Warning: do not change this file name
    np.savetxt("testY.dat", Yts)

if __name__ == '__main__':
    main()

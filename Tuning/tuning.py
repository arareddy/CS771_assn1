import numpy as np
import sys
from modshogun import LMNN, RealFeatures, MulticlassLabels
from sklearn.datasets import load_svmlight_file

training_data_size = 50000;
test_data_size = 10000; #validation data

# The training file is in libSVM format
total_data = load_svmlight_file("train.dat");

Xto = total_data[0].toarray(); # Converts sparse matrices to dense
Yto = total_data[1]; # The training labels, these are either 1,2 or 3

total_data_matrix = np.zeros((60000,101));
total_data_matrix[:,0] = Yto;
total_data_matrix[:,1:] = Xto;

np.random.shuffle(total_data_matrix);

#total_data = np.load("shuffled_training_data.npy");

tr_data = total_data_matrix[:50000,0:]

Xtr = tr_data[:,1:]; 
Ytr = tr_data[:,0]; # The training labels, these are either 1,2 or 3

test_data = total_data[50000:,0:];

Xtest = test_data[:,1:]; #Coordinates of the test data 
Ytest = test_data[:,0]; # The test labels

# Number of target neighbours per example - tune this using validation
for k in range(1,21):
    
    Predicted_labels = Ytest - 4; #Uncomputed Predicted_labels are negative
    Test_accuracy = 0;
    Success_cases = 0;
    M = Xtr; # This matrix is useful for intermediate computation
    Euclidean_norm = np.zeros(training_data_size); # This vector stores the Euclidean norm from each test point.
    indices_closest = np.zeros(k); # This vector stores the indices of the k nearest neighbours in the training data.
    i = 0; #Counters
    j = 0;
    vote = np.zeros(3); #This function is used for finding majority label among the k-nearest neighbours.

    for i in range(0,test_data_size):
        M = M - Xtest[i,0:]; #By broadcasting, we get a matrix in which every row is row-i'th training point
        Euclidean_norm = np.linalg.norm(M,axis=1); # Vector consisting of Enorm from all the training points.
        indices_closest = np.argpartition(Euclidean_norm, k)[:k]; 
        for j in range(0,k):
            vote[int(Ytr[indices_closest[j]])-1]+=1 # -1 is required because the labels are 1,2,3 and not 0,1,2
        Predicted_labels[i] = np.argmax(vote)+1;
        vote = np.zeros(3); #Reset vote
        M = Xtr; #Reset M for new test point.

    Success_cases = np.sum(Predicted_labels==Ytest); 
    Test_accuracy = Success_cases/test_data_size;

    with open("tuning_results.txt", "a") as myfile:
        myfile.write("%d %f\n" % (k,Test_accuracy));
    print(Success_cases);
    #np.savetxt("labels_k10",Predicted_labels);

import numpy as np
import sys
from modshogun import LMNN, RealFeatures, MulticlassLabels
from sklearn.datasets import load_svmlight_file

# The training file is in libSVM format
tr_data = load_svmlight_file("train.dat");

Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
Ytr = tr_data[1]; # The training labels, these are either 1,2 or 3

new_matrix = np.zeros((60000,101));
new_matrix[:,0] = Ytr;
new_matrix[:,1:] = Xtr;

np.random.shuffle(new_matrix);

np.save('shuffled_training_data',new_matrix);


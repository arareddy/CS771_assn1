import numpy as np
import sys
from metric_learn import ITML_Supervised
from sklearn.datasets import load_svmlight_file

def main(): 

    # Get training file name from the command line
    traindatafile = sys.argv[1]

	# The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);

    Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
    Ytr = tr_data[1]; # The trainig labels
    Xtr_train = Xtr[:6000];
    Ytr_train = Ytr[:6000];
    itml = ITML_Supervised();
    itml.fit(Xtr_train,Ytr_train);
    Met = itml.metric()
    # print Met;
    np.save("itml_model.npy", Met) 

if __name__ == '__main__':
    main()

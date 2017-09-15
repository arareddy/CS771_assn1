import numpy as np
from sklearn.datasets import load_svmlight_file

testdata = load_svmlight_file("assn1data/test.dat");
Yts = testdata[1];
predict = np.loadtxt("testY.dat");

#Yts = Yts[:100];
#predict = predict[:100];

Success_cases = np.sum(predict==Yts)/200;
print(Success_cases);

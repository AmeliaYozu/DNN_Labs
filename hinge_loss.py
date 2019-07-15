#!usr/bin/env python3
from utils.datasets.datasetsLoader import SimpleDatasetLoader
from utils.preprocessors.preprocessor import SimplePreprocessor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import pdb
#import argparse as ap


def hinge_loss_per_example(Y, S):
	L = []
	for i in range(Y.shape[0]):
		s = S[:,i]
		y = Y[i]
		Li = 0
		for (j, sj) in enumerate(s):
			if j!=y:
				Li+=max(0,sj-s[y]+1)
		L.append(Li)
	loss = sum(L)/Y.shape[0]
	return (L,loss)


if __name__ == "__main__":
	np.random.seed(1)

	# load from dataset
	p = SimplePreprocessor(32, 32)
	d = SimpleDatasetLoader("./datasets/animals", p)
	# X [32*32*3]
	(X, Y) = d.load(verbose=500)

	# initialize from random
	W = np.random.randn(3,3072)
	b = np.random.randn(3,1)

	X = X.reshape(X.shape[0],3072)
	le = LabelEncoder()
	Y = le.fit_transform(Y)

	X_trans = X.transpose()

	#Scoring function
	S = W.dot(X_trans)+b
	(L,loss) = hinge_loss_per_example(Y, S)
	print("The hinge loss is {}".format(loss))


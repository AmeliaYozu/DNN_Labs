from utils.preprocessors.preprocessor import SimplePreprocessor
from utils.datasets.datasetsLoader import SimpleDatasetLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np


def cross_entropy_loss(Y, S):
	L = []
	for i in range(Y.shape[0]):
		s = S[:,i]
		y = Y[i]
		l = np.exp(s[y])/np.sum(np.exp(s))
		L.append(l)
	loss = -np.log(np.sum(L)/Y.shape[0])
	return (L, loss)


if __name__ == "__main__":
	np.random.seed(1)
	sp = SimplePreprocessor(32,32)
	sd = SimpleDatasetLoader("./datasets/animals", sp)
	(data, labels) = sd.load(verbose=1000)
	
	K = 3
	N = data.shape[0]
	D = 32*32*3

	#flatten X from N * (32*32*3) to N * 3072
	X = data.reshape(N,D)
	X = X/255
	le = LabelEncoder()
	Y = le.fit_transform(labels)

	

	#init W, b
	W = np.random.randn(K,D)
	b = np.random.randn(K,1)

	X_trans = X.transpose()

	scores = W.dot(X_trans)+b
	(L, loss) = cross_entropy_loss(Y, scores)
	print("The cross entropy loss is {}".format(loss))

	#check
	count = 0
	for i in range(N):
		s = scores[:,i]
		y = Y[i]
		if s[y] == max(s):
			count += 1
	print("correctly predicted: {}/{}".format(count,N))












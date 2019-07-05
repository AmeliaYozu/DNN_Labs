from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.preprocessors import SimplePreprocessor
from utils.datasets import SimpleDatasetLoader
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
args = vars(ap.parse_args())

sp = SimplePreprocessor(32,32)
sd = SimpleDatasetLoader(args['dataset'],sp)
print("[INFO] loading images...")
(X,y) = sd.load(verbose=500)
X = X.reshape((X.shape[0], 3072)) #3072=32*32*3, X.shape[0] is the number of data points

print("[INFO] features matrix: {:.1f}MB".format(X.nbytes/(1024*1000)))
le = LabelEncoder()
y = le.fit_transform(y)

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=-1)
model.fit(trainX,trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
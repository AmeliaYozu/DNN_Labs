from utils import preprocessor
import numpy as np
from imutils import paths
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, path, preprocessor):
		self.path = path
		self.preprocessor = preprocessor

	def datasetLoad(self):
		X = []
		y = []
		imagePaths = list(paths.list_images(path))
		for imagePath in imagePaths:
			img = cv2.imread(imagePath)
			img = preprocessor.preprocess(img)
			X.append(img)
			y.append(imagePath.split(os.path.sep)[-2])
		return (np.array(X),np.array(y))


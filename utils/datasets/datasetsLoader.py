import numpy as np
from imutils import paths
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, path, preprocessor):
		self.path = path
		self.preprocessor = preprocessor

	def load(self, verbose=-1):
		X = []
		y = []
		imagePaths = list(paths.list_images(self.path))
		for (i, imagePath) in enumerate(imagePaths):
			img = cv2.imread(imagePath)
			img = self.preprocessor.preprocess(img)
			X.append(img)
			y.append(imagePath.split(os.path.sep)[-2])

			if verbose>0 and (i+1)%verbose == 0:
				print("[INFO] proceeded {}/{}".format((i+1),len(imagePaths)))

		return (np.array(X),np.array(y))


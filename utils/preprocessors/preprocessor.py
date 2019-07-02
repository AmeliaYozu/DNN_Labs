#!/usr/bin/env python3
import cv2

class SimplePreprocessor:
	def __init__(self, weight, height, inter=cv2.INTER_AREA):
		self.weight = weight
		self.height = height
		self.inter = inter

	def preprocess(self,img):
		return cv2.resize(img, (self.weight, self.height), interpolation=self.inter)
		
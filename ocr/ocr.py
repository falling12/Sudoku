import logging
import os
import pickle
import shutil
import cv2
import numpy as np
import requests
from PIL import Image
from skimage import feature as ft
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class OCR:
	def __init__(self):
		self.svm = SVC(kernel='rbf', random_state=0, gamma='auto', C=1.0, probability=True)
		self.sc = StandardScaler()
		self.score_ = None

	def fit(self, X, y, test_size=0.3, sample_weight=None):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
															random_state=0)
		self.sc.fit(X_train)
		X_train_std = self.sc.transform(X_train)
		X_test_std = self.sc.transform(X_test)

		self.svm.fit(X_train_std, y_train, sample_weight=sample_weight)

		y_pred = self.svm.predict(X_test_std)
		self.score_ = accuracy_score(y_test, y_pred)

	def _transform(self, X):
		return self.sc.transform(X)

	def predict(self, X):
		return self.svm.predict(self._transform(X))

	def predict_proba(self, X):
		return self.svm.predict_proba(self._transform(X))

	def score(self):
		return self.score_


def del_blur(img):
	"""
	去除验证码中的干扰元素
	:param img: 必须为OpenCV的img对象
				 即使用cv2.imread('temp.png', 0)打开的图片0 代表以二值化方式打开
	:return:
	"""
	# 双边滤波 去噪 效果不错
	# m_blur = cv2.bilateralFilter(img, 9, 75, 75)
	# oust 滤波 二值化图片
	ret, oust_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	fin = cv2.bilateralFilter(oust_img, 9, 75, 75)
	return fin


def hog(img):
	"""
	提取图片hog特征（将图片信息降维 二维数组变一维数组）
	:param img: 图片文件的地址或numpy.array对象
	:return f: 图片hog特征数据（一维）
	"""
	if isinstance(img, str):
		img = Image.open(img)
	elif isinstance(img, np.ndarray):
		img = Image.fromarray(img)
	else:
		raise ValueError('the input img is not fill the bill! \n '
						 'it must be an image file path or numpy.ndarray')

	return ft.hog(img, block_norm='L2-Hys', pixels_per_cell=(10, 10), cells_per_block=(10, 10))


def load_data(relaod=False):
	"""
	加载训练数据
	:param relaod:
	:return:
	"""
	X_train = []
	y_train = []

	x_path = 'x_train.pickle'
	y_path = 'y_train.pickle'

	x_exist = os.path.exists(x_path)
	y_exist = os.path.exists(y_path)

	if not relaod and x_exist and y_exist:
		try:
			with open(x_path, 'rb') as f:
				X_train = pickle.load(f)
			with open(y_path, 'rb') as f:
				y_train = pickle.load(f)
		except EOFError:
			pass
		finally:
			logging.debug('load by pickle')
			return X_train, y_train

	for label in os.listdir(r'data'):
		label_path = os.path.join(os.path.abspath('.'), 'data', label)
		label_items = os.listdir(label_path)
		for item_name in label_items:
			item = os.path.join(label_path, item_name)
			img = cv2.imread(item, 0)
			feature = hog(del_blur(img))
			X_train.append(feature)
			y_train.append(label)

	with open(x_path, 'wb+') as f:
		pickle.dump(X_train, f)

	with open(y_path, 'wb+') as f:
		pickle.dump(y_train, f)

	logging.debug('load by file')
	return X_train, y_train


def load_ocr(reload=False):
	"""
	加载已经训练好的模型（如果有的话）
	:param reload:
	:return:
	"""
	ocr_path = os.path.join('O:\JetBrains\PycharmProjects\sudoku\ocr', 'number.ocr')
	if not reload and os.path.exists(ocr_path):
		ocr = joblib.load(ocr_path)
	else:
		ocr = OCR()
		X, y = load_data()
		ocr.fit(X, y)
		joblib.dump(ocr, ocr_path)
	return ocr


def main():
	ocr = load_ocr()
	url = 'http://jwxt.wust.edu.cn/whkjdx/verifycode.servlet?0.12337475696465894'
	con = requests.get(url).content
	with open('temp.png', 'wb') as f:
		f.write(con)
	img = cv2.imread('temp.png', 0)
	os.remove('temp.png')

	img_arr = del_blur(img)

	# hog_arr = [hog(x) for x in split_img(img_arr)]
	hog_arr = ''
	pred = ocr.predict(hog_arr)

	name = ''.join(pred) + '.png'
	with open(os.path.join('verifycode', name), 'wb') as f:
		f.write(con)


def load_pic():
	pics_path = 'O:\JetBrains\PycharmProjects\sudoku\spilt'
	pics = os.listdir(pics_path)
	for pic in pics:
		pic_path = os.path.join(pics_path, pic)
		cv_img = cv2.imread(pic_path, 0)
		feature = hog(del_blur(cv_img))
		ocr = load_ocr()
		x = ocr.predict(feature.reshape(1, -1))
		print(x, 'moved')
		shutil.move(pic_path, os.path.join('O:\JetBrains\PycharmProjects\sudoku\ocr\data', str(x[0])))


def predict(img_path):
	cv_img = cv2.imread(img_path, 0)
	feature = hog(del_blur(cv_img))
	ocr = load_ocr()
	answer = ocr.predict(feature.reshape(1, -1))
	return int(answer[0])


if __name__ == '__main__':
	ocr = load_ocr(reload=True)
	print(ocr.score())

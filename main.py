import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
from ocr.ocr import predict, OCR
from PIL import Image
import skimage.feature as ft
from functools import reduce
from common.screenshot import pull_screenshot, check_screenshot
from logic import solve
from common.auto_adb import auto_adb


adb = auto_adb()


# 获取图片分割位置
def get_grid(img):
	# im = Image.open(img)
	im = img.convert("RGB")
	w, h = im.size
	data = im.getdata()
	fl = map(lambda x: 255 if x in ((187, 187, 187), (186, 186, 186)) else 0, data)
	grid_arr = np.array(list(fl)).reshape(h, w)
	h_position = []
	w_position = []
	for x in range(h):
		if reduce(lambda x, y: x + y, list(grid_arr[x, :])) > 200000:
			h_position.append(x)

	for x in range(w):
		if reduce(lambda x, y: x + y, list(grid_arr[:, x])) > 200000:
			w_position.append(x)

	h_position = h_position[1:-1:3]
	w_position = w_position[1:-1:3]

	hh = min([h_position[x + 1] - h_position[x] for x in range(len(h_position) - 1)])
	ww = min([w_position[x + 1] - w_position[x] for x in range(len(w_position) - 1)])

	index = 0
	for h_index in range(len(h_position) - 1):
		for w_index in range(len(w_position) - 1):
			w0 = w_position[w_index]
			h0 = h_position[h_index]
			yield {'crop': (w0 + 5, h0 + 5, w0 + ww - 5, h0 + hh - 5), 'index': index,
				   'center': (w0 + ww / 2, h0 + hh / 2)}
			index += 1


def split_and_predict(screenshot: Image):
	grid_list = list(get_grid(screenshot))
	a = screenshot
	for x in grid_list:
		c = a.crop(x['crop'])
		splited_path = 'spilt\{}.png'.format(x['index'])
		c.save(splited_path)
		yield predict(splited_path), x['index']


def main():
	num = {
		1: (120, 1560),
		2: (336, 1560),
		3: (557, 1560),
		4: (768, 1560),
		5: (975, 1560),
		6: (120, 1720),
		7: (336, 1720),
		8: (557, 1720),
		9: (768, 1720)
	}
	check_screenshot()
	for i in range(101):
		screenshot = pull_screenshot()
		grids = list(get_grid(screenshot))
		question = get_sudoku_arr(screenshot)
		print(question)
		answer = []

		def get_ans(arr):
			nonlocal answer
			answer = arr[:]

		solve(question, finish_callback=get_ans)
		print('{} 开始填写答案....'.format(i))
		if not answer:
			print('答案为空 ')
			continue
		press(200, 148)
		for grid in grids:

			index = grid['index']
			if question[index] == 0:
				press(*grid['center'])
				press(*num[answer[index]])
		time.sleep(0.2)
		press(319, 910)
		print('作答完成')


def get_sudoku_arr(screenshot: Image):
	a = list(split_and_predict(screenshot))
	arr = []
	i = 0
	for answer, index in a:
		arr.append(answer)
		assert index == i, '下标错乱'
		i += 1
	return arr


def press(x, y):
	cmd = 'shell input tap {x} {y}'.format(x=x, y=y)
	adb.run(cmd)


def found_num():
	num = {
		1: (120, 1560),
		2: (336, 1560),
		3: (557, 1560),
		4: (768, 1560),
		5: (975, 1560),
		6: (120, 1720),
		7: (336, 1720),
		8: (557, 1720),
		9: (768, 1720)
	}
	img = Image.open('autojump.png')
	img.convert('L')
	print(type(img))


if __name__ == '__main__':
	# print(get_sudoku_arr())
	main()
# adb.run('shell input tap 1019 241 ')

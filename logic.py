import time

arr = [0, 0, 0, 0, 7, 0, 0, 8, 0,
	   0, 0, 3, 0, 0, 0, 1, 4, 0,
	   0, 6, 0, 0, 0, 3, 0, 0, 5,
	   0, 5, 6, 0, 8, 0, 9, 7, 0,
	   0, 4, 0, 5, 9, 6, 2, 0, 0,
	   0, 0, 2, 7, 3, 1, 0, 0, 0,
	   4, 1, 5, 8, 0, 0, 3, 9, 6,
	   8, 0, 7, 0, 1, 9, 4, 0, 2,
	   6, 0, 9, 3, 5, 0, 8, 0, 7]


def index_to_xy(index: int):
	"""
	:param index: 数组下标
	:return: x,y  x代表列 y代表行
	"""
	y, x = divmod(index, 9)
	return x + 1, y + 1


def xy_to_index(x: int, y: int):
	"""

	:param x: 代表列（横坐标）
	:param y: 代表行
	:return:  返回下标
	"""
	return (y - 1) * 9 + x - 1


def row(arr: list, index: int):
	"""
	根据下标来获取同行的所有元素(横着的)
	:param arr:
	:param index:
	:return:
	"""
	x, y = index_to_xy(index)
	return [arr[xy_to_index(m, y)] for m in range(1, 10)]


def line(arr: list, index: int):
	"""
	通过下标来获取同列元素（竖着的）
	:param arr:
	:param index:
	:return:
	"""
	x, y = index_to_xy(index)
	return [arr[xy_to_index(x, m)] for m in range(1, 10)]


def gong(arr: list, index: int):
	"""根据下标来获取宫"""

	def _(arr, m):
		m = m
		n = m + 1
		q = n + 1
		return [arr[m], arr[n], arr[q], arr[m + 9], arr[n + 9], arr[q + 9], arr[m + 18], arr[n + 18], arr[q + 18]]

	x, y = index_to_xy(index)
	if y <= 3:
		if x <= 3:
			# gong 1
			return _(arr, 0)
		elif x <= 6:
			# gong 2
			return _(arr, 3)
		elif x <= 9:
			# gong 3
			return _(arr, 6)
	elif y <= 6:
		if x <= 3:
			# gong 4
			return _(arr, 27)
		elif x <= 6:
			# gong 5
			return _(arr, 30)
		else:
			# gong 6
			return _(arr, 33)
	elif y <= 9:
		if x <= 3:
			# gong 7
			return _(arr, 54)
		elif x <= 6:
			# gong 8
			return _(arr, 57)
		elif x <= 9:
			# gong 9
			return _(arr, 60)


def get_cases(arr, index):
	a = {1, 2, 3, 4, 5, 6, 7, 8, 9}
	if arr[index] != 0:
		return []
	g = set(gong(arr, index))
	l = set(line(arr, index))
	r = set(row(arr, index))
	return list(a - (g | l | r))


def is_finish(arr):
	return 0 not in arr


start_time = time.time()
end_time = 0
correct = 0


def solve(arr, index=0, finish_callback=None):
	'''
	:param arr:
	:param index:
	:param finish_callback:
	def finish_callback(arr):
		pass
	:return:
	'''
	# 跳到零所在的位置
	while arr[index] != 0:
		index += 1
	for case in get_cases(arr, index):
		arr[index] = case
		if is_finish(arr):
			if finish_callback:
				finish_callback(arr)
			else:
				print('do not have finish callback')
				print_arr(arr)
		else:
			solve(arr, index + 1, finish_callback=finish_callback)

		arr[index] = 0


def print_arr(arr):
	arr = list(map(lambda x: str(x) if x != 0 else '_', arr))
	for x in range(0, 9):
		print('  '.join(arr[9 * x:9 * x + 9]))


de = [0, 8, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 7, 0, 0, 0, 0, 8, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
	  0, 0, 6, 0, 0, 9, 0, 0, 0, 0, 7, 2, 0, 1, 0, 0, 0, 9, 3, 2, 0, 0, 0, 6, 4, 8, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	  0, 5, 0, 0, 0]

void = [0 for x in range(0, 81)]

if __name__ == '__main__':
	s = time.time()
	ansq = []


	def callback(arrq):
		global ansq
		ansq = list.copy(arrq)

	solve(de, finish_callback=callback)


import cv2
import time
import operator
import numpy as np
from PIL import Image
import pytesseract

def display_image(img):
	cv2.imshow('image', img) 
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def scale_and_centre(img, size, margin=0, background=0):
	
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)

	return cv2.resize(img, (size, size))


def find_digit(inp_img, scan_tl=None, scan_br=None):
	
	img = inp_img.copy() 
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point


def process_image(img):
	
	"""Smoothing filter"""
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
	
	"""Threshold"""
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	
	"""Invert color"""
	proc = cv2.bitwise_not(proc, proc)
	
	"""Dilate gridlines"""
	kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
	kernel=kernel.astype(np.uint8)
	dilated = cv2.dilate(proc, kernel)
	
	"""Find outermost square"""
	contours, h = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True) 
	square = contours[0]
	
	"""Find corners"""
	br, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in square]), key=operator.itemgetter(1))
	tl, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in square]), key=operator.itemgetter(1))
	bl, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in square]), key=operator.itemgetter(1))
	tr, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in square]), key=operator.itemgetter(1))
	
	br = square[br][0]
	tl = square[tl][0]
	bl = square[bl][0]
	tr = square[tr][0]


	"""Crop and wrap"""
	long_side = max([euclidean(br, tr), euclidean(tl, bl), euclidean(br, bl), euclidean(tl, tr)])
	src = np.array([tl, tr, br, bl], dtype='float32')
	dst = np.array([[0, 0], [long_side - 1, 0], [long_side - 1, long_side - 1], [0, long_side - 1]], dtype='float32')
	ptrans = cv2.getPerspectiveTransform(src, dst)
	dilated = cv2.warpPerspective(dilated, ptrans, (int(long_side), int(long_side)))
	
	"""Infers cells"""
	small_squares = []
	small_side = long_side/ 9
	for j in range(9):
		for i in range(9):
			p1 = (i * small_side, j * small_side)  # Top left corner of a bounding box
			p2 = ((i + 1) * small_side, (j + 1) * small_side)  # Bottom right corner of bounding box
			small_squares.append((p1, p2))
	
	"""Get digits"""
	digit_imgs = []
	for small_square in small_squares:
		digit_img = proc[int(small_square[0][1]):int(small_square[1][1]), int(small_square[0][0]):int(small_square[1][0])]
		h, w = digit_img.shape[:2]
		margin = int(np.mean([h, w]) / 2.5)
		_, bbox, seed = find_digit(digit_img, [margin, margin], [w - margin, h - margin])
		digit_img = digit_img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
		w = bbox[1][0] - bbox[0][0]
		h = bbox[1][1] - bbox[0][1]
		if w > 0 and h > 0 and (w * h) > 100 and len(digit_img) > 0:
			digit_imgs.append(scale_and_centre(digit_img, 28, 4))
		else:
			digit_imgs.append(np.zeros((28, 28), np.uint8)

	return digit_imgs


def readSudoku(path):
	original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	digit_imgs = process_image(original)
	digits = []
	for digit_img in digit_imgs:
    	text = pytesseract.image_to_string(cv2.bitwise_not(digit_img), config="--psm 10 ")
    	valid = True
   		if (text.isdigit()):
    		if not (int(text) in range(1,10)):
        		valid = False
		elif (not text):
    		text = 0
    	else
    		valid = False

		if(valid):
    		digits.append(int(text)
		else:
    		print("Could not recognize board. Please try again. ")
          		return []
    return digits
	
digit_array = readSudoku('img.jpg')
print(digit_array)
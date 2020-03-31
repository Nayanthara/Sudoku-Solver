  #Robert Morgowicz (rjm448) and Nayanthara Sajan (ns933)
  #Wednesday Lab Section
  #Final Project Main code loop

  #imports

  #Reading
  import time
  import cv2
  import operator
  import numpy as np
  from PIL import Image
  import pytesseract
  from picamera import PiCamera


  #Solving
  from ctypes import *
  #Writing
  import spidev
  #Misc
  import RPi.GPIO as GPIO


  #Init
  #GPIO Setup
  GPIO.setmode(GPIO.BCM)
  #Buttons
  GPIO.setup(5, GPIO.IN)
  GPIO.setup(26, GPIO.IN)
  GPIO.setup(16, GPIO.IN)
  #LEDs
  GPIO.setup(6, GPIO.OUT)
  GPIO.setup(13, GPIO.OUT)
  GPIO.setup(19, GPIO.OUT)
  #SPI init
  bus = 0
  device = 0
  spi = spidev.SpiDev()
  spi.open(bus, device)
  # Set SPI speed and mode
  spi.max_speed_hz = 500000
  spi.mode = 0
  # Camera
  camera = PiCamera()
  #-----------------------------------------------------------------------
  #READING CODE: Functions and globals
  #-----------------------------------------------------------------------
  def show_image(img):
  	"""Shows an image until any key is pressed"""
  	cv2.imshow('image', img)  # Display the image
  	cv2.waitKey(0)
  	cv2.destroyAllWindows()  # Close all windows


  def show_digits(digits, colour=255):
  	"""Shows list of 81 extracted digits in a grid format"""
  	rows = []
  	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
  	for i in range(9):
  		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
  		rows.append(row)
  	show_image(np.concatenate(rows))

  def pre_process_image(img, skip_dilate=False):
  	"""Smoothing filter"""
  	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

  	# Adaptive threshold using 11 nearest neighbour pixels
  	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

  	# Invert colours, so gridlines have non-zero pixel values.
  	# Necessary to dilate the image, otherwise will look like erosion instead.
  	proc = cv2.bitwise_not(proc, proc)

  	if not skip_dilate:
  		# Dilate the image to increase the size of the grid lines.
  		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
  		kernel=kernel.astype(np.uint8)
  		proc = cv2.dilate(proc, kernel)

  	return proc


  def find_corners(img):
  	"""Finds the 4 extreme corners of the largest contour in the image."""
  	contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
  	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
  	polygon = contours[0]  # Largest image

  	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
  	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

  	# Bottom-right point has the largest (x + y) value
  	# Top-left has point smallest (x + y) value
  	# Bottom-left point has smallest (x - y) value
  	# Top-right point has largest (x - y) value
  	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

  	# Return an array of all 4 points using the indices
  	# Each point is in its own array of one coordinate
  	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


  def distance_between(p1, p2):
  	"""Returns the scalar distance between two points"""
  	a = p2[0] - p1[0]
  	b = p2[1] - p1[1]
  	return np.sqrt((a ** 2) + (b ** 2))


  def crop_and_warp(img, crop_rect):
  	"""Crops and warps a rectangular section from an image into a square of similar size."""

  	# Rectangle described by top left, top right, bottom right and bottom left points
  	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

  	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
  	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

  	# Get the longest side in the rectangle
  	side = max([
  		distance_between(bottom_right, top_right),
  		distance_between(top_left, bottom_left),
  		distance_between(bottom_right, bottom_left),
  		distance_between(top_left, top_right)
  	])

  	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
  	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

  	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
  	m = cv2.getPerspectiveTransform(src, dst)

  	# Performs the transformation on the original image
  	return cv2.warpPerspective(img, m, (int(side), int(side)))


  def infer_grid(img):
  	"""Infers 81 cell grid from a square image."""
  	squares = []
  	side = img.shape[:1]
  	side = side[0] / 9

  	# Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
  	for j in range(9):
  		for i in range(9):
  			p1 = (i * side, j * side)  # Top left corner of a bounding box
  			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
  			squares.append((p1, p2))
  	return squares


  def cut_from_rect(img, rect):
  	"""Cuts a rectangle from an image using the top left and bottom right points."""
  	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


  def scale_and_centre(img, size, margin=0, background=0):
  	"""Scales and centres an image onto a new background square."""
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


  def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
  	"""
  	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
  	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
  	"""
  	img = inp_img.copy()  # Copy the image, leaving the original untouched
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


  def extract_digit(img, rect, size):
  	"""Extracts a digit (if one exists) from a Sudoku square."""

  	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

  	# Use fill feature finding to get the largest feature in middle of the box
  	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
  	h, w = digit.shape[:2]
  	margin = int(np.mean([h, w]) / 2.5)
  	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
  	digit = cut_from_rect(digit, bbox)

  	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
  	w = bbox[1][0] - bbox[0][0]
  	h = bbox[1][1] - bbox[0][1]

  	# Ignore any small bounding boxes
  	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
  		return scale_and_centre(digit, size, 4)
  	else:
  		return np.zeros((size, size), np.uint8)


  def get_digits(img, squares, size):
  	"""Extracts digits from their cells and builds an array"""
  	digits = []
  	img = pre_process_image(img.copy(), skip_dilate=True)
  	for square in squares:
  		digits.append(extract_digit(img, square, size))
  	return digits


  def parse_grid(path):
  	original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  	processed = pre_process_image(original)
  	corners = find_corners(processed)
  	cropped = crop_and_warp(original, corners)
  	squares = infer_grid(cropped)
  	digit_imgs = get_digits(cropped, squares, 28)
  	#show_digits(digit_imgs)
  	digits = []
  	for img in digit_imgs:
  		GPIO.output(6, GPIO.HIGH)
  		text = pytesseract.image_to_string(cv2.bitwise_not(img), config="--psm 10 ")
  		print(text)
  		GPIO.output(6, GPIO.LOW)
  		valid = True
  		if (text.isdigit()):
  			if not (int(text) in range(1,10)):
  				valid = False
  		elif (text):
  			valid = False
  		else:
  			text = 0

  		if(valid):
  			digits.append(int(text))
  		else:
  			print("Could not recognize board. Please try again. ")
  			return []
  	return digits

  #-----------------------------------------------------------------------
  #SOLVING CODE: Functions and globals
  #-----------------------------------------------------------------------
  #import clib and set return type
  so_file = "/home/pi/Project/solver.so"
  solver = CDLL(so_file)
  solve = solver.solve
  solve.restype = POINTER(c_int * 81)
  #-----------------------------------------------------------------------
  #WRITING CODE: Functions and globals
  #-----------------------------------------------------------------------
  #constants
  min_val = 0b000000000000
  max_val = 0b111111111111
  max_x = 4045
  max_y = 3925
  a_mask = 0b00110000
  b_mask = 0b10110000

  #global variables
  prev_x = 0b000000000000
  prev_y = 0b000000000000
  curr_x = 0b000000000000
  curr_y = 0b000000000000

  #SPI writing functions
  #NOTE: Y is the ROW direction for the array, X is the COLUMN direction
  def write_x(val):
  	msg = [ a_mask | (val >> 8), val & 0xff]
  	spi.xfer2(msg)

  def write_y(val):
  	msg = [ b_mask | (val >> 8), val & 0xff]
  	spi.xfer2(msg)

  def seek_y(y):
  	global curr_y
  	if (y > 8):
  		curr_y = 0b111111111111
  	elif (y < 0):
  		curr_y = 0
  	else:
  		curr_y = (int(max_y/9))*y
  	write_y(curr_y)

  def seek_x(x):
  	global curr_x
  	if (x > 8):
  		curr_x = 0b111111111111
  	elif (x < 0):
  		curr_x = 0
  	else:
  		curr_x = (int(max_x/9))*x
  	write_x(curr_x)

  def de_grid():
  	global curr_x
  	global curr_y
  	curr_x = curr_x + int(max_x/6/9)
  	curr_y = curr_y + int(max_y/4/9)
  	write_x(curr_x)
  	write_y(curr_y)
  	time.sleep(.25)

  def cross_right_y():
  	global curr_y
  	curr_y = curr_y + int(2*max_y/4/9)
  	write_y(curr_y)
  	time.sleep(.5)

  def cross_left_y():
  	global curr_y
  	curr_y = curr_y - int(2*max_y/4/9)
  	write_y(curr_y)
  	time.sleep(.5)


  def half_box_down_x():
  	global curr_x
  	curr_x = curr_x + int(2*max_x/6/9)
  	write_x(curr_x)
  	time.sleep(.5)

  def half_box_up_x():
  	global curr_x
  	curr_x = curr_x - int(2*max_x/6/9)
  	write_x(curr_x)
  	time.sleep(.5)


  def full_box_down_x():
  	global curr_x
  	curr_x = curr_x + int(4*max_x/6/9)
  	write_x(curr_x)
  	time.sleep(.5)

  def full_box_up_x():
  	global curr_x
  	curr_x = curr_x - int(4*max_x/6/9)
  	write_x(curr_x)
  	time.sleep(.5)

  def reset_x():
  	global curr_x
  	write_x(min_val)
  	time.sleep(1)
  	write_x(curr_x)
  	time.sleep(1)

  def re_grid():
  	global curr_x
  	global curr_y
  	curr_x = curr_x - int(max_x/6/9)
  	curr_y = curr_y - int(max_y/4/9)
  	if curr_x < 0:
  		curr_x = 0
  	if curr_y < 0:
  		curr_y = 0
  	write_x(curr_x)
  	write_y(curr_y)
  	time.sleep(.25)

  def draw(n):
  	global curr_y
  	de_grid()
  	if (n == 1):
  		curr_y = curr_y + 140
  		write_y(curr_y)
  		time.sleep(.5)
  		half_box_down_x()
  		half_box_down_x()
  		half_box_up_x()
  		half_box_up_x()
  		curr_y = curr_y - 140
  		write_y(curr_y)
  		time.sleep(.5)
  	elif (n==2):
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		half_box_down_x()
  		cross_right_y()
  		cross_left_y()
  		half_box_up_x()
  		cross_right_y()
  		half_box_up_x()
  		cross_left_y()
  	elif (n==3):
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		full_box_up_x()
  		cross_left_y()
  	elif (n==4):
  		half_box_down_x()
  		cross_right_y()
  		half_box_up_x()
  		half_box_down_x()
  		half_box_down_x()
  		half_box_up_x()
  		cross_left_y()
  		half_box_up_x()
  	elif (n==5):
  		cross_right_y()
  		cross_left_y()
  		half_box_down_x()
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		half_box_up_x()
  		cross_left_y()
  		half_box_up_x()
  	elif (n==6):
  		cross_right_y()
  		cross_left_y()
  		half_box_down_x()
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		half_box_up_x()
  		half_box_down_x()
  		full_box_up_x()
  	elif (n==7):
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		half_box_down_x()
  		full_box_up_x()
  		cross_left_y()
  	elif (n==8):
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		half_box_down_x()
  		cross_left_y()
  		half_box_up_x()
  		half_box_down_x()
  		half_box_up_x()
  		half_box_up_x()
  		half_box_down_x()
  		half_box_up_x()
  	elif (n==9):
  		cross_right_y()
  		half_box_down_x()
  		half_box_down_x()
  		cross_left_y()
  		cross_right_y()
  		half_box_up_x()
  		cross_left_y()
  		half_box_up_x()
  		half_box_down_x()
  		half_box_up_x()
  	else:
  		print("cannot write digit")
  	re_grid()

  #ISR for Calibration
  def calibrate_func(channel):
  	#Turn on LEDS
  	global calibrating
  	calibrating = True
  	#wait for digit to finish writing
  	time.sleep(2)
  	GPIO.output(6, GPIO.HIGH)
  	GPIO.output(19, GPIO.HIGH)
  	#Go To Extreemes
  	write_x(min_val)
  	print(0)
  	time.sleep(1)
  	write_x(max_x)
  	print(max_val)
  	time.sleep(1)
  	write_x(min_val)
  	print(0)
  	time.sleep(1)

  	write_y(min_val)
  	print(0)
  	time.sleep(1)
  	write_y(max_y)
  	print(max_val)
  	time.sleep(1)
  	write_y(min_val)
  	print(min_val)
  	time.sleep(1)
  	draw(9)
  	time.sleep(1)
  	#Increment up
  	i = 1
  	while i < 9:
  		print(i)
  		seek_x(i)
  		reset_x()
  		time.sleep(1)
  		seek_y(i)
  		time.sleep(1)
  		draw(i)
  		time.sleep(.5)
  		i = i + 1
  	#Increment down
  	i = 8
  	while i > -1:
  		seek_x(i)
  		reset_x()
  		print(i)
  		time.sleep(1)
  		seek_y(i)
  		time.sleep(1)
  		i = i - 1
  	write_y(min_val)
  	write_x(min_val)
  	time.sleep(1)
  	#Turn off LEDS
  	GPIO.output(6, GPIO.LOW)
  	GPIO.output(19, GPIO.LOW)
  	calibrating = False
  #ISR for quitting mid-solve
  def redo_func(channel):
  	global redo
  	redo = True
  #create event for calibrate
  GPIO.add_event_detect(5, GPIO.FALLING, callback=calibrate_func,bouncetime = 300)
  GPIO.add_event_detect(16, GPIO.FALLING, callback=redo_func,bouncetime = 300)
  calibrating = False
  run_forever = True
  redo = False
  #main loop
  while(run_forever):
  	#poll for start button
  	GPIO.output(13, GPIO.HIGH)
  	redo = False
  	if not GPIO.input(26):
  		GPIO.output(6, GPIO.LOW)
  		GPIO.output(13, GPIO.LOW)
  		GPIO.output(19, GPIO.LOW)
  		#---------------------------------------------------------------
  		#READ
  		#---------------------------------------------------------------
  		GPIO.output(6, GPIO.HIGH)
  		#camera.start_preview()
  		time.sleep(10)
  		camera.capture('img.jpg')
  		#camera.stop_preview()
  		board = parse_grid('img.jpg')
  		if len(board) == 0:
  			GPIO.output(6, GPIO.HIGH)
  			GPIO.output(13, GPIO.HIGH)
  			GPIO.output(19, GPIO.HIGH)
  			continue
  		print(board)
  		#should end with 1d 81 length list of python ints
  		#board = [0 for i in range(81)]
  		GPIO.output(6, GPIO.LOW)
  		#---------------------------------------------------------------
  		#SOLVE
  		#---------------------------------------------------------------
  		GPIO.output(13, GPIO.HIGH)
  		#create 2d boolean mask for inital array
  		mask = [[False for i in range(9)] for j in range(9)]
  		c = 0
  		while (c < 81):
  			if board[c] == 0:
  				print(c)
  				mask[int(c/9)][c%9] = True
  			c = c + 1
  		print("going to c")
  		#gen c-readable array
  		arr = (c_int * 81)(*board)
  		#solve and save returned array
  		arr2 = (solver.solve(arr))
  		#create 2d python array out of solved array
  		write_board = [[0 for i in range(9)] for j in range(9)]
  		j = 0
  		for i in arr2.contents:
  			write_board[int(j/9)][j%9] = i
  			j = j + 1
  		print(mask)
  		print(write_board)
  		GPIO.output(13, GPIO.LOW)
  		#---------------------------------------------------------------
  		#WRITE
  		#---------------------------------------------------------------
  		GPIO.output(19, GPIO.HIGH)
  		#reset x and y and prime pen
  		#Go To Extreemes
  		write_x(min_val)
  		print(0)
  		time.sleep(1)
  		write_x(max_x)
  		print(max_val)
  		time.sleep(1)
  		write_x(min_val)
  		print(0)
  		time.sleep(1)

  		write_y(min_val)
  		print(0)
  		time.sleep(1)
  		write_y(max_y)
  		print(max_val)
  		time.sleep(1)
  		write_y(min_val)
  		print(min_val)
  		time.sleep(1)
  		#for each row
  		a = 0
  		b = 0
  		while (a < 9):
  			#even rows go left -> right
  			if a%2 == 0:
  				b = 0
  				while(b < 9):
  					while calibrating:
  						pass
  					if redo:
  						continue
  					if mask[a][b]:
  						print("writing "+str(write_board[a][b])+" to "+str(a)+", "+str(b))
  						seek_x(a)
  						time.sleep(.5)
  						seek_y(b)
  						time.sleep(.5)
  						draw(write_board[a][b])
  					b = b + 1
  			#odd rows go right -> left
  			else:
  				b = 8
  				while(b > -1):
  					while calibrating:
  						pass
  					if redo:
  						continue
  					if mask[a][b]:
  						print("writing "+str(write_board[a][b])+" to "+str(a)+", "+str(b))
  						seek_x(a)
  						time.sleep(.5)
  						seek_y(b)
  						time.sleep(.5)
  						draw(write_board[a][b])
  					b = b - 1
  			#reset x after each row
  			#reset_x()
  			a = a + 1
  		seek_x(0)
  		time.sleep(1)
  		seek_y(0)
  		time.sleep(1)
  		GPIO.output(19, GPIO.LOW)
  		#END OF BUTTON CODE

  	time.sleep(.1)
  	GPIO.output(13, GPIO.HIGH)
  	time.sleep(.1)

  GPIO.cleanup()
  spi.close()


  //includes aren't displayed corectly because of html angle bracket things
  #include 
  #include 
  #include 
  #include 
  #include 
  #include 
  #define BILLION 1000000000L

  #define Q_SIZE 81

  bool f3 = false;
  pthread_mutex_t q_mutex = PTHREAD_MUTEX_INITIALIZER;
  int p_found = -1;

  struct job
  {
  	int v1;
  	int v2;
  };

  struct queue
  {
  	struct job **buffer;
  	int start;
  	int end;
  	int c;
  	int all_produced;
  };

  typedef struct
  {
  	int pid, x1, x2, y1, y2;
  	int **board;
  	struct queue *q;
  } GM;

  struct job *get_work(struct queue *q)
  {
  	pthread_mutex_lock(&q_mutex);
  	if (!q->c)
  	{
  		pthread_mutex_unlock(&q_mutex);
  		return NULL;
  	}
  	struct job *temp = q->buffer[q->start];
  	q->start = (q->start + 1) % Q_SIZE;
  	q->c--;
  	pthread_mutex_unlock(&q_mutex);
  	return temp;
  }

  void add_work(struct queue *q, struct job *toAdd)
  {
  	while (q->c == Q_SIZE)
  		;
  	pthread_mutex_lock(&q_mutex);
  	q->end = (q->end + 1) % Q_SIZE;
  	q->buffer[q->end] = toAdd;
  	q->c++;
  	pthread_mutex_unlock(&q_mutex);
  }

  //helper function, verifies the board still complies with the invariant
  bool verify_placement(int i, int j, int n, int **board)
  {
  	//no same number in the row
  	for (int x = 0; x < 9; x++)
  	{
  		if (x != i && board[x][j] == n)
  		{
  			return false;
  		}
  	}
  	//no same number in the column
  	for (int y = 0; y < 9; y++)
  	{
  		if (y != j && board[i][y] == n)
  		{
  			return false;
  		}
  	}
  	//no same number in the square
  	if (i < 3)
  	{
  		if (j < 3)
  		{
  			//top left
  			for (int x = 0; x < 3; x++)
  			{
  				for (int y = 0; y < 3; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//top middle
  		else if (j < 6)
  		{
  			for (int x = 0; x < 3; x++)
  			{
  				for (int y = 3; y < 6; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//top right
  		else
  		{
  			for (int x = 0; x < 3; x++)
  			{
  				for (int y = 6; y < 9; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  	}
  	else if (i < 6)
  	{
  		if (j < 3)
  		{
  			//middle left
  			for (int x = 3; x < 6; x++)
  			{
  				for (int y = 0; y < 3; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//true middle
  		else if (j < 6)
  		{
  			for (int x = 3; x < 6; x++)
  			{
  				for (int y = 3; y < 6; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//middle right
  		else
  		{
  			for (int x = 3; x < 6; x++)
  			{
  				for (int y = 6; y < 9; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  	}
  	else
  	{
  		if (j < 3)
  		{
  			//bottom left
  			for (int x = 6; x < 9; x++)
  			{
  				for (int y = 0; y < 3; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//bottom middle
  		else if (j < 6)
  		{
  			for (int x = 6; x < 9; x++)
  			{
  				for (int y = 3; y < 6; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  		//bottom right
  		else
  		{
  			for (int x = 6; x < 6; x++)
  			{
  				for (int y = 6; y < 9; y++)
  				{
  					if (x != i && y != j && board[x][y] == n)
  					{
  						return false;
  					}
  				}
  			}
  		}
  	}
  	return true;
  }

  //maps indecies of a 2d 9x9 array to a 1d 81 array in row-major order
  int arrayIndexMap(int row, int col)
  {
  	return row * 9 + col;
  }

  //function to print the solved board to the console
  void printSolution(int **board)
  {
  	printf("============SOLUTION============\n");
  	for (int i = 0; i < 9; i++)
  	{
  		if (i == 3 || i == 6)
  		{
  			printf(" ---------+-----------+--------- \n");
  		}
  		for (int j = 0; j < 9; j++)
  		{
  			if (j == 3 || j == 6)
  			{
  				printf(" | ");
  			}
  			printf(" %d ", board[i][j]);
  		}
  		printf("\n");
  	}
  }

  void copy_board(int **b1, int **b2)
  {
  	for (int i = 0; i < 9; i++)
  	{
  		for (int j = 0; j < 9; j++)
  		{
  			b2[i][j] = b1[i][j];
  		}
  	}
  }

  //function to generate boards
  void generate_board(int **board)
  {
  	//for now, hard-coded
  	/* Start:
  		 * 2 . 7 | 9 . . | . . .
  		 * 3 4 . | 5 7 2 | 1 . .
  		 * 5 . . | . . 1 | 2 8 .
  		 * ------+-------+------
  		 * 7 . 4 | 6 . . | . . 3
  		 * . . . | 7 5 . | . 2 8
  		 * 6 . . | 1 9 . | 4 . .
  		 * ------+-------+------
  		 * . 2 . | . 4 9 | 7 . 1
  		 * . 9 5 | 8 . . | 3 4 2
  		 * . 7 3 | . 1 5 | 8 . .
  		 *
  		 * Solution:
  		 * 2 1 7 | 9 8 6 | 5 3 4
  		 * 3 4 8 | 5 7 2 | 1 9 6
  		 * 5 6 9 | 4 3 1 | 2 8 7
  		 * ------+-------+------
  		 * 7 5 4 | 6 2 8 | 9 1 3
  		 * 9 3 1 | 7 5 4 | 6 2 8
  		 * 6 8 2 | 1 9 3 | 4 7 5
  		 * ------+-------+------
  		 * 8 2 6 | 3 4 9 | 7 5 1
  		 * 1 9 5 | 8 6 7 | 3 4 2
  		 * 4 7 3 | 2 1 5 | 8 6 9
  		 */
  	board[0][0] = 2;
  	board[0][2] = 7;
  	board[0][3] = 9;
  	board[1][0] = 3;
  	board[1][1] = 4;
  	board[1][3] = 5;
  	board[1][4] = 7;
  	board[1][5] = 2;
  	board[1][6] = 1;
  	board[2][0] = 5;
  	board[2][5] = 1;
  	board[2][6] = 2;
  	board[2][7] = 8;
  	board[3][0] = 7;
  	board[3][2] = 4;
  	board[3][3] = 6;
  	board[3][8] = 3;
  	board[4][3] = 7;
  	board[4][4] = 5;
  	board[4][7] = 2;
  	board[4][8] = 8;
  	board[5][0] = 6;
  	board[5][3] = 1;
  	board[5][4] = 9;
  	board[5][6] = 4;
  	board[6][1] = 2;
  	board[6][4] = 4;
  	board[6][5] = 9;
  	board[6][6] = 7;
  	board[6][8] = 1;
  	board[7][1] = 9;
  	board[7][2] = 5;
  	board[7][3] = 8;
  	board[7][6] = 3;
  	board[7][7] = 4;
  	board[7][8] = 2;
  	board[8][1] = 7;
  	board[8][2] = 3;
  	board[8][4] = 1;
  	board[8][5] = 5;
  	board[8][6] = 8;
  }

  void generate_board_hard(int **board)
  {
  	//for now, hard-coded
  	/* Start:
  		 * . . . | . . . | . . .
  		 * . . . | . . 3 | . 8 5
  		 * . . 1 | . 2 . | . . .
  		 * ------+-------+------
  		 * . . . | 5 . 7 | . . .
  		 * . . 4 | . . . | 1 . .
  		 * . 9 . | . . . | . . .
  		 * ------+-------+------
  		 * 5 . . | . . . | . 7 3
  		 * . . 2 | . 1 . | . . .
  		 * . . . | . 4 . | . . 9
  		 *
  		 * ============SOLUTION============
  		 * 9  6  3  |  8  5  4  |  7  2  1
  		 * 4  2  7  |  9  1  3  |  6  8  5
  		 * 8  5  1  |  7  2  6  |  3  9  4
  		 * ---------+-----------+---------
  		 * 1  8  6  |  5  3  7  |  9  4  2
  		 * 7  3  4  |  2  9  8  |  1  5  6
  		 * 2  9  5  |  4  6  1  |  8  3  7
  		 * ---------+-----------+---------
  		 * 5  1  9  |  6  8  2  |  4  7  3
  		 * 3  4  2  |  1  7  9  |  5  6  8
  		 * 6  7  8  |  3  4  5  |  2  1  9
  		 */
  	board[1][5] = 3;
  	board[1][7] = 8;
  	board[1][8] = 5;

  	board[2][2] = 1;
  	board[2][4] = 2;

  	board[3][3] = 5;
  	board[3][5] = 7;

  	board[4][2] = 4;
  	board[4][6] = 1;

  	board[5][1] = 9;

  	board[6][0] = 5;
  	board[6][7] = 7;
  	board[6][8] = 3;

  	board[7][2] = 2;
  	board[7][3] = 1;

  	board[8][4] = 4;
  	board[8][8] = 9;
  }

  //recursevly solve the board
  int solve_board(int **board, int i, int j)
  {
  	//base case
  	if (i == 9)
  	{
  		//end of board, done
  		return 1;
  	}
  	if (f3)
  	{
  		//someone else is done
  		return -1;
  	}
  	int d = board[i][j];
  	//if the digit is prefilled
  	if (d != 0)
  	{
  		//printf("preallocated digit\n");
  		//check invariant, if it's been violated return -1 immediatly; something else needs to change

  		if (!verify_placement(i, j, d, board))
  		{
  			return -1;
  		}
  		//otherwise recurse
  		else
  		{
  			//if end of row
  			if (j == 8)
  			{

  				//printf("next row\n");
  				//still rows to go through
  				return solve_board(board, i + 1, 0);
  			}
  			else
  			{
  				//printf("next col\n");
  				//next column over
  				return solve_board(board, i, j + 1);
  			}
  		}
  	}
  	else
  	{
  		//if not, while you still have digits to try
  		int k = 1;
  		int m = 0;
  		while (k < 10 && m < 1)
  		{
  			//try one, if it works, recurse
  			if (verify_placement(i, j, k, board))
  			{
  				//apply working digit
  				board[i][j] = k;
  				//printf("board[%d][%d]<-%d\n", i, j, board[i][j]);
  				//recurse
  				//if end of column
  				if (j == 8)
  				{
  					//printf("next row\n");
  					//go to next row
  					m = solve_board(board, i + 1, 0);
  				}
  				else
  				{
  					//printf("next col\n");
  					//still cols to go through
  					m = solve_board(board, i, j + 1);
  				}
  			}
  			//increment k and try again
  			k++;
  		}
  		//bubble up the solution if it's been found
  		if (m == 1)
  		{
  			return 1;
  		}
  		/*you've tried all the digits
  		return your digit to zero and try again, no solution, return -1 */
  		board[i][j] = 0;
  		return -1;
  	}
  }

  //thread function, recieve jobs and try solving
  void *do_solve(void *varg)
  {
  	GM *arg = varg;
  	int pid, v1, v2, x1, x2, y1, y2;
  	int **start_board;
  	struct queue *q;

  	pid = arg->pid;
  	start_board = arg->board;
  	q = arg->q;
  	x1 = arg->x1;
  	x2 = arg->x2;
  	y1 = arg->y1;
  	y2 = arg->y2;

  	struct timespec start, end;
  	double time;
  	clock_gettime(CLOCK_MONOTONIC, &start);
  	int jobs_processed = 0;

  	//make your own board
  	int **local_board = (int **)malloc(9 * sizeof(int *));
  	for (int a = 0; a < 9; a++)
  	{
  		local_board[a] = (int *)malloc(9 * sizeof(int));
  	}

  	//p0 spawns the jobs to be placed
  	if (pid == 0)
  	{
  		for (int i = 1; i < 10; i++)
  		{
  			for (int k = 1; k < 10; k++)
  			{

  				struct job *job = malloc(sizeof(struct job));
  				job->v1 = i;
  				job->v2 = k;
  				add_work(q, job);
  				//printf("Produced: %d, %d\n", i, j);
  			}
  		}
  		//printf("Left in queue: %d\n", q->c);
  		q->all_produced = 1;
  	}
  	//all threads take jobs and evaluate
  	//printf("Starting Jobs\n");
  	copy_board(start_board, local_board);
  	int found = -1;
  	while (!f3 && (!q->all_produced || q->c))
  	{
  		//new job, reset your board
  		//printf("Copying Board\n");

  		//printf("Consumer %d reading", pid);
  		struct job *j;
  		j = get_work(q);
  		if (j == NULL)
  		{
  			continue;
  		}
  		v1 = j->v1;
  		v2 = j->v2;
  		//printf("Atempting first two placement\n");
  		//try the first two, if they violate the invariant, get a new job
  		if (verify_placement(x1, y1, v1, local_board))
  		{
  			local_board[x1][y1] = v1;
  			if (verify_placement(x2, y2, v2, local_board))
  			{
  				local_board[x2][y2] = v2;
  				//if they're fine, solve the rest of the board
  				//printf("Continuing Solver\n");
  				found = solve_board(local_board, x2, y2);
  				if (found > 0)
  				{
  					f3 = true;
  					p_found = pid;
  					break;
  				}
  				else
  				{
  					//if the first was fine, but second wasn't, reset first
  					local_board[x1][y1] = 0;
  				}
  			}
  		}
  		jobs_processed++;
  	}
  	printf("Thread %d got %d jobs\n", pid, jobs_processed);
  	clock_gettime(CLOCK_MONOTONIC, &end);
  	time = BILLION * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  	time = time / BILLION;
  	printf("Thread %d elapsed: %lf seconds\n", pid, time);

  	pthread_exit((void *)local_board);
  }

  int ** oned_to_twod(int * board) {
  	int ** rhet = (int **)malloc(9 * sizeof(int *));
  	for (int a = 0; a < 9; a++)
  	{
  		rhet[a] = (int *)malloc(9 * sizeof(int));
  	}
  	for (int i = 0; i < 81; i++) {
  		rhet[i/9][i%9] = board[i];
  	}
  	return rhet;
  }

  int * twod_to_oned(int ** board) {
  	//printf("reconverting\n");
  	int * rhet = (int *)malloc(81 * sizeof(int));
  	for (int i = 0; i < 81; i++) {
  		 rhet[i] = board[i/9][i%9];
  	}
  	return rhet;
  }

  int* solve(int * board) {
  	//init
  	int a, b, **start_board;
  	int x1, x2, y1, y2;
  	bool f1 = false, f2 = false;
  	start_board = oned_to_twod(board);
  	printSolution(start_board);
  	//find the first two open spots on the board
  	//printf("Finding First Two Slots\n");
  	for (int i = 0; i < 9; i++)
  	{
  		for (int j = 0; j < 9; j++)
  		{
  			//printf("checking %d, %d\n", i, j);
  			if (f1 && f2)
  			{
  				break;
  			}
  			if (start_board[i][j] == 0)
  			{
  				if (f1)
  				{
  					x2 = i;
  					y2 = j;
  					f2 = true;
  				}
  				else
  				{
  					x1 = i;
  					y1 = j;
  					f1 = true;
  				}
  			}
  		}
  	}
  	if (!f1)
  	{
  		printf("Already Solved\n");
  	}
  	else if (!f2)
  	{
  		printf("Only one missing\n");
  		b = solve_board(start_board, x1, y1);
  	}
  	else
  	{
  		//printf("Starting Parallel\n");
  		//make the queue
  		struct queue *q = malloc(sizeof(struct queue));
  		q->buffer = malloc(sizeof(struct job *) * Q_SIZE);
  		q->start = 0;
  		q->end = Q_SIZE - 1;
  		q->c = 0;
  		q->all_produced = 0;

  		pthread_t *threads = malloc(4 * sizeof(threads));

  		//printf("init return structure\n");
  		void **local_boards = malloc(4 * sizeof(int **));
  		for (int i = 0; i < 4; i++)
  		{
  			local_boards[i] = NULL;
  		}
  		//printf("Spawning Threads\n");
  		for (int i = 0; i < 4; i++)
  		{
  			GM *arg = malloc(sizeof(*arg));
  			arg->q = q;
  			arg-> x1 = x1;
  			arg-> x2 = x2;
  			arg-> y1 = y1;
  			arg-> y2 = y2;
  			arg->pid = i;
  			arg->board = start_board;
  			pthread_create(&threads[i], NULL, do_solve, arg);
  		}

  		//printf("Recieving Threads\n");
  		for (int i = 0; i < 4; i++)
  		{
  			pthread_join(threads[i], &local_boards[i]);
  		}
  		start_board = (int **)local_boards[p_found];
  	}
  	//prints
  	if (b == -1)
  	{
  		printf("Exited in error\n");
  	}
  	printSolution(start_board);
  	int * rhet = twod_to_oned(start_board);
  	return rhet;
  }


  void main(int **board)
  {
  	//printf("Starting Main\n");
  	//timing setup
  	struct timespec initstart, initend;
  	struct timespec compstart, compend;
  	double inittime, comptime, totaltime;

  	/*-------------------------------------------------------
  		  Computation
  		 -------------------------------------------------------*/

  	//init
  	int a, b, **start_board;
  	int x1, x2, y1, y2;
  	bool f1 = false, f2 = false;
  	//for now, the board will be hard-coded
  	//printf("Making Start Board\n");
  	start_board = (int **)malloc(9 * sizeof(int *));
  	for (a = 0; a < 9; a++)
  	{
  		start_board[a] = (int *)malloc(9 * sizeof(int));
  	}
  	generate_board(start_board);
  	clock_gettime(CLOCK_MONOTONIC, &initstart);
  	//find the first two open spots on the board
  	//printf("Finding First Two Slots\n");
  	for (int i = 0; i < 9; i++)
  	{
  		for (int j = 0; j < 9; j++)
  		{
  			if (f1 && f2)
  			{
  				break;
  			}
  			if (start_board[i][j] == 0)
  			{
  				if (f1)
  				{
  					x2 = i;
  					y2 = j;
  					f2 = true;
  				}
  				else
  				{
  					x1 = i;
  					y1 = j;
  					f1 = true;
  				}
  			}
  		}
  	}
  	if (!f1)
  	{
  		printf("Already Solved\n");
  	}
  	else if (!f2)
  	{
  		printf("Only one missing\n");
  		b = solve_board(start_board, x1, y1);
  	}
  	else
  	{
  		//printf("Starting Parallel\n");
  		//make the queue
  		struct queue *q = malloc(sizeof(struct queue));
  		q->buffer = malloc(sizeof(struct job *) * Q_SIZE);
  		q->start = 0;
  		q->end = Q_SIZE - 1;
  		q->c = 0;
  		q->all_produced = 0;

  		pthread_t *threads = malloc(4 * sizeof(threads));

  		//printf("init return structure\n");
  		void **local_boards = malloc(4 * sizeof(int **));
  		for (int i = 0; i < 4; i++)
  		{
  			local_boards[i] = NULL;
  		}
  		clock_gettime(CLOCK_MONOTONIC, &initend);
  		clock_gettime(CLOCK_MONOTONIC, &compstart);
  		//printf("Spawning Threads\n");
  		for (int i = 0; i < 4; i++)
  		{
  			GM *arg = malloc(sizeof(*arg));
  			arg->q = q;
  			arg-> x1 = x1;
  			arg-> x2 = x2;
  			arg-> y1 = y1;
  			arg-> y2 = y2;
  			arg->pid = i;
  			arg->board = start_board;
  			pthread_create(&threads[i], NULL, do_solve, arg);
  		}

  		//printf("Recieving Threads\n");
  		for (int i = 0; i < 4; i++)
  		{
  			pthread_join(threads[i], &local_boards[i]);
  		}
  		start_board = (int **)local_boards[p_found];
  	}

  	//timing end
  	clock_gettime(CLOCK_MONOTONIC, &compend);
  	inittime = BILLION * (initend.tv_sec - initstart.tv_sec) + (initend.tv_nsec - initstart.tv_nsec);
  	inittime = inittime / BILLION;

  	comptime = BILLION * (compend.tv_sec - compstart.tv_sec) + (compend.tv_nsec - compstart.tv_nsec);
  	comptime = comptime / BILLION;
  	totaltime = inittime + comptime;

  	totaltime = comptime + inittime;

  	//prints
  	if (b == -1)
  	{
  		printf("Exited in error\n");
  	}
  	printf("Init time: %lf seconds\n", inittime);
  	printf("Comp time: %lf seconds\n", comptime);
  	printf("Total time: %lf seconds\n", totaltime);
  	printSolution(start_board);
  }

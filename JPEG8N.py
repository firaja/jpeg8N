from __future__ import division
import numpy as np
from scipy import misc
from scipy.fftpack import dct
import cv2


class JPEG8N:

	_Q = np.array([[16, 11, 10, 16, 24,  40,  51,  61 ], 
				   [12, 12, 14, 19, 26,  58,  60,  55 ],
				   [14, 13, 16, 24, 40,  57,  69,  56 ],
				   [14, 17, 22, 29, 51,  87,  80,  62 ],
				   [18, 22, 37, 56, 68,  109, 103, 77 ],
				   [24, 35, 55, 64, 81,  104, 113, 92 ],
				   [49, 64, 78, 87, 103, 121, 120, 101],
				   [72, 92, 95, 98, 112, 100, 103, 99 ]]).astype(float)
	# matrice Q1
	_Q1 = None
	# grandezza dei blocchi
	block_size = 8
	# array con i pixel dell'immagine originale
	original_image = None
	# array diviso in blocchi di array
	image_block = None
	# larghezza dell'immagine
	width = 0
	# altezza dell'immagine
	height = 0
	# parametro N
	N = 1
	# qualita'
	quality = 90

	def __init__(self, filename=None, N=1, mult=8):
		"""Costruttore. E' possibile inizializzare una immagine sia da 
		filename che impostando i blocchi manualmente"""
		self.N = N
		self.block_size = 8*N
		if filename is not None:
			self.original_image = self.__load_file(filename)
			print "image loaded"
			self.__resize()
			print "image resized to " + str(self.width) +"x"+ str(self.height) 
			self.__enblock()
			print "image divided by " +str(self.block_size) +"x" +str(self.block_size) + " blocks"


	def __load_file(self, filename):
		"""Metodo per il caricamento dell'immagine e conversione
		in array Python"""
		#img = misc.imread(filename, flatten=True).tolist()
		img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE).tolist()
		return img

	def __resize(self):
		"""Metodo di riadattamento delle misure dell'immagine in Metodo
		che sia di altezza e larghezza multipla di 8N"""
		img = self.original_image
		old_w, w = len(img[0]), len(img[0])
		old_h, h = len(img), len(img)
		while w % (8 * self.N) != 0:
			w += 1
		while h % (8 * self.N) != 0:
			h += 1
		last_px = img[-1][-1]
		last_row = img[-1]
		for i in range(old_h):
			img[i] = img[i] + [img[i][-1]] * (w-old_w)
		last_row = last_row + [last_px] * (w-old_w)
		img = img + [last_row] *(h-old_h)
		self.original_image = np.array(img)
		self.width = w
		self.height = h
		return True

	def __enblock(self):
		"""Metodo che divide l'immagine in array di blocchi 8N x 8N"""
		array = np.array([[[[0.] * self.block_size] * self.block_size ] * (self.width // self.block_size)] * (self.height // self.block_size))
		for y in xrange(self.height // self.block_size):
			for x in xrange(self.width // self.block_size):
				for j in xrange(self.block_size):
					for i in xrange(self.block_size):
						array[y][x][j][i] = self.original_image[self.block_size*y + j][self.block_size*x + i]
		self.image_block = array

	def __dct2D(self, x, inverse=False):
		"""Metodo che effettua la DCT-II o la DCT-III di Scipy"""
		t = 2 if not inverse else 3
		#temp = dct(x, type=t, norm='ortho').transpose()
		return dct(dct(x, norm='ortho', type=t, axis=0), norm='ortho', axis=1, type=t)#.transpose()

	def __get_qf(self):
		"""Standardizzazione del paramentro qualita'"""
		if self.quality < 1:
			self.quality = 1
		if self.quality > 100:
			self.quality = 100
		if self.quality < 50:
			return 5000/self.quality/100.0
		else:
			return (200-self.quality*2)/100.0

	def __force_baseline(q):
		array = np.array([[0.] * (self.N*8)] * (self.N*8))
		for y in range(len(q)):
			for x in range(len(q[0])):
				array[y][x] = q[y][x] if q[y][x] <= 255 else 255
		return array

	def __quantitize(self, block, inverse=False):
		"""Quantizzazione e sua inversa"""
		array = np.array([[0.] * self.block_size] * self.block_size)
		if not inverse:
			"""for y in range(self.block_size):
				for x in range(self.block_size):
					array[y][x] = round(float(block[y][x]) / float(self.Q1[y][x]))"""
			array = np.round(np.divide(block, self.Q1))
		else:
			"""for y in range(self.block_size):
				for x in range(self.block_size):
					array[y][x] = block[y][x] * self.Q1[y][x]"""
			array = np.multiply(block, self.Q1)
		return array

	def set_Q1(self):
		"""Calcolo di Q1 in base a quality e Q"""
		if self.__get_qf() != 0:
			self.Q1 = self.__stretch_matrix(np.around(np.multiply(self.__get_qf(), self._Q))).astype(float)
		else:
			self.Q1 = self.__stretch_matrix(np.ones(self._Q.shape)).astype(float)

	def __stretch_matrix(self, matrix):
		return np.repeat(np.repeat(matrix, self.N, axis=0), self.N, axis=1)

	def __normalize(self, block):
		array = np.array([[0.] * (self.block_size)] * (self.block_size))
		for y in range(self.block_size):
			for x in range(self.block_size):
				if block[y][x] < 0:
					array[y][x] = 0
				elif block[y][x] > 255:
					array[y][x] = 255
				else:
					array[y][x] = block[y][x]
		return array

	def join(self, width, height):
		array = np.array([[0.] * width] * height)
		for y in xrange(len(self.image_block)):
			for x in xrange(len(self.image_block[0])):
				for j in xrange(self.block_size):
					for i in xrange(self.block_size):
						array[y*self.block_size+j][x*self.block_size+i] = self.image_block[y][x][j][i]
		self.width = width
		self.height = height
		self.original_image = array

	def compress(self, quality):
		"""Metodo che effettua la compressione JPEG8N dell'immagine"""
		self.quality = quality
		self.set_Q1()
		print "quantization matrix generated with quality=" +str(self.quality)
		for y in range(len(self.image_block)):
			for x in range(len(self.image_block[0])):
				self.image_block[y][x] = self.__dct2D(self.image_block[y][x].astype(float), inverse=False)
				self.image_block[y][x] = self.__quantitize(self.image_block[y][x], inverse=False)
		print "image compressed"
		return (self.image_block, self.quality)

	@staticmethod
	def uncompress(compressed_image_structure):
		"""MEtodo che effettua la decompressione JPEG8N e istanzia una
		nuova immagine"""
		print "uncompressing image..."
		array = compressed_image_structure[0]
		bs = len(array[0][0])
		img = JPEG8N(filename=None, N=bs//8)
		img.quality = compressed_image_structure[1]
		img.image_block = compressed_image_structure[0]
		img.set_Q1()
		for y in range(len(img.image_block)):
			for x in range(len(img.image_block[0])):
				img.image_block[y][x] = img.__quantitize(img.image_block[y][x], inverse=True)
				img.image_block[y][x] = img.__dct2D(img.image_block[y][x].astype(float), inverse=True)
				img.image_block[y][x] = img.__normalize(img.image_block[y][x])
		print "joining..."
		img.join(len(img.image_block[0])*bs, len(img.image_block)*bs)
		print "image uncompressed (" +str(img.width) + "x"+ str(img.height)+")"
		return img

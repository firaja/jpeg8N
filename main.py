import JPEG8N as j
#import matplotlib.pyplot as plt
import os
import sys
from PyQt4.QtGui import *
import numpy as np
import pylab
import matplotlib.cm as cm
import Image

class TheGui(QMainWindow):

	def __init__(self):
		super(TheGui, self).__init__()
		self.initUI()

	def initUI(self):
		# Create textbox
		self.imagepath = QLineEdit(self)
		self.imagepath.move(0, 0)
		self.imagepath.resize(300,30)

		self.enne = QLineEdit(self)
		self.enne.move(60, 60)
		self.enne.resize(240,30)

		self.quality = QLineEdit(self)
		self.quality.move(60, 90)
		self.quality.resize(240,30)

		#Add text
		self.label = QLabel(self)
		self.label.setText('quality=')
		self.label.move(5,90)

		self.label2 = QLabel(self)
		self.label2.setText('N=')
		self.label2.move(5,60)
		# Add a button
		brw = QPushButton('Browse', self)
		brw.setToolTip('Click to select an image')
		brw.clicked.connect(self.selectFile)
		brw.move(0, 30)  
		brw.resize(300,30)

		load = QPushButton('Execute', self)
		load.setToolTip('Click to select an image')
		load.clicked.connect(self.execute)
		load.move(0, 120)  
		load.resize(300,30)

		
	
		self.setGeometry(50, 50, 300, 150)
		self.setWindowTitle('JPEG8N Client')
		self.show()

	def selectFile(self):
		self.imagepath.setText(QFileDialog.getOpenFileName())

		
	def execute(self):
		f = pylab.figure()
		n = int(str(self.enne.text()))
		# inizializzazione immagine
		image = j.JPEG8N(str(self.imagepath.text()), N=n)
		q = int(str(self.quality.text()))
		# compressione
		compressed_image_structure = image.compress(q)
		f.add_subplot(1, 2, 1)
		pylab.imshow(image.original_image,cmap=cm.Greys_r)
		image = None
		# decompressione
		uncompressed_image = j.JPEG8N.uncompress(compressed_image_structure)
		# disegna le due immagine originale/compressa
		f.add_subplot(1, 2, 0)
		pylab.imshow(uncompressed_image.original_image,cmap=cm.Greys_r)
		pylab.title('quality='+str(q)+' & N='+str(n))
		pylab.show()



def main():
	
	app = QApplication(sys.argv)
	ex = TheGui()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()

import random
import uuid
from edge import EdgeShape
from direction import Direction 
import numpy as np
from PIL import Image

class PuzzlePiece:
	NIB_PERCENT = 20 / 100

	def __init__(self):
		self.imgData = None
		self.coords_x = None
		self.coords_y = None
		self.edgeGeometry = 0 # L, D, R, U (2 bits per direction totals 1 byte)
		self.id = uuid.uuid4()
		self.correctEdgeIds = []

	def getEdgeGeometry(self, direction):
		if direction == Direction.UP:
			return EdgeShape(int('{0:08b}'.format(self.edgeGeometry)[6:8], 2))
		elif direction == Direction.RIGHT:
			return EdgeShape(int('{0:08b}'.format(self.edgeGeometry)[4:6], 2))
		elif direction == Direction.DOWN:
			return EdgeShape(int('{0:08b}'.format(self.edgeGeometry)[2:4], 2))
		elif direction == Direction.LEFT:
			return EdgeShape(int('{0:08b}'.format(self.edgeGeometry)[0:2], 2))

	def setEdgeGeometry(self, direction, edgeShape):
		if direction == Direction.UP:
			self.edgeGeometry = self.edgeGeometry | (edgeShape.value)
		elif direction == Direction.RIGHT:
			self.edgeGeometry = self.edgeGeometry | (edgeShape.value << 2)
		elif direction == Direction.DOWN:
			self.edgeGeometry = self.edgeGeometry | (edgeShape.value << 4)
		elif direction == Direction.LEFT:
			self.edgeGeometry = self.edgeGeometry | (edgeShape.value << 6)

	def rotate(self):
		rotationAmount = random.randint(0, 3)
		self.rotate_defined(rotationAmount)

	def rotate_defined(self, rotationAmount):
		self.imgData = np.rot90(self.imgData, rotationAmount)

		ror = lambda val, r_bits, max_bits: \
		    ((val & (2**max_bits-1)) >> r_bits%max_bits) | \
		    (val << (max_bits-(r_bits%max_bits)) & (2**max_bits-1))

		self.edgeGeometry = ror(self.edgeGeometry, (rotationAmount) * 2, 8)

	def displayPiece(self):
		img = Image.fromarray(self.imgData, 'RGB')
		img.show()
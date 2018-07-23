import random
import uuid
from edge import EdgeShape
from direction import Direction 

class PuzzlePiece:
	NIB_PERCENT = 20 / 100

	def __init__(self):
		self.imgData = None
		#(singlePieceHeight * self.yNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, singlePieceWidth * self.xNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, 3) 
		self.coords_x = random.randint(0, 10)
		self.coords_y = random.randint(0, 10)
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
		rotationAmount = random.randint(0,3)
		self.imgData = np.rot90(self.imgData, rotationAmount)
		self.edgeGeometry = self.edgeGeometry << (2 * rotationAmount) | self.edgeGeometry >> (8 - 2 * rotationAmount)

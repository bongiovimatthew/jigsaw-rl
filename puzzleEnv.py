from enum import Enum
from PIL import Image
import numpy as np
import random
import math
class EdgeShape(Enum):
	STRAIGHT = 0
	IN = 1
	OUT = 2

	def GetComplement(edgeShape):
		if (edgeShape == STRAIGHT):
			return STRAIGHT
		if (edgeShape == IN):
			return OUT

		return IN


class Direction(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def GetComplement(direction):
		if (direction == UP):
			direction = DOWN
		if (direction == DOWN):
			direction = UP
		if (direction == RIGHT):
			direction = LEFT
		if (direction == LEFT):
			direction = RIGHT


class PuzzlePiece:
	NIB_PERCENT = 10 / 100

	def init(self, imgData, coords):
		self.imgData = imgData # tbd
		self.coords = coords
		self.edgeGeometry = 0 # L, D, R, U (2 bits per direction totals 1 byte)

	def getEdgeGeometry(self, direction):
		if direction == UP:
			return EdgeShape(int(bin(edgeGeometry)[2:][6:8]), 2)
		elif direction == RIGHT:
			return EdgeShape(int(bin(edgeGeometry)[2:][4:6]), 2)
		elif direction == DOWN:
			return EdgeShape(int(bin(edgeGeometry)[2:][2:4]), 2)
		elif direction == LEFT:
			return EdgeShape(int(bin(edgeGeometry)[2:][0:2]), 2)

	def setEdgeGeometry(self, direction, edgeShape):
		if direction == UP:
			edgeGeometry = edgeGeometry | (edgeShape)
		elif direction == RIGHT:
			edgeGeometry = edgeGeometry | (edgeShape << 2)
		elif direction == DOWN:
			edgeGeometry = edgeGeometry | (edgeShape << 4)
		elif direction == LEFT:
			edgeGeometry = edgeGeometry | (edgeShape << 6)


class Puzzle:
	piecesArray = None

	def createEdge(self, piece0Coords, piece1Coords, p0EdgeDirection):
		if (piecesArray[piece0Coords].getEdgeGeometry(p0EdgeDirection) != EdgeShape.STRAIGHT):
			return

		p1EdgeDirection = Direction.GetComplement(p0EdgeDirection)

		p0EdgeShape = EdgeShape(random.randint(1,2))
		p1EdgeShape = EdgeShape.GetComplement(p0EdgeShape)

		piecesArray[piece0Coords].setEdgeGeometry(p0EdgeDirection, p0EdgeShape)
		piecesArray[piece1Coords].setEdgeGeometry(p1EdgeDirection, p1EdgeShape)

	def generatePuzzleGeometry(self, xNumPieces, yNumPieces):
		self.piecesArray = numpy.zeros(shape=(xNumPieces,yNumPieces))
		for x in xNumPieces:
			for y in yNumPieces:
				piecesArray[x, y].coords = (x, y)
				if (x < (xNumPieces - 1)):
					createEdge((x,y), (x+1,y), Direction.RIGHT)

				if (x > 0):
					createEdge((x,y), (x-1,y), Direction.LEFT)

				if (y < (xNumPieces - 1)):
					createEdge((x,y), (x,y+1), Direction.DOWN)

				if (y > 0):
					createEdge((x,y), (x,y-1), Direction.UP)

	def breakImageToPieces(xNumPieces, yNumPieces, imageArray):
		# image = Image.open(input)
		# pad = 4 #pixels
		# padded_array = np.pad(im_array, ((pad,pad+1),(pad,pad+1),(0,0)), 'constant')
        # generatePuzzleGeometry(xNumPieces, yNumPieces)

		imgWidth, imgHeight, rgb = imageArray.shape

		nibHeight = math.floor((imgWidth / xNumPieces) * PuzzlePiece.NIB_PERCENT)
		imgSliceWidth = math.floor(imgWidth / xNumPieces) 
		singlePieceWidth = imgSliceWidth + 2 * nibHeight
		
		imgSliceHeight = math.floor(imgHeight / yNumPieces)
		singlePieceHeight = imgSliceHeight  + 2 * nibHeight 
		paddedImageArray = np.pad(imageArray, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant')
		# print(paddedImageArray)

		for x in range(xNumPieces):
			for y in range(yNumPieces):
				x1 = (x * imgSliceWidth) - nibHeight
				x2 = x1 + singlePieceWidth

				y1 = (y * imgSliceHeight) - nibHeight
				y2 = y1 + singlePieceHeight

				imgData = paddedImageArray[y1 + nibHeight : y2 + nibHeight, x1 + nibHeight : x2 + nibHeight]
				# print(singlePieceWidth)
				# print(singlePieceHeight)
				# print(imgData.shape)
				# piecesArray[x, y].imgData
				# imgDisp = Image.fromarray(imgData, 'RGB')
				# imgDisp.show()
				# aaa

			# piecesArray[x, y].imgData
		imgDisp = Image.fromarray(paddedImageArray, 'RGB')
		imgDisp.show()

def main():
	image = Image.open('zambia_map.jpg')
	im_array = np.array(image)

	Puzzle.breakImageToPieces(4,4,im_array)
	return

if __name__ == "__main__":
    main()
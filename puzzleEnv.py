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

	def GetAllDirections():
		return [UP, RIGHT, DOWN, LEFT]

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

	def __init__(self, imgLocation, xNumPieces, yNumPieces):
		self.imgFileName = imgLocation
		self.xNumPieces = xNumPieces
		self.yNumPieces = yNumPieces

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


	def addOuterNib(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		# Use square nibs for now
		nibWidth = nibHeight

		print(singlePieceImgData.shape)

		# upper box for an out nib

		# (0,0) (nibHeight, 0)
		# (0, ((singlePieceHeight - nibWidth) / 2) (nibHeight, (0, ((singlePieceHeight - nibWidth) / 2))
		
		boxHeight = math.floor((singlePieceHeight - nibWidth) / 2)
		singlePieceImgData[0:nibHeight, 0:boxHeight] = (0, 0, 0)
		singlePieceImgData[0:nibHeight, singlePieceHeight - boxHeight : singlePieceHeight ] = (0, 0, 0)


		# lower box for an out nib
		# (0, ((singlePieceHeight + nibWidth) / 2) ), (nibHeight, ((singlePieceHeight + nibWidth) / 2) )
		# (0, singplePiecehight) (nibHeight, singlePieceheight)

		# singlePieceImgData[0:nibHeight, ((singlePieceHeight + nibWidth) / 2) : singlePieceHeight ] = (0, 0, 0)
		# singlePieceImgData[0:nibHeight, ((singlePieceHeight + nibWidth) / 2) : singlePieceHeight ] = (0, 0, 0)

		return singlePieceImgData

	def addNibs(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		rotationsForOuterNibsMapping = {
			Direction.UP :0,
			Direction.RIGHT :1,
			Direction.DOWN :2,
			Direction.LEFT :3			
		}

		direction = Direction.DOWN
		# for direction in Direction.GetAllDirections():
			# if (piecesArray[x][y].getEdgeGeometry(direction) == EdgeShape.OUT):
		singlePieceImgData = np.rot90(singlePieceImgData, rotationsForOuterNibsMapping[direction])
		singlePieceImgData = self.addOuterNib(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
		singlePieceImgData = np.rot90(singlePieceImgData, 4 - rotationsForOuterNibsMapping[direction])

		return singlePieceImgData


	def breakImageToPieces(self, xNumPieces, yNumPieces, imageArray):
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
				x = 2
				y = 2
				x1 = (x * imgSliceWidth) - nibHeight
				x2 = x1 + singlePieceWidth

				y1 = (y * imgSliceHeight) - nibHeight
				y2 = y1 + singlePieceHeight

				singlePieceImgData = paddedImageArray[y1 + nibHeight : y2 + nibHeight, x1 + nibHeight : x2 + nibHeight]

				singlePieceWithNibsImgData = self.addNibs(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				# print(singlePieceWidth)
				# print(singlePieceHeight)
				# print(imgData.shape)
				# piecesArray[x, y].imgData
				imgDisp = Image.fromarray(singlePieceWithNibsImgData, 'RGB')
				imgDisp.show()
				aaa

			# piecesArray[x, y].imgData
		imgDisp = Image.fromarray(paddedImageArray, 'RGB')
		imgDisp.show()

	def generatePuzzle(self):
		# Read in image
		image = Image.open(self.imgFileName)
		im_array = np.array(image)

		# Generate the nibs that we want to use
		# generatePuzzleGeometry(self.xNumPieces, self.yNumPieces)

		# Break the image array into each piece
		self.breakImageToPieces(self.xNumPieces, self.yNumPieces,im_array)

def main():
	puzzle = Puzzle('zambia_map.jpg',4, 4)
	puzzle.generatePuzzle()
	return

if __name__ == "__main__":
    main()
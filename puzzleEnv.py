from enum import Enum
from PIL import Image
import numpy as np
import random
import math

# y = im.shape[0]
# x = im.shape[1]

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

	def __init__(self, coords):
		# self.imgData = imgData # tbd
		self.coords = coords
		self.edgeGeometry = 0 # L, D, R, U (2 bits per direction totals 1 byte)

	def getEdgeGeometry(self, direction):
		if direction == Direction.UP:
			return EdgeShape(int(bin(self.edgeGeometry)[2:][6:8], 2))
		elif direction == Direction.RIGHT:
			return EdgeShape(int(bin(self.edgeGeometry)[2:][4:6], 2))
		elif direction == Direction.DOWN:
			return EdgeShape(int(bin(self.edgeGeometry)[2:][2:4], 2))
		elif direction == Direction.LEFT:
			return EdgeShape(int(bin(self.edgeGeometry)[2:][0:2], 2))

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
		if (self.piecesArray[piece0Coords[0]][piece0Coords[1]].getEdgeGeometry(p0EdgeDirection) != EdgeShape.STRAIGHT):
			return

		p1EdgeDirection = Direction.GetComplement(p0EdgeDirection)

		p0EdgeShape = EdgeShape(random.randint(1,2))
		p1EdgeShape = EdgeShape.GetComplement(p0EdgeShape)

		piecesArray[piece0Coords[0]][piece0Coords[1]].setEdgeGeometry(p0EdgeDirection, p0EdgeShape)
		piecesArray[piece1Coords[0]][piece1Coords[1]].setEdgeGeometry(p1EdgeDirection, p1EdgeShape)

	def generatePuzzleGeometry(self, xNumPieces, yNumPieces):
		self.piecesArray = [[None for i in range(xNumPieces)] for i in range(yNumPieces)]
		for x in range(xNumPieces):
			for y in range(yNumPieces):
				self.piecesArray[y][x] = PuzzlePiece((y, x))
				if (x < (xNumPieces - 1)):
					self.createEdge((y,x), (y,x+1), Direction.RIGHT)

				if (x > 0):
					self.createEdge((y,x), (y,x-1), Direction.LEFT)

				if (y < (xNumPieces - 1)):
					self.createEdge((y,x), (y+1,x), Direction.DOWN)

				if (y > 0):
					self.createEdge((y,x), (y-1,x), Direction.UP)


	def addOuterNib(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		# Use square nibs for now
		nibWidth = nibHeight

		boxWidth = math.floor((singlePieceWidth - nibWidth) / 2)
		singlePieceImgData[0:nibHeight, 0:boxWidth] = (0, 0, 0)
		singlePieceImgData[0:nibHeight, singlePieceWidth - boxWidth : singlePieceWidth ] = (0, 0, 0)

		return singlePieceImgData

	def addInnerNib(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		# Use square nibs for now
		nibWidth = nibHeight

		boxWidth = math.floor((singlePieceWidth - nibWidth) / 2)
		singlePieceImgData[0:nibHeight, 0:singlePieceWidth] = (0, 0, 0)
		singlePieceImgData[nibHeight:2 * nibHeight, boxWidth : singlePieceWidth - boxWidth ] = (0, 0, 0)

		return singlePieceImgData
	def addNibs(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		for direction in Direction.GetAllDirections():
			if (piecesArray[x][y].getEdgeGeometry(direction) == EdgeShape.OUT):
				singlePieceImgData = np.rot90(singlePieceImgData, direction.value)
				singlePieceImgData = self.addOuterNib(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - direction.value)
			elif (piecesArray[x][y].getEdgeGeometry(direction) == EdgeShape.IN):
				singlePieceImgData = np.rot90(singlePieceImgData, rotation[direction])
				singlePieceImgData = self.addInnerNib(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - rotation[direction])
		# print(singlePieceImgData)

		return singlePieceImgData


	def breakImageToPieces(self, xNumPieces, yNumPieces, imageArray):
		imgHeight, imgWidth, rgb = imageArray.shape
		nibHeight = math.floor((imgWidth / xNumPieces) * PuzzlePiece.NIB_PERCENT)

		imgSliceWidth = math.floor(imgWidth / xNumPieces) 
		singlePieceWidth = imgSliceWidth + 2 * nibHeight
		
		imgSliceHeight = math.floor(imgHeight / yNumPieces)
		singlePieceHeight = imgSliceHeight  + 2 * nibHeight 
		paddedImageArray = np.pad(imageArray, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant')
		# print(paddedImageArray)

		finalImagePiece = None
		for x in range(xNumPieces):
			singlePuzzleCol = None
			for y in range(yNumPieces):
				x1 = (x * imgSliceWidth) - nibHeight
				x2 = x1 + singlePieceWidth

				y1 = (y * imgSliceHeight) - nibHeight
				y2 = y1 + singlePieceHeight

				singlePieceImgData = paddedImageArray[y1 + nibHeight : y2 + nibHeight, x1 + nibHeight : x2 + nibHeight]

				singlePieceWithNibsImgData = self.addNibs(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				paddedImagePiece = np.pad(singlePieceWithNibsImgData, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant', constant_value=(255,255,255))
				np.concatenate((singlePuzzleCol, paddedImagePiece), 0)

			np.concatenate((finalImagePiece, singlePuzzleCol), 0)

				# print(singlePieceWidth)
				# print(singlePieceHeight)
				# print(imgData.shape)
				# piecesArray[x, y].imgData
		imgDisp = Image.fromarray(finalImagePiece, 'RGB')
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
		self.generatePuzzleGeometry(self.xNumPieces, self.yNumPieces)

		# Break the image array into each piece
		self.breakImageToPieces(self.xNumPieces, self.yNumPieces,im_array)

def main():
	puzzle = Puzzle('zambia_map.jpg',4, 4)
	puzzle.generatePuzzle()
	return

if __name__ == "__main__":
    main()
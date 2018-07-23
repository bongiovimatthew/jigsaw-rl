from enum import Enum
from PIL import Image
import numpy as np
import random
import math
import uuid

# y = im.shape[0]
# x = im.shape[1]

class EdgeShape(Enum):
	STRAIGHT = 0
	IN = 1
	OUT = 2

	def GetComplement(edgeShape):
		if (edgeShape == EdgeShape.STRAIGHT):
			return EdgeShape.STRAIGHT
		if (edgeShape == EdgeShape.IN):
			return EdgeShape.OUT

		return EdgeShape.IN


class Direction(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def GetComplement(direction):
		if (direction == Direction.UP):
			direction = Direction.DOWN
		elif (direction == Direction.DOWN):
			direction = Direction.UP
		elif (direction == Direction.RIGHT):
			direction = Direction.LEFT
		elif (direction == Direction.LEFT):
			direction = Direction.RIGHT

		return direction

	def GetAllDirections():
		return [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

class PuzzlePiece:
	NIB_PERCENT = 20 / 100

	def __init__(self):
		self.imgData = None
		#(singlePieceHeight * self.yNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, singlePieceWidth * self.xNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, 3) 
		self.coords_x = random.randint(0, 10)
		self.coords_y = random.randint(0, 10)
		self.edgeGeometry = 0 # L, D, R, U (2 bits per direction totals 1 byte)
		self.id = uuid.uuid4()

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


class Puzzle:
	NUMBER_OF_PIECES_TO_SCALE_BY = 3

	def __init__(self, imgLocation, xNumPieces, yNumPieces):

		self.imgFileName = imgLocation
		self.xNumPieces = xNumPieces
		self.yNumPieces = yNumPieces

		# These are built out by generatePuzzle
		self.piecesArray = None
		self.listOfPiecesAvailable = None
		self.puzzleBoard = None

	def createEdge(self, piece0Coords, piece1Coords, p0EdgeDirection):
		if (self.piecesArray[piece0Coords[0]][piece0Coords[1]].getEdgeGeometry(p0EdgeDirection) != EdgeShape.STRAIGHT):
			return

		p1EdgeDirection = Direction.GetComplement(p0EdgeDirection)

		p0EdgeShape = EdgeShape(random.randint(1,2))
		p1EdgeShape = EdgeShape.GetComplement(p0EdgeShape)

		self.piecesArray[piece0Coords[0]][piece0Coords[1]].setEdgeGeometry(p0EdgeDirection, p0EdgeShape)
		self.piecesArray[piece1Coords[0]][piece1Coords[1]].setEdgeGeometry(p1EdgeDirection, p1EdgeShape)

		# print("New edge")
		# print(piece0Coords)
		# print(p0EdgeShape)
		# print(self.piecesArray[piece0Coords[0]][piece0Coords[1]].getEdgeGeometry(p0EdgeDirection))
		# print(p0EdgeDirection)

		# print("Compliment edge")
		# print(piece1Coords)
		# print(p1EdgeShape)
		# print(self.piecesArray[piece1Coords[0]][piece1Coords[1]].getEdgeGeometry(p1EdgeDirection))
		# print(p1EdgeDirection)

	def generatePuzzlePieces(self, xNumPieces, yNumPieces):
		self.piecesArray = [[PuzzlePiece((y, x)) for x in range(xNumPieces)] for y in range(yNumPieces)]
		for x in range(xNumPieces):
			for y in range(yNumPieces):
				piecesArray[y][x].coords_x = 
				piecesArray[y][x].coords_y = 

				if (x < (xNumPieces - 1)):
					self.createEdge((y,x), (y,x+1), Direction.RIGHT)

				if (x > 0):
					self.createEdge((y,x), (y,x-1), Direction.LEFT)

				if (y < (xNumPieces - 1)):
					self.createEdge((y,x), (y+1,x), Direction.DOWN)

				if (y > 0):
					self.createEdge((y,x), (y-1,x), Direction.UP)


	def addOuterNib(self, singlePieceImgData, xAxisLengthPostRotation, nibHeight):
		# Use square nibs for now
		nibWidth = nibHeight

		boxWidth = math.floor((xAxisLengthPostRotation - nibWidth) / 2)
		singlePieceImgData[0:nibHeight, 0:boxWidth] = (0, 0, 0)
		singlePieceImgData[0:nibHeight, boxWidth + nibWidth : xAxisLengthPostRotation] = (0, 0, 0)

		return singlePieceImgData

	def addInnerNib(self, singlePieceImgData, xAxisLengthPostRotation, nibHeight):
		# Use square nibs for now
		nibWidth = nibHeight

		boxWidth = math.floor((xAxisLengthPostRotation - nibWidth) / 2)
		singlePieceImgData[0:nibHeight, 0:xAxisLengthPostRotation] = (0, 0, 0)
		singlePieceImgData[nibHeight:2 * nibHeight, boxWidth : boxWidth + nibWidth ] = (0, 0, 0)

		return singlePieceImgData

	def addNibs(self, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		for direction in Direction.GetAllDirections():
			xAxisLengthPostRotation = singlePieceHeight if (direction.value % 2) else singlePieceWidth
			if (self.piecesArray[y][x].getEdgeGeometry(direction) == EdgeShape.OUT):
				singlePieceImgData = np.rot90(singlePieceImgData, direction.value)
				singlePieceImgData = self.addOuterNib(singlePieceImgData, xAxisLengthPostRotation, nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - direction.value)
			elif (self.piecesArray[y][x].getEdgeGeometry(direction) == EdgeShape.IN):
				singlePieceImgData = np.rot90(singlePieceImgData, direction.value)
				singlePieceImgData = self.addInnerNib(singlePieceImgData, xAxisLengthPostRotation, nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - direction.value)
		# print(singlePieceImgData)

		return singlePieceImgData

	# Returns the size of a single piece
	def breakImageToPieces(self, xNumPieces, yNumPieces, imageArray):
		imgHeight, imgWidth, rgb = imageArray.shape
		nibHeight = math.floor((imgWidth / xNumPieces) * PuzzlePiece.NIB_PERCENT)

		imgSliceWidth = math.floor(imgWidth / xNumPieces) 
		singlePieceWidth = imgSliceWidth + 2 * nibHeight
		
		imgSliceHeight = math.floor(imgHeight / yNumPieces)
		singlePieceHeight = imgSliceHeight  + 2 * nibHeight 
		paddedImageArray = np.pad(imageArray, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant')

		for x in range(xNumPieces):
			for y in range(yNumPieces):
				x1 = (x * imgSliceWidth) - nibHeight
				x2 = x1 + singlePieceWidth

				y1 = (y * imgSliceHeight) - nibHeight
				y2 = y1 + singlePieceHeight

				singlePieceImgData = paddedImageArray[y1 + nibHeight : y2 + nibHeight, x1 + nibHeight : x2 + nibHeight].copy()

				singlePieceImgData = self.addNibs(singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				singlePieceImgData = np.pad(singlePieceImgData, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant', constant_values = (255))
				
				self.piecesArray[y][x].imgData = singlePieceImgData

		return (singlePieceWidth, singlePieceHeight)

	def getPuzzleBoardWidth(self, singlePieceWidth):
		return singlePieceWidth * self.xNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY

	def getPuzzleBoardHeight(self, singlePieceHeight):
		return singlePieceHeight * self.yNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY

	def createPuzzleBoard(self, singlePieceWidth, singlePieceHeight):
		self.puzzleBoard = np.zeros((getPuzzleBoardHeight(singlePieceHeight), getPuzzleBoardHeight(singlePieceWidth), 3))
		return

	def getCorrectPuzzleArray(self):
		return self.piecesArray

	# Generates the randomly placed, randomly rotated pieces
	#  Rotation based on image data (no geom) 
	def createRandomPuzzlePieceArray(self):
		self.listOfPiecesAvailable = [self.piecesArray[y][x] for y in range(self.yNumPieces) for x in range(self.xNumPieces)]
		random.shuffle(self.listOfPiecesAvailable)
		for piece in listOfPiecesAvailable:
			piece.imgData = np.rot90(piece.imgData, random.randint(0,3))
		return self.listOfPiecesAvailable
	
	def getPiecesAsOneBigImage(self):
		finalImagePieces = None
		for x in range(self.xNumPieces):
			singlePuzzleCol = None
			for y in range(self.yNumPieces):
				if singlePuzzleCol is not None:
					singlePuzzleCol = np.concatenate((singlePuzzleCol, self.piecesArray[y][x].imgData), 0)
				else:
					singlePuzzleCol = self.piecesArray[y][x].imgData

			if finalImagePieces is not None:					
				finalImagePieces = np.concatenate((finalImagePieces, singlePuzzleCol), 1)
			else:
				finalImagePieces = singlePuzzleCol

		return finalImagePieces

	def displayPuzzlePieces(self):
		finalImagePieces = self.getPiecesAsOneBigImage()
		imgDisp = Image.fromarray(finalImagePieces, 'RGB')
		imgDisp.show()

	def concatImages(self, img1, img2):
		PADDING = 20

		img1Width, img1Height = img1.size[:2]
		img2Width, img2Height = img2.size[:2]

		maxHeight = np.max([img1Height, img2Height]) + PADDING
		totalWidth = img1Width+img2Width + PADDING

		newImage = Image.new('RGB', (totalWidth, maxHeight), color=(255,255,255))
		newImage.paste(img1, (PADDING, int((maxHeight - img1Height)/2))) 
		newImage.paste(img2, (img1Width + PADDING, int((maxHeight - img2Height)/2))) 

		return newImage

	def displayPuzzle(self):
		IMAGE_SCALE_RATIO = 4

		puzzleBoardImg = Image.fromarray(self.puzzleBoard, 'RGB')
		puzzleBoardImg = puzzleBoardImg.resize((math.floor(self.puzzleBoard.shape[1] / IMAGE_SCALE_RATIO), math.floor(self.puzzleBoard.shape[0] / IMAGE_SCALE_RATIO)))

		finalImagePiecesArr = self.getPiecesAsOneBigImage()
		imgPieces = Image.fromarray(finalImagePiecesArr, 'RGB')

		finalPuzzleBoard = self.concatImages(puzzleBoardImg, imgPieces)
		finalPuzzleBoard.show()

	def generatePuzzle(self):
		# Read in image
		image = Image.open(self.imgFileName)
		im_array = np.array(image)

		# Generate the nibs that we want to use
		self.generatePuzzlePieces(self.xNumPieces, self.yNumPieces)

		# Break the image array into each piece
		singlePieceWidth, singlePieceHeight = self.breakImageToPieces(self.xNumPieces, self.yNumPieces, im_array)

		self.puzzleBoard = self.createPuzzleBoard(singlePieceWidth, singlePieceHeight)

		# Display the puzzle pieces
		# self.displayPuzzlePieces()
		
		#self.displayPuzzle()


def main():
	puzzle = Puzzle('zambia_map.jpg',4, 4)
	puzzle.generatePuzzle()
	return

if __name__ == "__main__":
    main()
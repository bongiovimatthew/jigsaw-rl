from PIL import Image
import numpy as np
import random 
import math 

from Environment.puzzlePiece import PuzzlePiece 
from Environment.direction import Direction 
from Environment.edge import EdgeShape 
from Environment.puzzle import Puzzle 

class PuzzleFactory:
	NUMBER_OF_PIECES_TO_SCALE_BY = 2

	def __init__(self):
		return 

	def generatePuzzle(self, imgFileName, xNumPieces, yNumPieces):
		# Read in image
		image = Image.open(imgFileName)
		im_array = np.array(image)

		puzzle = Puzzle(xNumPieces, yNumPieces)

		# Generate the geometry of the puzzle pieces 
		self.generatePuzzlePiecesGeometry(puzzle)

		# Slice the full image, and apply the correct edge geometry to the image data
		puzzle.singlePieceWidth, puzzle.singlePieceHeight = self.generatePuzzlePieceImages(puzzle, im_array)
		self.createPuzzleBoard(puzzle)

		self.generateCorrectEdges(puzzle)

		return puzzle 

	def generateCorrectEdges(self, puzzle):
		for x in range(puzzle.xNumPieces):
			for y in range(puzzle.yNumPieces):
				# UPPER EDGE
				if y > 0:
					puzzle.piecesArray[y][x].correctEdgeIds.append(puzzle.piecesArray[y - 1][x].id)
				else:
					puzzle.piecesArray[y][x].correctEdgeIds.append(0)

				# RIGHT EDGE
				if x < (puzzle.xNumPieces - 1):
					puzzle.piecesArray[y][x].correctEdgeIds.append(puzzle.piecesArray[y][x + 1].id)
				else:
					puzzle.piecesArray[y][x].correctEdgeIds.append(0)

				# DOWN EDGE
				if y < (puzzle.yNumPieces - 1):
					puzzle.piecesArray[y][x].correctEdgeIds.append(puzzle.piecesArray[y + 1][x].id)
				else:
					puzzle.piecesArray[y][x].correctEdgeIds.append(0)

				# LEFT EDGE
				if x > 0:
					puzzle.piecesArray[y][x].correctEdgeIds.append(puzzle.piecesArray[y][x - 1].id)
				else:
					puzzle.piecesArray[y][x].correctEdgeIds.append(0)	

	def createEdge(self, puzzle, piece0Coords, piece1Coords, p0EdgeDirection):
		if (puzzle.piecesArray[piece0Coords[0]][piece0Coords[1]].getEdgeGeometry(p0EdgeDirection) != EdgeShape.STRAIGHT):
			return

		p1EdgeDirection = Direction.GetComplement(p0EdgeDirection)

		p0EdgeShape = EdgeShape(random.randint(1,2))
		p1EdgeShape = EdgeShape.GetComplement(p0EdgeShape)

		puzzle.piecesArray[piece0Coords[0]][piece0Coords[1]].setEdgeGeometry(p0EdgeDirection, p0EdgeShape)
		puzzle.piecesArray[piece1Coords[0]][piece1Coords[1]].setEdgeGeometry(p1EdgeDirection, p1EdgeShape)

	def generatePuzzlePiecesGeometry(self, puzzle):
		puzzle.piecesArray = [[PuzzlePiece(y, x) for x in range(puzzle.xNumPieces)] for y in range(puzzle.yNumPieces)]

		for x in range(puzzle.xNumPieces):
			for y in range(puzzle.yNumPieces):
				if (x < (puzzle.xNumPieces - 1)):
					self.createEdge(puzzle, (y,x), (y,x+1), Direction.RIGHT)

				if (x > 0):
					self.createEdge(puzzle, (y,x), (y,x-1), Direction.LEFT)

				if (y < (puzzle.xNumPieces - 1)):
					self.createEdge(puzzle, (y,x), (y+1,x), Direction.DOWN)

				if (y > 0):
					self.createEdge(puzzle, (y,x), (y-1,x), Direction.UP)

				# print("y:{0}, x:{1}, direction:{2}, getEdgeGeometry:{3}".format(y, x, Direction.RIGHT, puzzle.piecesArray[y][x].getEdgeGeometry(Direction.RIGHT)))

				# print("y:{0}, x:{1}, direction:{2}, getEdgeGeometry:{3}".format(y, x, Direction.DOWN, puzzle.piecesArray[y][x].getEdgeGeometry(Direction.DOWN)))

				# print("y:{0}, x:{1}, direction:{2}, getEdgeGeometry:{3}".format(y, x, Direction.LEFT, puzzle.piecesArray[y][x].getEdgeGeometry(Direction.LEFT)))

				# print("y:{0}, x:{1}, direction:{2}, getEdgeGeometry:{3}".format(y, x, Direction.UP, puzzle.piecesArray[y][x].getEdgeGeometry(Direction.UP)))

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

	def addNibs(self, puzzle, singlePieceImgData, singlePieceDimensions, coords, nibHeight):
		x, y = coords
		singlePieceWidth, singlePieceHeight = singlePieceDimensions

		for direction in Direction.GetAllDirections():
			xAxisLengthPostRotation = singlePieceHeight if (direction.value % 2) else singlePieceWidth
			if (puzzle.piecesArray[y][x].getEdgeGeometry(direction) == EdgeShape.OUT):
				singlePieceImgData = np.rot90(singlePieceImgData, direction.value)
				singlePieceImgData = self.addOuterNib(singlePieceImgData, xAxisLengthPostRotation, nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - direction.value)
			elif (puzzle.piecesArray[y][x].getEdgeGeometry(direction) == EdgeShape.IN):
				singlePieceImgData = np.rot90(singlePieceImgData, direction.value)
				singlePieceImgData = self.addInnerNib(singlePieceImgData, xAxisLengthPostRotation, nibHeight)
				singlePieceImgData = np.rot90(singlePieceImgData, 4 - direction.value)

		return singlePieceImgData

	# Returns the size of a single piece
	def generatePuzzlePieceImages(self, puzzle, imageArray):
		
		imgHeight, imgWidth, rgb = imageArray.shape
		nibHeight = math.floor((imgWidth / puzzle.xNumPieces) * PuzzlePiece.NIB_PERCENT)

		imgSliceWidth = math.floor(imgWidth / puzzle.xNumPieces) 
		singlePieceWidth = imgSliceWidth + 2 * nibHeight
		
		imgSliceHeight = math.floor(imgHeight / puzzle.yNumPieces)
		singlePieceHeight = imgSliceHeight  + 2 * nibHeight 
		paddedImageArray = np.pad(imageArray, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant')
		
		for x in range(puzzle.xNumPieces):
			for y in range(puzzle.yNumPieces):
				x1 = (x * imgSliceWidth) - nibHeight
				x2 = x1 + singlePieceWidth

				y1 = (y * imgSliceHeight) - nibHeight
				y2 = y1 + singlePieceHeight

				singlePieceImgData = paddedImageArray[y1 + nibHeight : y2 + nibHeight, x1 + nibHeight : x2 + nibHeight].copy()

				singlePieceImgData = self.addNibs(puzzle, singlePieceImgData, (singlePieceWidth, singlePieceHeight), (x, y), nibHeight)
				#singlePieceImgData = np.pad(singlePieceImgData, ((nibHeight, nibHeight), (nibHeight, nibHeight),(0,0)), 'constant', constant_values = (255))
				
				puzzle.piecesArray[y][x].imgData = singlePieceImgData

		return (singlePieceImgData.shape[1], singlePieceImgData.shape[0])

	def createPuzzleBoard(self, puzzle):
		puzzle.puzzleBoard = np.zeros((puzzle.singlePieceHeight * puzzle.yNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, puzzle.singlePieceWidth * puzzle.xNumPieces * self.NUMBER_OF_PIECES_TO_SCALE_BY, 3), dtype=np.uint8)
	
	def createRandomPuzzlePieceArray(self, puzzle):		
		listOfPiecesAvailable = [puzzle.piecesArray[y][x] for y in range(puzzle.yNumPieces) for x in range(puzzle.xNumPieces)]
		random.shuffle(listOfPiecesAvailable)

		return listOfPiecesAvailable

	# Generates the randomly placed, randomly rotated pieces
	#  Rotation based on image data (no geom) 
	def placePiecesOnBoard(puzzle, listOfPiecesAvailable):		
		sideDimension = puzzle.xNumPieces * PuzzleFactory.NUMBER_OF_PIECES_TO_SCALE_BY

		initialPieceGuidArray = []

		for i in range(sideDimension):
			initialPieceGuidArray.append([])
			for j in range(sideDimension):
				initialPieceGuidArray[i].append([])
		 
		for piece in listOfPiecesAvailable:
			# piece.rotate()

			done = False 
			while not done: 
				piece.coords_x = piece.correct_coords_x # random.randint(0, sideDimension - 1)
				piece.coords_y = piece.correct_coords_y # random.randint(0, sideDimension - 1)

				#piece.coords_x = random.randint(0, sideDimension - 1)
				#piece.coords_y = random.randint(0, sideDimension - 1)

				if (piece.coords_x == 2) and (piece.coords_y == 2):
					piece.coords_x = 3 # random.randint(0, sideDimension - 1)
					# piece.coords_y = 4 # random.randint(0, sideDimension - 1)

				if len(initialPieceGuidArray[piece.coords_y][piece.coords_x]) == 0:
					initialPieceGuidArray[piece.coords_y][piece.coords_x].append(piece.id) 
					done = True 

		return initialPieceGuidArray
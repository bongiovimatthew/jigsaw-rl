from enum import Enum
from PIL import Image
from puzzlePiece import PuzzlePiece
from edge import EdgeShape
from direction import Direction
import numpy as np
import random
import math

class Puzzle:
	
	def __init__(self, xNumPieces, yNumPieces ):

		self.xNumPieces = xNumPieces
		self.yNumPieces = yNumPieces

		# These are built out by generatePuzzle
		self.piecesArray = None
		self.puzzleBoard = None
		self.singlePieceWidth = None
		self.singlePieceHeight = None 

	def getCorrectPuzzleArray(self):
		return self.piecesArray

	# Generates the randomly placed, randomly rotated pieces
	#  Rotation based on image data (no geom) 
	def createRandomPuzzlePieceArray(self):
		listOfPiecesAvailable = [self.piecesArray[y][x] for y in range(self.yNumPieces) for x in range(self.xNumPieces)]
		random.shuffle(listOfPiecesAvailable)
		for piece in listOfPiecesAvailable:
			piece.rotate()
		return listOfPiecesAvailable



	#
	# Puzzle Display Functions 
	#

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

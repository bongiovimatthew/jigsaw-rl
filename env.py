import numpy as np

# Remove when done testing 
from PIL import Image


class State:
	
	def __init__(self, puzzle):

		self.pieceState = puzzle.createRandomPuzzlePieceArray()
		self.board = puzzle.puzzleBoard
		self.solution = puzzle.getCorrectPuzzleArray()
		
	def getStateImage(self):
		boardCopy = self.board.copy()
		piece = self.pieceState[0]

		for piece in self.pieceState: 
			baseY = piece.coords_y * piece.imgData.shape[0]
			yHeight = piece.imgData.shape[0]

			baseX = piece.coords_x * piece.imgData.shape[1]
			xWidth = piece.imgData.shape[1]

			print(boardCopy.shape)
			print(baseX, baseY)
			print(piece.coords_x, piece.coords_y)

			print(boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth].shape)
			print(piece.imgData.shape)
			boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()
			
			img = Image.fromarray(boardCopy, 'RGB')
			img.show()
			dfk

		return boardCopy

	def getPieceUsingId(self, id):
		for piece in self.pieceState:
			if piece.id == id:
				return piece

	def getScoreOfAPieceInASingleDirection(self, piece, directionToLook, adjacentCoords_x, adjacentCoords_y):
		CORRECT_IMAGE_SCORE = 100
		CORRECT_GEOMMETRY_SCORE = 10
		INCORRECT_GEOMMETRY_SCORE = -2
		NOT_CONNECTED_SCORE = -1

		score = 0
		pieceEdgeGeommetry = piece.getEdgeGeometry(directionToLook)

		if (pieceEdgeGeommetry == EdgeShape.STRAIGHT):
			if adjacentCoords_x < 0 or adjacentCoords_y < 0:
				score += CORRECT_GEOMMETRY_SCORE + CORRECT_IMAGE_SCORE
			else:
				adjacentPieceId = self.puzzleBoardAsPiecesArray[adjacentCoords_y][adjacentCoords_x]
				if (adjacentPieceId != 0):
					score += INCORRECT_GEOMMETRY_SCORE
				else:
					score += CORRECT_GEOMMETRY_SCORE + CORRECT_IMAGE_SCORE


		# Account for IN and OUT
		else:
			if adjacentCoords_x < 0 or adjacentCoords_y < 0:
				score += NOT_CONNECTED_SCORE
			else:
				adjacentPieceId = self.puzzleBoardAsPiecesArray[adjacentCoords_y][adjacentCoords_x]
				if (adjacentPieceId == 0):
					score += NOT_CONNECTED_SCORE
				else:
					adjacentPieceDirection = Direction.GetComplement(directionToLook)
					adjacentPieceGeommetry = self.getPieceUsingId(adjacentPieceId).getEdgeGeometry(adjacentPieceDirection) 
					if adjacentPieceGeommetry == EdgeShape.GetComplement(pieceEdgeGeommetry):
						score += CORRECT_GEOMMETRY_SCORE
						if piece.correctEdges[directionToLook] == adjacentPieceId:
							score += CORRECT_IMAGE_SCORE
					else:
						score += INCORRECT_GEOMMETRY_SCORE

		return score

	def getScoreOfCurrentState(self): 
		score = 0
		for piece in self.pieceState: 
			# piece to the left
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.LEFT, piece.coords_x - 1, piece.coords_y)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.RIGHT, piece.coords_x + 1, piece.coords_y)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.UP, piece.coords_x, piece.coords_y - 1)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.DOWN, piece.coords_x, piece.coords_y + 1)


def main():
	state = State()

	return

if __name__ == "__main__":
    main()
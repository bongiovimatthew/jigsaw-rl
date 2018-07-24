import numpy as np
from direction import Direction 
from edge import EdgeShape

#from gym.envs.toy_text import discrete

### Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self):
        raise NotImplementedError('Inheriting classes must override step')

class ActionSpace(object):
    
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)

class PuzzleEnvironment(Environment):
	CORRECT_IMAGE_SCORE = 100
	CORRECT_GEOMMETRY_SCORE = 10
	INCORRECT_GEOMMETRY_SCORE = -2
	NOT_CONNECTED_SCORE = -1

	
	def __init__(self, puzzle, randomizedPieces):

		self.pieceState = randomizedPieces 
		self.board = puzzle.puzzleBoard
		self.solution = puzzle.getCorrectPuzzleArray()

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

	def _convert_state(self, state):
        converted = np.unravel_index(state, self.shape)
        return np.asarray(list(converted), dtype=np.float32)

	def step(self, action):
        reward = self.getScoreOfCurrentState()
        done = self.isDone(reward)
        if done:
        	reward *= 100

        # info = {'prob':self.P[self.s][action][0][0]}
        # self.s = self.P[self.s][action][0][1]

        return (self._convert_state(self.s, action), reward, done, info)
		
	def render(self):
		boardCopy = self.board.copy()
		piece = self.pieceState[0]

		for piece in self.pieceState: 
			baseY = piece.coords_y * piece.imgData.shape[0]
			yHeight = piece.imgData.shape[0]

			baseX = piece.coords_x * piece.imgData.shape[1]
			xWidth = piece.imgData.shape[1]

			boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()
			
		return boardCopy

	def isDone(self, reward):
		if reward == (PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE) * self.solution.shape[1] * self.solution.shape[0]:
			return True

		return False


	def getPieceUsingId(self, id):
		for piece in self.pieceState:
			if piece.id == id:
				return piece

	def getScoreOfAPieceInASingleDirection(self, piece, directionToLook, adjacentCoords_x, adjacentCoords_y):
		score = 0
		pieceEdgeGeommetry = piece.getEdgeGeometry(directionToLook)

		if (pieceEdgeGeommetry == EdgeShape.STRAIGHT):
			if adjacentCoords_x < 0 or adjacentCoords_y < 0:
				score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE
			else:
				adjacentPieceId = self.puzzleBoardAsPiecesArray[adjacentCoords_y][adjacentCoords_x]
				if (adjacentPieceId != 0):
					score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE
				else:
					score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE


		# Account for IN and OUT
		else:
			if adjacentCoords_x < 0 or adjacentCoords_y < 0:
				score += PuzzleEnvironment.NOT_CONNECTED_SCORE
			else:
				adjacentPieceId = self.puzzleBoardAsPiecesArray[adjacentCoords_y][adjacentCoords_x]
				if (adjacentPieceId == 0):
					score += PuzzleEnvironment.NOT_CONNECTED_SCORE
				else:
					adjacentPieceDirection = Direction.GetComplement(directionToLook)
					adjacentPieceGeommetry = self.getPieceUsingId(adjacentPieceId).getEdgeGeometry(adjacentPieceDirection) 
					if adjacentPieceGeommetry == EdgeShape.GetComplement(pieceEdgeGeommetry):
						score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE
						if piece.correctEdges[directionToLook] == adjacentPieceId:
							score += PuzzleEnvironment.CORRECT_IMAGE_SCORE
					else:
						score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE

		return score

	def getScoreOfCurrentState(self): 
		score = 0
		for piece in self.pieceState: 
			# piece to the left
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.LEFT, piece.coords_x - 1, piece.coords_y)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.RIGHT, piece.coords_x + 1, piece.coords_y)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.UP, piece.coords_x, piece.coords_y - 1)
			score += self.getScoreOfAPieceInASingleDirection(piece, Direction.DOWN, piece.coords_x, piece.coords_y + 1)
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
	
	# actions: 
	# 0 - cycle selected piece 
	# 1 - rotate 90 counterclockwise once 
	# 2 - rot90 cc twice 
	# 3 - rot90 cc three times 
	# 4 - translate up 
	# 5 - translate right 
	# 6 - translate down 
	# 7 - translate left 
	MAX_ACTIONS_NUM = 7

	def __init__(self, puzzle, randomizedPieces):
		
		self.pieceState = randomizedPieces 
		#self.board = puzzle.puzzleBoard
		#self.solution = puzzle.getCorrectPuzzleArray()
		
		self.puzzle = puzzle 

		self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))

	def action(self): 
		return self.action_space

    def _limit_coordinates(self, coord):
        #coord[0] = min(coord[0], self.shape[0] - 1)
        #coord[0] = max(coord[0], 0)
        #coord[1] = min(coord[1], self.shape[1] - 1)
        #coord[1] = max(coord[1], 0)
        #return coord
        return

	def _convert_state(self, state, action):

		# Move the selected 
        #converted = np.unravel_index(state, self.shape)
        #return np.asarray(list(converted), dtype=np.float32)
        return 

	def step(self, action):
        #reward = self.P[self.s][action][0][2]
        #done = self.P[self.s][action][0][3]
        #info = {'prob':self.P[self.s][action][0][0]}
        #self.s = self.P[self.s][action][0][1]

        #return (self._convert_state(self.s), reward, done, info)
		return 

	def render(self):
		boardCopy = self.puzzle.puzzleBoard.copy()
		piece = self.pieceState[0]

		for piece in self.pieceState: 
			baseY = piece.coords_y * piece.imgData.shape[0]
			yHeight = piece.imgData.shape[0]

			baseX = piece.coords_x * piece.imgData.shape[1]
			xWidth = piece.imgData.shape[1]

			boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()
			
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
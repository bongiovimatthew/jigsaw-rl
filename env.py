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

    def __init__(self, puzzle, randomizedPieces, guidArray):
        
        self.pieceState = randomizedPieces 
        self.puzzle = puzzle

        # Contains the relative position of the piece IDs in the current state 
        self.guidArray = guidArray

        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))

    def action(self): 
        return self.action_space

    def step(self, action):
        reward = self.getScoreOfCurrentState()
        done = self.isDone(reward)
        if done:
            reward *= 100

        # info = {'prob':self.P[self.s][action][0][0]}
        # self.s = self.P[self.s][action][0][1]

        return (self._convert_state(self.s, action), reward, done, info)
        
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

    def isDone(self, reward):
        if reward == (PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE) * self.puzzle.getCorrectPuzzleArray.shape[1] * self.puzzle.getCorrectPuzzleArray.shape[0]:
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
            if adjacentCoords_x < 0 or adjacentCoords_y < 0 or adjacentCoords_x >=self.puzzle.xNumPieces or adjacentCoords_y >=self.puzzle.yNumPieces:
                print("Coords of adjacent")
                print(adjacentCoords_x, adjacentCoords_y)
                score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE
            else:
                adjacentPieceId = self.guidArray[adjacentCoords_y][adjacentCoords_x]
                print("AdjpieceID")
                print(adjacentPieceId)
                if (adjacentPieceId != 0):
                    score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE
                else:
                    score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE

        # Account for IN and OUT
        else:
            if adjacentCoords_x < 0 or adjacentCoords_y < 0 or adjacentCoords_x >=self.puzzle.xNumPieces or adjacentCoords_y >=self.puzzle.yNumPieces:
                score += PuzzleEnvironment.NOT_CONNECTED_SCORE
            else:
                adjacentPieceId = self.guidArray[adjacentCoords_y][adjacentCoords_x]
                if (adjacentPieceId == 0):
                    score += PuzzleEnvironment.NOT_CONNECTED_SCORE
                else:
                    adjacentPieceDirection = Direction.GetComplement(directionToLook)
                    adjacentPieceGeommetry = self.getPieceUsingId(adjacentPieceId).getEdgeGeometry(adjacentPieceDirection) 
                    if adjacentPieceGeommetry == EdgeShape.GetComplement(pieceEdgeGeommetry):
                        score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE
                        if piece.correctEdgeIds[directionToLook] == adjacentPieceId:
                            score += PuzzleEnvironment.CORRECT_IMAGE_SCORE
                    else:
                        score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE

        return score

    def getScoreOfCurrentState(self): 
        score = 0
        count = 0
        for piece in self.pieceState: 
            # piece to the left
            lScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.LEFT, piece.coords_x - 1, piece.coords_y)
            rScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.RIGHT, piece.coords_x + 1, piece.coords_y)
            uScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.UP, piece.coords_x, piece.coords_y - 1)
            dScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.DOWN, piece.coords_x, piece.coords_y + 1)
            
            print("Piece count {0}".format(count))
            print("l,r,u,d")
            print(lScore, rScore, uScore, dScore)
            count += 1

        return lScore + rScore + uScore + dScore
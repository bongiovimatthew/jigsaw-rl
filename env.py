import numpy as np
from direction import Direction 
from edge import EdgeShape
from enum import Enum

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

class Actions(Enum):
    ACTION_CYCLE = 0 
    ACTION_ROT90_1 = 1
    ACTION_ROT90_2 = 2
    ACTION_ROT90_3 = 3
    ACTION_TRANS_UP = 4
    ACTION_TRANS_RIGHT = 5
    ACTION_TRANS_DOWN = 6
    ACTION_TRANS_LEFT = 7

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
        
        # pieceState is an array of PuzzlePiece objects
        self.pieceState = randomizedPieces 
        self.puzzle = puzzle
        # Contains the relative position of the piece IDs in the current state 
        self.guidArray = guidArray
        self.currentPieceIndex = 0
        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))

        self.debugMode = True 

    def action(self): 
        return self.action_space

    def _rotate_piece(self, pieceId, numRotations):
        # Update pieceState 
        for piece in self.pieceState: 
            if piece.id == pieceId: 
                piece.rotate_defined(numRotations)

        # No guidArray updates needed 
        return

    def _translate_piece(self, pieceId, direction): 
        # Update guidArray
        maxX = len(self.guidArray) - 1
        maxY = maxX 

        for x in range(len(self.guidArray)):
            for y in range(len(self.guidArray)):
                
                if pieceId in self.guidArray[y][x]: 
                    newX = x 
                    newY = y 

                    if direction == Direction.UP:
                        newY = y - 1
                        if newY < 0: 
                            newY = 0
                    elif direction == Direction.RIGHT:
                        newX = x + 1
                        if newX > maxX: 
                            newX = maxX
                    elif direction == Direction.DOWN:
                        newY = y + 1
                        if newY > maxY: 
                            newY = maxY 
                    elif direction == Direction.LEFT:
                        newX = x - 1
                        if newX < 0: 
                            newX = 0 

                    # print("y,x")
                    # print(y, x)
                    # print("newy,newx")
                    # print(newY, newX)
                    # print("guidArray")
                    # print(self.guidArray)

                    self.guidArray[y][x].remove(pieceId)
                    self.guidArray[newY][newX].append(pieceId)
                    
                    # Update pieceState 
                    for piece in self.pieceState: 
                        if piece.id == pieceId: 
                            piece.coords_x = newX
                            piece.coords_y = newY
                    return

    def _convert_state(self, action):

        if action == Actions.ACTION_CYCLE.value: 
            self.currentPieceIndex = (self.currentPieceIndex + 1) % (len(self.pieceState) - 1)

        if action >= Actions.ACTION_ROT90_1.value and action <= Actions.ACTION_ROT90_3.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            numRotations = action
            self._rotate_piece(currentPiece.id, numRotations) 
        
        if action >= Actions.ACTION_TRANS_UP.value and action <= Actions.ACTION_TRANS_LEFT.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            direction = directions[action - 4]
            self._translate_piece(currentPiece.id, direction)

        return 

    def step(self, action):
        reward = 1 #self.getScoreOfCurrentState()
        done = False #self.isDone(reward)
        if done:
            reward *= 100

        # info = {'prob':self.P[self.s][action][0][0]}
        # self.s = self.P[self.s][action][0][1]

        print("Peforming Action: {0}".format(Actions(action)))
        return (self._convert_state(action), reward, done)
        
    def render(self):
        boardCopy = self.puzzle.puzzleBoard.copy()
        piece = self.pieceState[0]
        count = 0

        for piece in self.pieceState: 
            baseY = piece.coords_y * piece.imgData.shape[0]
            yHeight = piece.imgData.shape[0]

            baseX = piece.coords_x * piece.imgData.shape[1]
            xWidth = piece.imgData.shape[1]

            boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()

            if self.debugMode and self.currentPieceIndex == count: 
                # Add a green bar on the current piece 
                boardCopy[ baseY : baseY + yHeight, baseX : baseX + 15] = [0, 255, 0]                
            count += 1
            
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
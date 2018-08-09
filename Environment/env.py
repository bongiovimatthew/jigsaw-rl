import numpy as np
from enum import Enum

from Environment.direction import Direction 
from Environment.edge import EdgeShape 
from Environment.puzzleFactory import PuzzleFactory
from PIL import Image
from celery.contrib import rdb

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
    ACTION_TRANS_UP = 1
    ACTION_TRANS_RIGHT = 2
    ACTION_TRANS_DOWN = 3
    ACTION_TRANS_LEFT = 4
    ACTION_ROT90_1 = 5
    ACTION_ROT90_2 = 6
    ACTION_ROT90_3 = 7

class PuzzleEnvironment(Environment):
    CORRECT_IMAGE_SCORE = 4
    INCORRECT_OVERLAY_SCORE = -200
    CORRECT_GEOMMETRY_SCORE = 1
    INCORRECT_GEOMMETRY_SCORE = -2
    NOT_CONNECTED_SCORE = -1
    CORRECT_PLACEMENT_SCORE = 100

    MAX_ACTIONS_NUM = 5

    def __init__(self):
        self.oldScore = 0
        self.debugMode = True
        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))

        # Generate the puzzle 
        factory = PuzzleFactory()
        puzzle = factory.generatePuzzle('images\\rainier_small.jpg', 3, 3)
        randomizedPieces = factory.createRandomPuzzlePieceArray(puzzle)
        
        # pieceState is an array of PuzzlePiece objects
        self.pieceState = randomizedPieces 
        self.puzzle = puzzle

        self.setupEnvironment()

    def reset(self):
        print("RESETTING")
        self.setupEnvironment()
        return self.render()

    def setupEnvironment(self):
        # Contains the relative position of the piece IDs in the current state 
        self.guidArray = PuzzleFactory.placePiecesOnBoard(self.puzzle, self.pieceState)

        self.currentPieceIndex = 0
        count = 0 
        for piece in self.pieceState: 
            if (piece.coords_y == 2) and piece.coords_x == 3:
                self.currentPieceIndex = count

            if (self.debugMode):
                print("piece.guid:{0}, piece.coords_x:{1}, piece.coords_y:{2}".format(piece.id, piece.coords_x, piece.coords_y))

            count += 1

        self.oldScore = self.getScoreOfCurrentState()
        self.stepCount = 0

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

        # for x in range(len(self.guidArray)):
        #     for y in range(len(self.guidArray)):
        #         print(" piece.coords_x:{1}, piece.coords_y:{2} guidArray:{0}".format(self.guidArray[y][x], x, y))

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

                    if len(self.guidArray[newY][newX]) == 0: 
                        self.guidArray[y][x].remove(pieceId)
                        self.guidArray[newY][newX].append(pieceId)
                        
                        # Update pieceState 
                        for piece in self.pieceState: 
                            if piece.id == pieceId: 
                                piece.coords_x = newX
                                piece.coords_y = newY

                        print("SUCCESS MOVE, pieceId:{0}, newY:{1}, newX:{2} maxX:{3} direction:{4} x:{5} y:{6} guidArr:{7}".format(pieceId, newY, newX, maxX, direction, x, y, self.guidArray[newY][newX]))

                    else:
                        print("BLOCKED MOVE, pieceId:{0}, newY:{1}, newX:{2} maxX:{3} direction:{4} x:{5} y:{6} guidArr:{7}".format(pieceId, newY, newX, maxX, direction, x, y, self.guidArray[newY][newX]))
                    


                    return

    def _convert_state(self, action):
        if action == Actions.ACTION_CYCLE.value: 
            self.currentPieceIndex = (self.currentPieceIndex + 1) % len(self.pieceState)

        if action >= Actions.ACTION_ROT90_1.value and action <= Actions.ACTION_ROT90_3.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            numRotations = action
            self._rotate_piece(currentPiece.id, numRotations) 
        
        if action >= Actions.ACTION_TRANS_UP.value and action <= Actions.ACTION_TRANS_LEFT.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            direction = directions[action - 1]
            self._translate_piece(currentPiece.id, direction)

        return self.render()

    def step(self, action):
        self.stepCount += 1
        next_state = self._convert_state(action)
        currentScore = self.getScoreOfCurrentState()
        done = self.isMaxReward(currentScore) or (self.stepCount > 500)

        tempOldScore = self.oldScore
        self.oldScore = currentScore

        reward = currentScore - tempOldScore
        if self.isMaxReward(currentScore):
            # reward *= 100
            reward = 1
        else:
            reward = 0

        info = {'score':currentScore, 'oldScore': tempOldScore, 'action': action, 'step': self.stepCount}

        if (self.debugMode):
            print("Current Reward: {0}, IsDone: {1}, currentScore: {2}, oldScore: {3}".format(reward, done, currentScore, tempOldScore))
            print("Performing Action: {0}, currentPieceIndex: {1}".format(Actions(action), self.currentPieceIndex))

        if (done):
            print("COMPLETED EPISODE!, reward:{0} currentScore:{1}\n\n\n\n\n\n".format(reward, currentScore))

        if (action == Actions.ACTION_CYCLE.value):
            reward = -1            

        reward = np.clip(reward, -2, 2)
        reward = np.float32(reward)
        return (next_state, reward, done, info)
        
    def render(self, mode=None):
        boardCopy = self.puzzle.puzzleBoard.copy()
        piece = self.pieceState[0]
        count = 0

        for piece in self.pieceState: 
            baseY = piece.coords_y * piece.imgData.shape[0]
            yHeight = piece.imgData.shape[0]

            baseX = piece.coords_x * piece.imgData.shape[1]
            xWidth = piece.imgData.shape[1]

            boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()

            if self.currentPieceIndex == count: 
                print("CURRENTPIECeINDEX: {0}", self.currentPieceIndex)
                # Add a green bar on the current piece 
                greenSquareW = 5
                greenSquareH = 5
                boardCopy[ baseY : baseY + greenSquareH, baseX : baseX + greenSquareW] = [0, 255, 0]                
                boardCopy[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX : baseX + greenSquareW] = [0, 255, 0]                
                boardCopy[ baseY : baseY + greenSquareH, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
                boardCopy[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
            count += 1

        return boardCopy

    def isMaxReward(self, reward):
        if reward == (PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE) * len(self.puzzle.getCorrectPuzzleArray()) * len(self.puzzle.getCorrectPuzzleArray()) * 4 + PuzzleEnvironment.CORRECT_PLACEMENT_SCORE * len(self.puzzle.getCorrectPuzzleArray()) * len(self.puzzle.getCorrectPuzzleArray()): 
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
            if adjacentCoords_x < 0 or adjacentCoords_y < 0 or adjacentCoords_x >= len(self.guidArray[0]) or adjacentCoords_y >= len(self.guidArray):

                score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE
            else:
                adjacentPieceIdLength = len(self.guidArray[adjacentCoords_y][adjacentCoords_x])

                if (adjacentPieceIdLength != 0):
                    score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE 
                else:
                    score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE

        # Account for IN and OUT
        else:
            if adjacentCoords_x < 0 or adjacentCoords_y < 0 or adjacentCoords_x >= len(self.guidArray[0]) or adjacentCoords_y >= len(self.guidArray):
                score += PuzzleEnvironment.NOT_CONNECTED_SCORE
            else:
                adjacentPieceIds = self.guidArray[adjacentCoords_y][adjacentCoords_x]
                if (len(adjacentPieceIds) == 0):
                    score += PuzzleEnvironment.NOT_CONNECTED_SCORE
                else:
                    for adjacentPieceId in adjacentPieceIds:
                        adjacentPieceDirection = Direction.GetComplement(directionToLook)
                        adjacentPieceGeommetry = self.getPieceUsingId(adjacentPieceId).getEdgeGeometry(adjacentPieceDirection) 
                        if adjacentPieceGeommetry == EdgeShape.GetComplement(pieceEdgeGeommetry):
                            score += PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE
                            if piece.correctEdgeIds[directionToLook.value] == adjacentPieceId:
                                score += PuzzleEnvironment.CORRECT_IMAGE_SCORE
                        else:
                            score += PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE

        return score

    def getScoreOfCurrentState(self): 
        score = 0
        count = 0

        for piece in self.pieceState: 
            pieceScore = 0
            # piece to the left
            lScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.LEFT, piece.coords_x - 1, piece.coords_y)
            rScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.RIGHT, piece.coords_x + 1, piece.coords_y)
            uScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.UP, piece.coords_x, piece.coords_y - 1)
            dScore = self.getScoreOfAPieceInASingleDirection(piece, Direction.DOWN, piece.coords_x, piece.coords_y + 1)
            count += 1
            pieceScore += lScore + rScore + uScore + dScore

            if len(self.guidArray[piece.coords_y][piece.coords_x]) > 1:
                pieceScore += PuzzleEnvironment.INCORRECT_OVERLAY_SCORE

            if piece.coords_x == piece.correct_coords_x and piece.coords_y == piece.correct_coords_y:
                pieceScore += PuzzleEnvironment.CORRECT_PLACEMENT_SCORE
                
            score += pieceScore

        return score
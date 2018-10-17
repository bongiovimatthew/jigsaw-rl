
import numpy as np
from enum import Enum

from Environment.direction import Direction 
from Environment.edge import EdgeShape 
from Environment.puzzleFactory import PuzzleFactory
from PIL import Image
from celery.contrib import rdb
from scipy.misc import imsave

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
 
    ACTION_TRANS_UP = 0
    ACTION_TRANS_RIGHT = 1
    ACTION_TRANS_DOWN = 2
    ACTION_TRANS_LEFT = 3
    ACTION_ROT90_1 = 4
    ACTION_ROT90_2 = 6
    ACTION_ROT90_3 = 7
    ACTION_CYCLE = 10 

class PuzzleEnvironment(Environment):
    CORRECT_IMAGE_SCORE = 4
    INCORRECT_OVERLAY_SCORE = -200
    CORRECT_GEOMMETRY_SCORE = 1
    INCORRECT_GEOMMETRY_SCORE = -2
    NOT_CONNECTED_SCORE = -1
    CORRECT_PLACEMENT_SCORE = 100

    # actions: 
    # 0 - cycle selected piece 
    # 1 - rotate 90 counterclockwise once 
    # 2 - rot90 cc twice 
    # 3 - rot90 cc three times 
    # 4 - translate up 
    # 5 - translate right 
    # 6 - translate down 
    # 7 - translate left 

    MAX_ACTIONS_NUM = 5

    def __init__(self):
        self.oldScore = 0
        self.debugMode = False
        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))

        # Generate the puzzle 
        factory = PuzzleFactory()
        puzzle = factory.generatePuzzle('images\\green_square.jpg', 2, 2)
        # self.pieceState = factory.createRandomPuzzlePieceArray(puzzle)
        self.pieceState = factory.getPuzzlePieceArray(puzzle)
        # pieceState is an array of PuzzlePiece objects
        self.puzzle = puzzle

        self.setupEnvironment()


    def reset(self):
        self.setupEnvironment()
        return self.render()

    def setupEnvironment(self):
        # Contains the relative position of the piece IDs in the current state 
        allocations = PuzzleFactory.getRandomAllocationOnlyOnePiece(self.puzzle,self.pieceState)

        self.guidArray = PuzzleFactory.placePiecesOnBoard(self.puzzle, self.pieceState,allocations)

        self.currentPieceIndex = len(self.pieceState)-1  # only last piece gets to move
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

    def _translate_piece(self, pieceId, direction,overlapAllowed=True): 
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
                    if overlapAllowed:
                        self.apply_translate(pieceId,x,y,newX,newY)
                    
                    else:
                        if len(self.guidArray[newY][newX]) == 0: 
                            self.apply_translate(pieceId,x,y,newX,newY)
                        else:
                            self.add_overlapp_score(pieceId)

                    return
    def apply_translate(self,pieceId,x,y,newX,newY):
        self.guidArray[y][x].remove(pieceId)
        self.guidArray[newY][newX].append(pieceId)
                                # Update pieceState 
        for piece in self.pieceState: 
            if piece.id == pieceId: 
                piece.coords_x = newX
                piece.coords_y = newY
                piece.overlappedScore = 0 
    def add_overlapp_score(self,pieceId):
        for piece in self.pieceState:
            if piece.id == pieceId:
                piece.overlappedScore += PuzzleEnvironment.INCORRECT_OVERLAY_SCORE
    def _convert_state(self, action):

        # if action == Actions.ACTION_CYCLE.value: 
        #     self.currentPieceIndex = (self.currentPieceIndex + 1) % len(self.pieceState)
        if action >= Actions.ACTION_ROT90_1.value and action <= Actions.ACTION_ROT90_3.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            numRotations = action-3
            self._rotate_piece(currentPiece.id, numRotations) 
        
        if action >= Actions.ACTION_TRANS_UP.value and action <= Actions.ACTION_TRANS_LEFT.value:
            currentPiece = self.pieceState[self.currentPieceIndex]
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            direction = directions[action]
            self._translate_piece(currentPiece.id, direction,overlapAllowed=False)

        return self.render()

    def saveState(self,fileName):

        img = Image.fromarray(self.render(), 'RGB')
        img.save(fileName)


    def step(self, action):
        self.stepCount += 1
        next_state = self._convert_state(action)
        currentScore = self.getScoreOfCurrentState()
        # rdb.set_trace()

        done = self.isMaxReward(currentScore) or (self.stepCount > 1000)

        tempOldScore = self.oldScore
        self.oldScore = currentScore

        reward = currentScore - tempOldScore
        if self.isMaxReward(currentScore):
            reward *= 100

        info = {'score':currentScore, 'oldScore': tempOldScore, 'action': action, 'step': self.stepCount}

        if (self.debugMode):
            print("Current Reward: {0}, IsDone: {1}, currentScore: {2}, oldScore: {3}".format(reward, done, currentScore, tempOldScore))
            print("Performing Action: {0}".format(Actions(action)))

        if (done):
            print("COMPLETED EPISODE!, reward:{0} currentScore:{1}".format(reward, currentScore))
            img = Image.fromarray(self.render(), 'RGB')
            img.save(r'files/state_images/episode%d.jpg'%self.stepCount)

        # if (action == Actions.ACTION_CYCLE.value):
        #     reward = -1            

        return (next_state, reward, done, info)
        
    def render(self, mode=None):
        boardCopy = self.puzzle.puzzleBoard.copy()
        piece1 = self.pieceState[0]
        count = 0

        for piece in self.pieceState: 

            baseY = piece.coords_y * piece.imgData.shape[0]
            yHeight = piece.imgData.shape[0]

            baseX = piece.coords_x * piece.imgData.shape[1]
            xWidth = piece.imgData.shape[1]
            boardCopy[ baseY : baseY + yHeight, baseX : baseX + xWidth] = piece.imgData.copy()

            if self.currentPieceIndex == count: 
                # Add a green bar on the current piece 
                greenSquareW = 5
                greenSquareH = 5
                boardCopy[ baseY : baseY + greenSquareH, baseX : baseX + greenSquareW] = [0, 255, 0]                
                boardCopy[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX : baseX + greenSquareW] = [0, 255, 0]                
                boardCopy[ baseY : baseY + greenSquareH, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
                boardCopy[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
            count += 1
        if (self.debugMode):
            print("piece.guid:{0}, piece.coords_x:{1}, piece.coords_y:{2}".format(piece.id, piece.coords_x, piece.coords_y))

            
        return boardCopy
    def indicateCurrentPieceBy(self,board,indication,piece):
        baseX,xWidth,baseY,yHeight = piece.getCoordinates()
        if indication == 'green-square':
            greenSquareW = 5
            greenSquareH = 5
            board[ baseY : baseY + greenSquareH, baseX : baseX + greenSquareW] = [0, 255, 0]                
            board[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX : baseX + greenSquareW] = [0, 255, 0]                
            board[ baseY : baseY + greenSquareH, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
            board[ baseY + yHeight - greenSquareH : baseY + yHeight, baseX + xWidth - greenSquareW : baseX + xWidth] = [0, 255, 0]                
        # elif indication == 'whiten':
            



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
            pieceScore += piece.overlappedScore 
            # if piece.overlappedScore < 0 :
            #     rdb.set_trace()
            if piece.coords_x == piece.correct_coords_x and piece.coords_y == piece.correct_coords_y:
                pieceScore += PuzzleEnvironment.CORRECT_PLACEMENT_SCORE
                
            score += pieceScore

        # normalizedScore = self.getNormalizedScore(score)
        #print("normalized Score %f"%normalizedScore)
        return score

    def getNormalizedScore(self,score):
        minScore = (len(self.pieceState) -1 ) * PuzzleEnvironment.INCORRECT_OVERLAY_SCORE  # (n-1) pieces on top of each others
        minScore = minScore + len(self.pieceState) * 4* (PuzzleEnvironment.INCORRECT_GEOMMETRY_SCORE ) # all sides are wrong

        maxScore = len(self.pieceState) * PuzzleEnvironment.CORRECT_PLACEMENT_SCORE   # (n) correctly placed peices
        maxScore  = maxScore + len(self.pieceState) * 4 * ( PuzzleEnvironment.CORRECT_GEOMMETRY_SCORE + PuzzleEnvironment.CORRECT_IMAGE_SCORE)
        normalizedScore = (score - minScore)/(maxScore - minScore) 
        return normalizedScore

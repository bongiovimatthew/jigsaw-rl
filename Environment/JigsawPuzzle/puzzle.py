from PIL import Image
import numpy as np


class Puzzle:

    def __init__(self, xNumPieces, yNumPieces):

        self.xNumPieces = xNumPieces
        self.yNumPieces = yNumPieces

        # These are built out by generatePuzzle
        self.piecesArray = None
        self.puzzleBoard = None
        self.singlePieceWidth = None
        self.singlePieceHeight = None

    def getCorrectPuzzleArray(self):
        return self.piecesArray

    #
    # Puzzle Display Functions
    #

    def getPiecesAsSingleLineImage(self, statePieces):
        finalImagePieces = None

        for piece in statePieces:
            if finalImagePieces is not None:
                finalImagePieces = np.concatenate((finalImagePieces, piece.imgData), 0)
            else:
                finalImagePieces = piece.imgData

        return finalImagePieces

    def getPiecesAsOneBigImage(self):
        finalImagePieces = None
        for x in range(self.xNumPieces):
            singlePuzzleCol = None
            for y in range(self.yNumPieces):
                if singlePuzzleCol is not None:
                    singlePuzzleCol = np.concatenate(
                        (singlePuzzleCol, self.piecesArray[y][x].imgData), 0)
                else:
                    singlePuzzleCol = self.piecesArray[y][x].imgData

            if finalImagePieces is not None:
                finalImagePieces = np.concatenate((finalImagePieces, singlePuzzleCol), 1)
            else:
                finalImagePieces = singlePuzzleCol

        return finalImagePieces

    def displayPuzzlePieces(self, displayType, pieces):

        if displayType == 'board':
            finalImagePieces = self.getPiecesAsOneBigImage()
            imgDisp = Image.fromarray(finalImagePieces, 'RGB')
            imgDisp.show()

        if displayType == 'line':
            finalImage = self.getPiecesAsSingleLineImage(pieces)
            imgDisp = Image.fromarray(finalImage, 'RGB')
            imgDisp.show()

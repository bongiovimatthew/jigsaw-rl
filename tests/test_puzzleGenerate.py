from Environment.puzzleFactory import PuzzleFactory


class PuzzleGenerateTests:

    def TestGenerateAndDisplayPuzzle():
        factory = PuzzleFactory()
        puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
        puzzle.displayPuzzlePieces('board', None)

    def TestDisplayPuzzlePiece():
        factory = PuzzleFactory()
        puzzle = factory.generatePuzzle('images\\rainier.jpg', 4, 4)
        puzzle.piecesArray[0][0].displayPiece()

    def TestPuzzlePieceRotate():
        factory = PuzzleFactory()
        puzzle = factory.generatePuzzle('images\\rainier.jpg', 4, 4)

        puzzle.piecesArray[0][0].displayPiece()
        puzzle.piecesArray[0][0].rotate()
        puzzle.piecesArray[0][0].displayPiece()
        print("Piece Geometry: ")
        print(puzzle.piecesArray[0][0].edgeGeometry)

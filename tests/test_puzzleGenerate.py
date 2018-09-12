from Environment.JigsawPuzzle.puzzleFactory import PuzzleFactory
from pathlib import Path

class PuzzleGenerateTests:

    def TestGenerateAndDisplayPuzzle():
        factory = PuzzleFactory()

        img_folder = Path("images/")
        puzzle_img_path = img_folder / "rainier_small.jpg"

        puzzle = factory.generatePuzzle(puzzle_img_path, 3, 3)
        puzzle.displayPuzzlePieces('board', None)

    def TestDisplayPuzzlePiece():
        factory = PuzzleFactory()

        img_folder = Path("images/")
        puzzle_img_path = img_folder / "rainier_small.jpg"

        puzzle = factory.generatePuzzle(puzzle_img_path, 4, 4)
        puzzle.piecesArray[0][0].displayPiece()

    def TestPuzzlePieceRotate():
        factory = PuzzleFactory()
        
        img_folder = Path("images/")
        puzzle_img_path = img_folder / "rainier_small.jpg"

        puzzle = factory.generatePuzzle(puzzle_img_path, 4, 4)

        puzzle.piecesArray[0][0].displayPiece()
        puzzle.piecesArray[0][0].rotate()
        puzzle.piecesArray[0][0].displayPiece()
        print("Piece Geometry: ")
        print(puzzle.piecesArray[0][0].edgeGeometry)

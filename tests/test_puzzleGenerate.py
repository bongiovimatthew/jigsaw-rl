from Environment.JigsawPuzzle.puzzleFactory import PuzzleFactory
from Environment.JigsawPuzzle.puzzleEnvironment import PuzzleEnvironment
from pathlib import Path
from PIL import Image

class PuzzleGenerateTests:

    def TestPuzzleEnvironment(): 
        paths = ['full_2_2.json', 'single_piece_3_3.json', 'single_piece_3_3_cycle.json', 'single_piece_3_3_random_cycle.json']
        env = PuzzleEnvironment(paths[2])
        state = env.render()
        img = Image.fromarray(state, 'RGB')
        img.show()

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

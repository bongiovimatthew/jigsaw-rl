from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 

def TestGenerateAndDisplayPuzzle():
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 4, 4)
	puzzle.displayPuzzlePieces('board', None)

def TestPuzzlePieceRotate(): 
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 4, 4)

	puzzle.piecesArray[0][0].displayPiece()
	puzzle.piecesArray[0][0].rotate()
	puzzle.piecesArray[0][0].displayPiece()
	print("Piece Geometry: ")
	print(puzzle.piecesArray[0][0].edgeGeometry)

def main():
	TestGenerateAndDisplayPuzzle()
	# TestPuzzlePieceRotate()

if __name__ == "__main__":
    main()
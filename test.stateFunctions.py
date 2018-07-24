from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 
from env import PuzzleEnvironment
from PIL import Image

def DrawPiecesInArrayOrder(initialPieceState, puzzle):
	puzzle.displayPuzzlePieces('line', initialPieceState)
	


def main():
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
	initialPieceState, guidArray = factory.createRandomPuzzlePieceArray(puzzle)

	env = PuzzleEnvironment(puzzle, initialPieceState, guidArray)
	img = Image.fromarray(env.render(), 'RGB')
	img.show()

	print("Current State Score: {0}".format(env.getScoreOfCurrentState()))
	DrawPiecesInArrayOrder(initialPieceState, puzzle)


if __name__ == "__main__":
    main()
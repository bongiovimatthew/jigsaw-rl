from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 
from env import State
from PIL import Image

def main():
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
	initialPieceState = factory.createRandomPuzzlePieceArray(puzzle)

	state = State(puzzle, initialPieceState)
	img = Image.fromarray(state.render(), 'RGB')
	img.show()

	print("Current State Score: %s" % state.getScoreOfCurrentState())

if __name__ == "__main__":
    main()
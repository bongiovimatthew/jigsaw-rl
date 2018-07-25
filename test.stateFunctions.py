from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 
from env import PuzzleEnvironment
from PIL import Image
import random 

def DrawPiecesInArrayOrder(initialPieceState, puzzle):
	puzzle.displayPuzzlePieces('line', initialPieceState)
	
def TestInitialScoring(): 
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
	initialPieceState, guidArray = factory.createRandomPuzzlePieceArray(puzzle)

	env = PuzzleEnvironment(puzzle, initialPieceState, guidArray)
	img = Image.fromarray(env.render(), 'RGB')
	img.show()

	print("Current State Score: {0}".format(env.getScoreOfCurrentState()))
	DrawPiecesInArrayOrder(initialPieceState, puzzle)


def TestActions(): 
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier_small.jpg', 3, 3)
	initialPieceState, guidArray = factory.createRandomPuzzlePieceArray(puzzle)

	env = PuzzleEnvironment()

	img = Image.fromarray(env.render(), 'RGB')
	img.show()

	# actions: 
	ACTION_CYCLE = 0 
	ACTION_ROT90_1 = 1
	ACTION_ROT90_2 = 2
	ACTION_ROT90_3 = 3
	ACTION_TRANS_UP = 4
	ACTION_TRANS_RIGHT = 5
	ACTION_TRANS_DOWN = 6
	ACTION_TRANS_LEFT = 7

	# for i in range(200): 
	# 	randomAction = random.randint(0, 7)
	# 	env.step(randomAction)

	env.step(ACTION_CYCLE)
	#env.step(ACTION_TRANS_DOWN)

	img = Image.fromarray(env.render(), 'RGB')
	img.show()

def main():
	#TestInitialScoring()
	TestActions()
	
if __name__ == "__main__":
    main()
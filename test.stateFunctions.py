from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 
from env import State
from PIL import Image

def main():
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
	
	state = State(puzzle)
	img = Image.fromarray(state.getStateImage(), 'RGB')
	img.show()

if __name__ == "__main__":
    main()
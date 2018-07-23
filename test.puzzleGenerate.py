from puzzle import Puzzle
from puzzleFactory import PuzzleFactory 

def main():
	factory = PuzzleFactory()
	puzzle = factory.generatePuzzle('images\\rainier.jpg', 4, 4)
	puzzle.displayPuzzlePieces()

if __name__ == "__main__":
    main()
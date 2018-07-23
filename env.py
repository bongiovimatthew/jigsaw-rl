class State:
	
	def __init__(self):

		puzzle = Puzzle('zambia_map.jpg',4, 4)
		puzzle.generatePuzzle()
		self.pieceState = puzzle.createRandomPuzzlePieceArray()
		self.board = puzzle.createPuzzleBoard()
		

	def getStateImage(self):
		boardCopy = copy(self.board)
		for piece in self.pieceState: 
			boardCopy[piece.coords_y:piece.coords_y + piece.imgData.shape[0]][piece.coords_x: piece.coords_x + piece.imgData.shape[1]] = piece.imgData 

		return boardCopy



def main():
	

	return

if __name__ == "__main__":
    main()
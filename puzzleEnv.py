from enum import Enum

class EdgeShape(enum):
	STRAIGHT = 0
	IN = 1
	OUT = 2

	def GetComplement(edgeShape):
		if (edgeShape == STRAIGHT):
			return STRAIGHT
		if (edgeShape == IN):
			return OUT

		return IN

class PuzzlePiece:
	
	def init(self, imgData, coords):
		self.imgData = imgData # tbd
		self.coords = coords
		self.edgeGeometry = 0 # L, D, R, U (2 bits per direction totals 1 byte)

	def getEdgeGeometry(self, direction):
		if direction == UP:
			return EdgeShape(int(bin(edgeGeometry)[2:][6:8]), 2)
		else if direction == RIGHT:
			return EdgeShape(int(bin(edgeGeometry)[2:][4:6]), 2)
		else if direction == DOWN:
			return EdgeShape(int(bin(edgeGeometry)[2:][2:4]), 2)
		else if direction == LEFT:
			return EdgeShape(int(bin(edgeGeometry)[2:][0:2]), 2)

	def setEdgeGeometry(self, direction, edgeShape):
		if direction == UP:
			edgeGeometry = edgeGeometry | (edgeShape)
		else if direction == RIGHT:
			edgeGeometry = edgeGeometry | (edgeShape << 2)
		else if direction == DOWN:
			edgeGeometry = edgeGeometry | (edgeShape << 4)
		else if direction == LEFT:
			edgeGeometry = edgeGeometry | (edgeShape << 6)

class Direction(enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def GetComplement(direction):
		if (direction == UP):
			direction = DOWN
		if (direction == DOWN):
			direction = UP
		if (direction == RIGHT):
			direction = LEFT
		if (direction == LEFT):
			direction = RIGHT

class Puzzle:
	piecesArray = [][]

	def createEdge(piece0Coords, piece1Coords, p0EdgeDirection):
		if (piecesArray[piece0Coords].getEdgeGeometry(p0EdgeDirection) != EdgeShape.STRAIGHT):
			return

		p1EdgeDirection = Direction.GetComplement(p0EdgeDirection)

		p0EdgeShape = EdgeShape(random.rand(1,3))
		p1EdgeShape = EdgeShape.GetComplement(p0EdgeShape)

		piecesArray[piece0Coords].setEdgeGeometry(p0EdgeDirection, p0EdgeShape)
		piecesArray[piece1Coords].setEdgeGeometry(p1EdgeDirection, p1EdgeShape)

	def generatePuzzleGeometry(puzzleWidth, puzzleHeight):
		self.piecesArray = [puzzleWidth][puzzleHeight]
		for x in puzzleWidth:
			for y in puzzleHeight:
				piecesArray[x, y].coords = (x, y)
				if (x < (puzzleWidth - 1)):
					createEdge((x,y), (x+1,y), Direction.RIGHT)

				if (x > 0):
					createEdge((x,y), (x-1,y), Direction.LEFT)

				if (y < (puzzleWidth - 1)):
					createEdge((x,y), (x,y+1), Direction.DOWN)
				if (y > 0):
					createEdge((x,y), (x,y-1), Direction.UP)



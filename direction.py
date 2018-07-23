from enum import Enum

class Direction(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def GetComplement(direction):
		if (direction == Direction.UP):
			direction = Direction.DOWN
		elif (direction == Direction.DOWN):
			direction = Direction.UP
		elif (direction == Direction.RIGHT):
			direction = Direction.LEFT
		elif (direction == Direction.LEFT):
			direction = Direction.RIGHT

		return direction

	def GetAllDirections():
		return [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

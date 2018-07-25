from enum import Enum

class EdgeShape(Enum):
	STRAIGHT = 0
	IN = 1
	OUT = 2

	def GetComplement(edgeShape):
		if (edgeShape == EdgeShape.STRAIGHT):
			return EdgeShape.STRAIGHT
		if (edgeShape == EdgeShape.IN):
			return EdgeShape.OUT

		return EdgeShape.IN


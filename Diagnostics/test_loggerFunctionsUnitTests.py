import logger 
from .env import PuzzleEnvironment

class LoggerFunctionUnitTests: 
	def TestLoggerImageWrite(): 
		env = PuzzleEnvironment()
		data = env.render()
		logger.log_state_image(data)

		return 
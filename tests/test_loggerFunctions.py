from Diagnostics.logger import Logger 
from Environment.env import PuzzleEnvironment

class LoggerFunctionUnitTests:

	def TestLoggerImageWrite(): 
		env = PuzzleEnvironment()
		data = env.render()
		Logger.log_state_image(data, 0, 0, -1, (224, 224))

		return 
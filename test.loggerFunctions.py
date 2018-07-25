import logger 
from env import PuzzleEnvironment

def TestLoggerImageWrite(): 
	env = PuzzleEnvironment()
	data = env.render()
	logger.log_state_image(data)

	return 

def main():
	TestLoggerImageWrite()
	
if __name__ == "__main__":
    main()
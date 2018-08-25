from tests.test_puzzleGenerate import PuzzleGenerateTests
from tests.test_loggerFunctions import LoggerFunctionUnitTests

def main():
    PuzzleGenerateTests.TestGenerateAndDisplayPuzzle()
    LoggerFunctionUnitTests.TestLoggerImageWrite()
    return
    
if __name__ == "__main__":
    main()
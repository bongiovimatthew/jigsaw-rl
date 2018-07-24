from experiment import Experiment 
from env import PuzzleEnvironment
from qLearningAgent import QLearningAgent 

def main():
    # Generate the puzzle 
    factory = PuzzleFactory()
    puzzle = factory.generatePuzzle('images\\rainier.jpg', 3, 3)
    initialPieceState, guidArray = factory.createRandomPuzzlePieceArray(puzzle)

    env = PuzzleEnvironment(puzzle, initialPieceState, guidArray)

    interactive = True
    agent = QLearningAgent(range(env.action_space.n))
    experiment = Experiment(env, agent)
    experiment.run_qlearning(10, interactive)

if __name__ == "__main__":
    main()
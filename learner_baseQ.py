from experiment import Experiment 
from env import PuzzleEnvironment
from qLearningAgent import QLearningAgent 

def main():

    env = PuzzleEnvironment()

    interactive = True
    agent = QLearningAgent(range(env.action_space.n))
    experiment = Experiment(env, agent)
    experiment.run_qlearning(1, interactive)

if __name__ == "__main__":
    main()
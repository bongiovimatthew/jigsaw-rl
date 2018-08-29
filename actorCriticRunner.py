from Learners.ActorCriticLearner import ActorCriticLearner as lrn
from Environment.puzzleEnvironment import PuzzleEnvironment
from Environment.snakeEnvironment import SnakeEnvironment
import argparse

parser = argparse.ArgumentParser(description='Actor Critic Learning')
parser.add_argument('--gamma', type=float, default=0.75, metavar='F', help='the discounting factor (default:0.99)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='F', help='the learning rate (default:0.00025)')
parser.add_argument('--game-length', type=int, default='10000', metavar='N', help='assumed maximal length of an episode (deafult:10000)')
parser.add_argument('--T-max', type=int, default=1200, metavar='N', help='the length of the training (default:120)')
parser.add_argument('--batch-length', type=int, default=4, metavar='N', help='the length of the training batch (default:4)')
args = parser.parse_args()

#env = PuzzleEnvironment()
env = SnakeEnvironment()
lrn.execute_agent(env, args.batch_length, args.game_length, args.T_max, args.gamma, args.lr)
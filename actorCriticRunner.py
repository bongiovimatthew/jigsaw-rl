from Learners.ActorCriticLearner import ActorCriticLearner as lrn
from Environment.JigsawPuzzle.puzzleEnvironment import PuzzleEnvironment
from Environment.Snake.snakeEnvironment import SnakeEnvironment
import argparse
import time

parser = argparse.ArgumentParser(description='Actor Critic Learning')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F',
                    help='the discounting factor (default:0.99)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='F',
                    help='the learning rate (default:0.00025)')
parser.add_argument('--game-length', type=int, default='10000', metavar='N',
                    help='assumed maximal length of an episode (deafult:10000)')
parser.add_argument('--T-max', type=int, default=1500, metavar='N',
                    help='the length of the training (default:120)')
parser.add_argument('--batch-length', type=int, default=4, metavar='N',
                    help='the length of the training batch (default:4)')
parser.add_argument('--config', type=str, default='single_piece_3_3.json', metavar='N',
                    help='the initializiation config to use for the environment (only Puzzle supported)')

parser.add_argument('--load-model', dest='load_model', action='store_true',
                    help='use existing model')
parser.add_argument('--dont-load-model', dest='load_model', action='store_false',
                    help='dont use existing model')
parser.set_defaults(load_model=True)

parser.add_argument('--evaluate-mode', dest='evaluate_mode', action='store_true',
                    help='evaluate stored model')
parser.add_argument('--training-mode', dest='evaluate_mode', action='store_false',
                    help='training mode')
parser.set_defaults(evaluate_mode=False)

args = parser.parse_args()

#env = PuzzleEnvironment(args.config)
env = SnakeEnvironment()

start_time = time.time()
lrn.execute_agent(env, args.batch_length, args.game_length, args.T_max, args.gamma, args.lr, args.load_model, args.evaluate_mode)
elapsed_time = time.time() - start_time
print("Time: {0}".format(elapsed_time))

from HumanGame.humanAgent import HumanAgent 
from HumanGame.humanSnakeAgent import HumanSnakeAgent
import argparse

parser = argparse.ArgumentParser(description='Actor Critic Learning')
parser.add_argument('--env', type=int, default=0, metavar='N', help='the environment. 0: puzzle, 1: snake')
args = parser.parse_args()

game = None
if args.env > 0: 
    game = HumanSnakeAgent()
else:
    game = HumanAgent()

game.run()
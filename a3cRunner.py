from multiprocessing import Lock, Pool
import argparse

from A3C import learner as lrn
from Diagnostics.logger import Logger as logger

# Read the parameters to initialize the algorithm.
parser = argparse.ArgumentParser(description='A3C algorithm')

parser.add_argument('--atari-env', default='Breakout-v0', metavar='S',
        help='the name of the Atari environment (default:Breakout-v0)')
parser.add_argument('--num-cores', type=int, default=1, metavar='N',
        help='the number of cores should be exploited (default:2)')
parser.add_argument('--t-max', type=int, default=15, metavar='N',
        help='update frequency (default:5)')
parser.add_argument('--game-length', type=int, default='100000', metavar='N',
        help='assumed maximal length of an episode (deafult:10000)')
parser.add_argument('--T-max', type=int, default=100000, metavar='N',
        help='the length of the training (default:120)')
parser.add_argument('--epoch-size', type=int, default=1000, metavar='N',
        help='the frequency of evaluation during training (default:25)')
parser.add_argument('--eval-num', type=int, default=0, metavar='N',
        help='the number of evaluations in an evaluation session (default:10)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F',
        help='the discounting factor (default:0.99)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='F',
        help='the learning rate (default:0.00025)')
parser.add_argument('--moving-pieces-count', type=int, default=1, metavar='F',
        help='number of pieces that are dislocated in the puzzle. default is all. ')
parser.add_argument('--display', type=int, default=0, metavar='F',
        help='display the puzzle while being trained')
parser.add_argument('--test-mode', action='store_true',
        help='training or evaluation')

args = parser.parse_args()

# IF TRAIN mode -> train the learners
def set_args_for_test(args):
    args.T_max = 0 ; args.t_max = 50000; args.game_length= 200
    args.epoch_size = 0 ; args.eval_num=0; args.gamma = 0;args.num_cores=1
    return args
def executable(process_id):
    lrn.execute_agent(process_id, args)

if not args.test_mode:
    
    print ('Training mode.')
    
        
    # start the processes
    if __name__ == '__main__':
        
        n = args.num_cores
        l = Lock()

        sh = lrn.create_shared(args.moving_pieces_count) # list with two lists inside
                                               # contains the parameters as numpy arrays
        
        pool = Pool(n, initializer = lrn.init_lock_shared, initargs = (l,sh,))
        logger.create_folders(l,args)
        idcs = [0] * n
        for p in range(0, n):
            idcs[p] = p
  
        pool.map(executable, idcs)
            
        pool.close()
        pool.join()
        
        # for_saving = lrn.create_agent(args)
        # logger.save_model(for_saving, sh)
         
# IF EVALUATION mode -> evaluate a test run (rewards, video)
else:
    
    print ('Evaluation mode.')
    args = set_args_for_test(args)
    ag = lrn.create_agent_for_evaluation(args)
    ag.evaluate()    


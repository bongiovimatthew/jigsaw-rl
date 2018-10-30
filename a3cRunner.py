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

max_episode_length = args.game_length
gamma = args.gamma # discount rate for advantage estimation and reward discounting
s_size = 76800 # Observations are greyscale frames of 160 * 160 * 3
a_size = 5 # Agent can move Left, Right, Up, Down, or NOOP
load_model = False
model_path = './model'

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)


start_time = time.time()
lrn.execute_agent(env, args.batch_length, args.game_length, args.T_max, args.gamma, args.lr, args.load_model, args.evaluate_mode)
elapsed_time = time.time() - start_time
print("Time: {0}".format(elapsed_time))

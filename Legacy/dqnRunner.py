from Environment.env import PuzzleEnvironment
from QLearning.dqn import DeepQAgent
from PIL import Image
from argparse import ArgumentParser

def as_ale_input(environment):
    """Convert the Atari environment RGB output (210, 160, 3) to an ALE one (84, 84).
    We first convert the image to a gray scale image, and resize it.
    Attributes:
        environment (Tensor[input_shape]): Environment to be converted
    Returns:
         Tensor[84, 84] : Environment converted
    """
    return np.array(Image.fromarray(environment).convert('L').resize((84, 84)))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of epochs to run (epoch = 250k actions')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Flag for enabling Tensorboard')

    args = parser.parse_args()

    # 1. Make environment:
    env = PuzzleEnvironment()

    # 2. Make agent
    agent = DeepQAgent((1, 84, 84), env.action_space.n, monitor=args.plot)

    # Train
    current_step = 0
    max_steps = args.epoch * 250000
    current_state = as_ale_input(env.reset())

    while current_step < max_steps:
        action = agent.act(current_state)
        new_state, reward, done, _ = env.step(action)
        new_state = as_ale_input(new_state)

        # Clipping reward for training stability
        reward = np.clip(reward, -1, 1)

        agent.observe(current_state, action, reward, done)
        agent.train()

        current_state = new_state

        if done:
            current_state = as_ale_input(env.reset())

        current_step += 1
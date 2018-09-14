# Reinforcement Learning - Multiple Games

## Overview 

This project contains an implementation of a jigsaw puzzle game, a snake game, and multiple learning agents which learn to play both games.

## Jigsaw Game 

The puzzle game takes an image and generates a simple jigsaw puzzle, similar to below. The jigsaw puzzle is then randomly initialized onto a playing surface, at which point, our learning agent can interact with it. 

![Puzzle Sliced](./images/docs/puzzle_sliced.png =250x250)

![Puzzle Initialized](./images/docs/puzzle_shuffled.PNG =250x250)


## Running the Code 

Before you run the code, you must install all dependencies: 

`pipenv install -e .` 

Once the dependencies are installed, you should run the virtual environment with: 

`pipenv shell`



Once the virtual environment is up and running, you can begin the actor critic learning agents by running the following: 

`python actorCriticRunner.py` 

This will run the A2C learning agent against the Jigsaw Puzzle game for `--T-max` training games, updating the NN as it learns. 

If you wish to run the agent against the Snake game instead of the Jigsaw Puzzle, you should adjust the environment definition in `actorCriticRunner.py`. 


## Human Testing

If you want to run the Jigsaw Puzzle game in human-playable form, you can run: 

`python humanGameRunner.py` 


If you want to run the Snake game in human-playable form, you can run: 

`python .\Environment\Snake\snake.py`

Note that the snake game currently runs the pygame implementation, and not the pillow implementation. This means that the human version of the game does not properly test the snake environment. If you wish to test against the true Snake environment, you should run the unit tests. 


## Unit Tests 

There are a handful of unit tests you can use to validate environment behavior. You should adjust `unittestsRunner.py` to execute the unit tests you wish to run. 


## Visualizing the Model with TensorBoard 

You can use TensorBoard to visualize the model training progress, and monitor several health metrics. 

To do this, run the following: 

    tensorboard --logdir=tf_train  http://localhost:6006/

Then, navigate to http://localhost:6006/


## Package Overview 

jigsaw-rl
.
+-- ActionChoosers: contains the action choosers responsible for selecting the action an agent should take 
+-- Brain: the neural network logic 
+-- Diagnostics: logging, metrics, etc. 
+-- Environment
|   +-- JigsawPuzzle: the Jigsaw Puzzle game 
|   +-- Snake: the Snake game 
+-- HumanGame: runs the Jigsaw Puzzle environment so that a human can play (instead of a learning agent)
+-- Learners: the learning agents
+-- tests: unit tests 
+-- actorCriticRunner.py
+-- humanGameRunner.py
+-- unittestsRunner.py


## How the Game Works 


## How the Learner Works 


## Further Reading 

* [A3C Learning against Atari games](https://arxiv.org/pdf/1602.01783v1.pdf)


* [Deep Q Learning against Atari games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


flake8 --ignore="E265,E501"
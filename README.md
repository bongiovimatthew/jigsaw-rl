# Reinforcement Learning - Jigsaw Puzzle 

## Overview 

This project contains an implementation of a jigsaw puzzle game, along with a learning agent. 
The puzzle game takes an image and generates a simple jigsaw puzzle, similar to below. The jigsaw puzzle is then randomly initialized onto a playing surface, at which point, our learning agent can interact with it. 

![Puzzle Sliced](./images/docs/puzzle_sliced.png =250x250)

![Puzzle Initialized](./images/docs/puzzle_shuffled.PNG =250x250)

## Running the Code 

You will need to install a handful of dependencies before running this project. 


* __CNTK__: Used to perform NN operations, maintain state, etc. 

	`pip install cntk`

* __Python Imaging Library__: Used for generating and manipulating puzzle pieces from an image 

	`pip install pillow `

* __Numpy__: Used for numerous array and mathematical operations 

    `pip install numpy `

* __SciPy__: Used for 

    `pip install scipy `


* __Pynput__: Used for controlling the human version of the puzzle game 

	`pip install pynput`

* __Celery__: Used for debugging (setting breakpoints, analyzing memory, etc.)

    `pip install celery`



Once all the dependencies are installed, you can begin the different learning agents by running one of the following: 

`python a3cRunner.py` 

This will run the A3C learning agent against the Jigsaw Puzzle game for `--T-max` training games, updating the NN as it learns. After the training games, the model will be evaluated. 


`python dqnRunner.py` 

This will run the Deep Q learning agent


`python humanGameRunner.py` 

This will run the human-playable game that interacts with the puzzle environment the same way the learners do. 



## Package Overview 

jigsaw-rl/ 

   __A3C/__ - package containing the A3C learner and supporting Neural Network code 

   __Environment/__ - package containing the jigsaw puzzle environment that the learners interact with 
   
   __Diagnostics/__ - package containing diagnostics tools such as logging 

   __HumanGame/__ - package containing a human-playable game that interacts with the environment the same way the learners do 

   __QLearning/__ - package containing the Deep Q leaner and supporting Neural Network code



## How the Game Works 


## How the Learner Works 


## Further Reading 

* [A3C Learning against Atari games](https://arxiv.org/pdf/1602.01783v1.pdf)


* [Deep Q Learning against Atari games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## Visualizing the Model with TensorBoard 

You can use TensorBoard to visualize the model training progress, and monitor several health metrics. 

To do this, run the following: 

    tensorboard --logdir=log

Then, navigate to http://localhost:6006/
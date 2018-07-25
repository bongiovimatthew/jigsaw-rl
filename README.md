# Reinforcement Learning - Jigsaw Puzzle 

## Overview 

This project contains an implementation of a jigsaw puzzle game, along with a learning agent. 
The puzzle game takes an image and generates a simple jigsaw puzzle, similar to below. The jigsaw puzzle is then randomly initialized onto a playing surface, at which point, our learning agent can interact with it. 

![Puzzle Sliced](./images/docs/puzzle_sliced.png=250x250)

![Puzzle Initialized](./images/docs/puzzle_shuffled.png=250x250)

## Running the Code 

You will need to install a handful of dependencies before running this project. 

* __AI Gym__: Used for 

	`pip install gym`

* __CNTK__: Used to perform NN operations, maintain state, etc. 

	`pip install cntk`

* __Python Imaging Library__: Used for generating and manipulating puzzle pieces from an image 

	`pip install pillow `

* __Pynput__: Used for controlling the human version of the puzzle game 

	`pip install pynput`


Once all the dependencies are installed, you can begin the learning agent by running: 

`python a3c.py` 

This will run the learning agent against the Jigsaw Puzzle game for `--T-max` training games, updating the NN as it learns. After the training games, the model will be evaluated. 

## Package Overview 

jigsaw-rl/ 
	Environment/
	Diagnostics/
	Extras/
	images/
	QLearning/

## How the Game Works 


## How the Learner Works 
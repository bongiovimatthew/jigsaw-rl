import learner
from Diagnostics import logger
from Environment import PuzzleEnvironment
from PIL import Image


def TestImageProcessor():
    env = PuzzleEnvironment()
    obs = env.reset()

    # process_img(obs)
    imageFinal = learner.process_img(obs)

    #imgDisp = Image.fromarray(imageFinal, 'RGB')
    # imgDisp.show()

    logger.log_state_image(imageFinal)
    return


def main():
    TestImageProcessor()


if __name__ == "__main__":
    main()

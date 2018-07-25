import learner 
import logger 
from env import PuzzleEnvironment
from PIL import Image

# Remove after testing 
import numpy as np
import scipy.misc as sc

def process_img(observation):

    # Input: the raw frame as a list with shape (210, 160, 3)
    # https://gym.openai.com/envs/Breakout-v0 
    input_img = np.array(observation)

    imgDisp = Image.fromarray(input_img, 'RGB')
    imgDisp.show()
    
    # Reshape input to meet with CNTK expectations.
    img = np.reshape(input_img, (3, 342, 342))

    # Mapping from RGB to gray scale. (Shape remains unchanged.)
    # Y = (2*R + 5*G + 1*B)/8

    print(img.shape)

    img_gray = np.zeros((1, img.shape[1], img.shape[2]))
    
    print(img_gray.shape)

    img_gray = (0.0722*img[0,:,:] + 0.7152*img[1,:,:] + 0.2126*img[2,:,:])
    
    img2 = np.reshape(np.uint8(img_gray), (342, 342))

    imgDisp = Image.fromarray(img2, 'L')
    imgDisp.show()

    # Rescaling image to 1x84x84.
    img_cropped_gray_resized = np.zeros((1,84,84))
    img_cropped_gray_resized[0,:,:] = sc.imresize(img_gray, (1,84,84), interp='bilinear', mode=None)
    
    # Saving memory. Colors goes from 0 to 255.
    img_final = np.uint8(img_cropped_gray_resized)


    logger.log_state_image(img_final)

    return img_final





def TestImageProcessor(): 
	env = PuzzleEnvironment()
	obs = env.reset()
	
	process_img(obs)
	#imageFinal = learner.process_img(obs) 

	#imgDisp = Image.fromarray(imageFinal, 'RGB')
	#imgDisp.show()

	#logger.log_state_image(imageFinal)
	return 

def main():
	TestImageProcessor()
	
if __name__ == "__main__":
    main()
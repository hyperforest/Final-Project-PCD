# Import library
print('>>> Importing library...')
import numpy as np
import cv2
import os
import sys
from datetime import datetime

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
print('Done importing library.')

model, letter_list = '', ''
logo = '''
   _____           __  ___             _         __    ___  ________ 
  / __(_)__  ___ _/ / / _ \_______    (_)__ ____/ /_  / _ \/ ___/ _ \\
 / _// / _ \/ _ `/ / / ___/ __/ _ \  / / -_) __/ __/ / ___/ /__/ // /
/_/ /_/_//_/\_,_/_/ /_/  /_/  \___/_/ /\__/\__/\__/ /_/   \___/____/ 
                                 |___/                               
'''

### Function for DIP techniques

def log_transform(img, c = 1):
    
    '''
        @ Apply log transform to image

        @ Parameter: img, c (default 1)

        @ Each intensity I in the image will be transformed into
          I' = c * log10 (1 + I)
    '''

    return (c * np.log10(1 + img)).astype('uint8')

def inv_log_transform(img, c = 1, omega = 0):
    
    '''
        @ Apply inverse log transform to image

        @ Parameter: img, c (default 1), omega (default 0)

        @ Each intensity I in the image will be transformed into
          I' = c * (pow(I, omega))
    '''

    img = img / 255
    return (c * np.power(img, omega))
        
def contrast_stretching(img, a = 0, b = 255):

    '''
        @ Apply contrast-stretching to image

        @ Stretch the minimum intensity of image to a,
          stretch the maximum intensity of image to b,
          and stretch other intensity uniformly in range [a..b]

        @ Parameter: img (can be RGB or grayscale), a (default 0), b (default 255)

        @ Each intensity I in the image will be transformed into
          I' = (I - Min) * ((b - a) / (Max - Min)) + a
    '''

    Min = img.min()
    Max = img.max()
    
    return np.floor(((img.astype('float') - Min) * (b - a) / (Max - Min) + a)).astype('uint8')

def threshold(img, th):

    '''
        @ Threshold an grayscale image;
          all intensity >= threshold will be 1, else 0

        @ Parameter: img, th (threshold value)
    '''

    return (img >= th).astype('uint8') * 255

def downsample(image, interpolation = cv2.INTER_LINEAR):
    
    '''
        @ Downsample the image using pre-defined interpolation method

        @ Parameter : img, interpolation (default cv2.INTER_LINEAR)
          interpolation can be either cv2.INTER_LINEAR, cv2.INTER_NEAREST,
          cv2.INTER_CUBIC, or other cv2 valid interpolation

    '''
    
    return cv2.resize(image, (28, 28), interpolation = interpolation)

### Function for executing

def mine(image, path):

    '''
        @ Parameter : image, path (string)
        @ Save image to path (do nothing if the directory not exist)
        @ File name format : DD-MM-YY HH-MM-SS.JPG
    '''

    fn = path + '\\' + str(datetime.now())[:19].replace(':', '-') + '.JPG'
    cv2.imwrite(fn, image)

def get_model():

	'''
        @ Load weights from pre-trained model
    '''

	print('\n>>> Loading model...')

	num_classes = 24
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1)))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.20))
	model.add(Dense(num_classes, activation = 'softmax'))

	model.load_weights('model\\model_polos_w2.h5')
	print('Done loading model.\n')

	return model
        
def show_image(img, win_name = ''):
    cv2.imshow(win_name, img)

def info():
    print(logo)
    print("-- Press q to capture.")
    print("-- Hold z to exit.")
    print()

def result(ctr, letter, percentage):
    print('(+) Trial #              :', ctr)
    print('(+) Most possible letter : {}'.format(letter))
    print('(+) Confidency           : {:.2f}%'.format(100 * percentage))

def olah(image):
    image = contrast_stretching(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = downsample(image, cv2.INTER_LINEAR)
    return image

def predict(image):
    image = image / 255
    prediction_array = model.predict(image.reshape(1, 28, 28, 1))

    idx = np.argmax(prediction_array)
    letter = letter_list[idx]
    percentage = prediction_array[0][idx]
    
    return (letter, percentage)

def get_frame(image):
    # create black RGB image of 224x224 pixel
    frame = np.zeros((224, 224, 3), np.uint8)

    # make it green
    frame[:,:,1] = frame[:,:,1] + 255

    # create frame
    copy = image[480-224:, :224]
    frame[5:5+224-10, 5:5+224-10] = copy[5:5+224-10, 5:5+224-10]
    image[480-224:, :224] = frame

    return image

def main():
    global model, letter_list

    model = get_model()
    letter_list = [chr(i) for i in range(ord('A'), ord('Z'))]
    letter_list.pop(9)
    ctr = 1
    cap_asli = cv2.VideoCapture(0)
    
    info()

    while 1:
        _, img_asli = cap_asli.read()

        img_input = img_asli[480-224:, :224]
        cv2.imshow('Detect Here!', get_frame(img_asli.copy()))

        if cv2.waitKey(1) % 0xFF == ord('q'):
            os.system('cls')
            info()

            img_input = olah(img_input)
            letter, percentage = predict(img_input)

            result(ctr, letter, percentage)
            show_image(img_input, win_name = 'Predicted Image.')

            ctr += 1

            mine(img_input, 'mined\\small')
            if ctr % 5 == 0 and len(sys.argv) == 2:
                if sys.argv[1] == 'mine':
                    mine(img_asli, 'mined\\big')
        
        elif cv2.waitKey(1) & 0xFF == ord('z'):
            break

    cap_asli.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

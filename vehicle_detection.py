from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape


from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box
disable_existing_loggers=False

from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()


keras.backend.set_image_dim_ordering('th')

#YOLO Tiny Keras Implementation, thanks to Udacity Vehicle Detection Project


vehicle_model = Sequential()
vehicle_model.add(Convolution2D(16, (3, 3),input_shape=(3,448,448),padding='same',strides=(1,1)))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2)))
vehicle_model.add(Convolution2D(32,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vehicle_model.add(Convolution2D(64,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vehicle_model.add(Convolution2D(128,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vehicle_model.add(Convolution2D(256,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vehicle_model.add(Convolution2D(512,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vehicle_model.add(Convolution2D(1024,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(Convolution2D(1024,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(Convolution2D(1024,(3,3) ,padding='same'))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(Flatten())
vehicle_model.add(Dense(256))
vehicle_model.add(Dense(4096))
vehicle_model.add(LeakyReLU(alpha=0.1))
vehicle_model.add(Dense(1470))


vehicle_model.summary()
load_weights(vehicle_model,'./yolo-tiny.weights')


image = plt.imread(filename)
image_crop = image[300:650,500:,:]
resized = cv2.resize(image_crop,(448,448))
batch = np.transpose(resized,(2,0,1))
batch = 2*(batch/255.) - 1
batch = np.expand_dims(batch, axis=0)
out = vehicle_model.predict(batch)
boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.12)   #Parameter Tuning
f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
ax1.imshow(image)
ax2.imshow(draw_box(boxes,plt.imread(filename),[[500,1280],[300,650]]))
plt.show()
def frame_function(image):
    crop = image[300:650,500:,:]
    resized = cv2.resize(crop,(448,448))
    batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = vehicle_model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.12)  #Parameter Tuning
    return draw_box(boxes,image,[[500,1280],[300,650]])



project_video_output = './project_video_outputs.mp4'#Enter name of output video
clip_input = VideoFileClip("./project_video.mp4") #Enter name of input video



lane_clip = clip_input.fl_image(frame_function) #This function expects color images!!
lane_clip.write_videofile(project_video_output, audio=False)

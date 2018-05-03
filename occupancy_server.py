#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import numpy
import datetime
import os
import sys
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from functools import partial
import time
import io

import image_process

def predict_occupancy(graph,img):
    """ Returns a 1 if the room is predicted to be occupied, else 0 """

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img.astype( numpy.float16), 'user object' )
    # Get the results from NCS
    output, userobj = graph.GetResult()

    return numpy.argmax(output)

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len( devices ) == 0:
	print( 'No devices found' )
	quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device( devices[0] )
device.OpenDevice()

# Read the graph file into a buffer
with open( GRAPH_PATH, mode='rb' ) as f:
	blob = f.read()

# Load the graph buffer into the NCS
graph = device.AllocateGraph( blob )

# Camera Setup
camera = PiCamera()
time.sleep(0.2)
stream = io.BytesIO()
camera.resolution = (3280, 2464)

# Capture Rate Setup
now = datetime.datetime.now()
today10pm = now.replace(hour = 22, minute=0, second=0,microsecond=0)
lastCaptured = now
captureRate = 15

# Load the image mean
mean_img = numpy.load(HOME_PATH + '/data/image_mean.npy').mean(1).mean(1) 

while(True):
    timestamp = datetime.datetime.now()

    if timestamp > today10pm:
        sys.exit(0)
    if (timestamp - lastCaptured).seconds >= captureRate:
        lastCapured = timestamp
        print("CAPTURING IMAGE: ")
        camera.capture(stream, format="jpeg")
        data = numpy.fromstring(stream.getvalue(),dtype=numpy.uint8)
        image = cv2.imdecode(data,1)
        
        # Performing image processing steps
        image = crop_image(image)

        divided_images = split_to_five(image)
       
        normalize_img_with_mean=partial(normalize_img, mean_img=mean_img)

        normalized_images = list(map(normalize_img_with_mean, divided_images))
       
        # Returning a vector of the predicted room occupancy
        room_vector = list(map(partial(predict_occupancy,graph), normalized_images))

        print("Room vector {}".format(room_vector))

        stream.truncate(0)
        stream.seek(0)

graph.DeallocateGraph()
device.CloseDevice()



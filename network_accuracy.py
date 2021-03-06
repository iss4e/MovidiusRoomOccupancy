#!/usr/bin/python3

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

import image_process as img_ps

def predict_occupancy(graph,img):
    # Load the image as a half-precision floating point array
    graph.LoadTensor( img.astype( numpy.float16), 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    return numpy.argmax(output)

HOME_PATH               = '/home/pi/occupancy_detection'
GRAPH_PATH              = HOME_PATH + '/data/graph' 
mean_img = numpy.load(HOME_PATH + '/data/image_mean.npy').mean(1).mean(1) 
output_path = sys.argv[1]

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

print("Setting up camera...")
camera = PiCamera()
time.sleep(0.2)
stream = io.BytesIO()
camera.resolution = (3280, 2464)

now = datetime.datetime.now()
endTime = now.replace(hour = 19, minute=0, second=0,microsecond=0)
lastCaptured = now
captureRate = 5

numUploaded = 0
dayString = now.strftime("%b-%d-%a")
uploadPath = os.path.join(output_path, dayString)

print("Creating upload path {}".format(uploadPath))

try:
    os.makedirs(uploadPath)
except OSError:
    if not os.path.isdir(uploadPath):
        raise

while(True):
    timestamp = datetime.datetime.now()
    if timestamp > endTime:
        sys.exit(0)
    if (timestamp - lastCaptured).seconds >= captureRate:
        print("Capturing Image...")
        lastCapured = timestamp
        camera.capture(stream, format="jpeg")
        data = numpy.fromstring(stream.getvalue(),dtype=numpy.uint8)
        image = cv2.imdecode(data,1)

        image = img_ps.crop_image(image)

        filePath = os.path.join(uploadPath, "cap_{:03d}.jpg".format(numUploaded))
        cv2.imwrite(filePath, image)
	
        divided_images = img_ps.split_to_five(image)

        normalize_img_with_mean=partial(img_ps.normalize_img, mean_img=mean_img)

        normalized_images = list(map(normalize_img_with_mean, divided_images))
 
        room_vector = list(map(partial(predict_occupancy,graph), normalized_images))
        
        room_vec_file = os.path.join(uploadPath, "cap_{:03d}.txt".format(numUploaded))
        
        with open(room_vec_file, 'w+') as room_txt:
            room_txt.write(''.join([str(vec) for vec in room_vector]))

        numUploaded += 1

        stream.truncate(0)
        stream.seek(0)

graph.DeallocateGraph()
device.CloseDevice()



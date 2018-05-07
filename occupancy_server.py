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
import socket

import image_process as img_ps

def predict_occupancy(graph,img):
    """ Returns a 1 if the room is predicted to be occupied, else 0 """

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img.astype( numpy.float16), 'user object' )
    # Get the results from NCS
    output, userobj = graph.GetResult()
    # return output.tolist()
    return numpy.argmax(output)


# Modifiable paths
home_path = '/home/pi/occupancy_detection'
graph_path = home_path + '/data/graph'

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len( devices ) == 0:
	print( 'No devices found' )
	quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device( devices[0] )
device.OpenDevice()

# Read the graph file into a buffer
with open( graph_path, mode='rb' ) as f:
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
captureRate = 5

# NETWORKING DETAILS
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #IP-4 address
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# bind socket to an IP address (of the computer)
name = "raspberrypi.local"
ip = socket.gethostbyname(name) # IP address of the host computer
print("IP Address of Movidius Pi {}".format(ip))
print("Attempting to listen...")
port = 1234
address = (ip, port)
server.bind(address)
server.listen(1)
client, addr = server.accept()
print("Started listening on {} : {}".format(ip, port))


# Load the image mean
mean_img = numpy.load( home_path+ '/data/image_mean.npy').mean(1).mean(1) 

while(True):
    timestamp = datetime.datetime.now()

    if timestamp > today10pm:
        sys.exit(0)
    data = client.recv(1024)
    print("Received {} from the client".format(data))
    
    if data.decode() == "Read":
        lastCapured = timestamp
        print("CAPTURING IMAGE")
        camera.capture(stream, format="jpeg")
        data = numpy.fromstring(stream.getvalue(),dtype=numpy.uint8)
        image = cv2.imdecode(data,1)
        
        # Performing image processing steps
        image = img_ps.crop_image(image)

        divided_images = img_ps.split_to_five(image)
       
        normalize_img_with_mean=partial(img_ps.normalize_img, mean_img=mean_img)

        normalized_images = list(map(normalize_img_with_mean, divided_images))
       
        # Returning a vector of the predicted room occupancy
        room_vector = list(map(partial(predict_occupancy,graph), normalized_images))

        message = " ".join(map(str,room_vector))

        client.sendto(message.encode('utf-8'), address)

        print("Room vector {}".format(room_vector))

        stream.truncate(0)
        stream.seek(0)

graph.DeallocateGraph()
device.CloseDevice()



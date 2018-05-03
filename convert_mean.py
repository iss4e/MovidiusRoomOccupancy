#! /usr/bin/env python3

import caffe
import numpy as np
import sys
import os

""" Converts a Caffe specific binaryproto image mean file into a numpy array file"""

binary_proto_path = '/home/sasha-d/research_2018/model2/input/mean.binaryproto' 
output_numpy_mean_path = 'image_mean.npy' 

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(binary_proto_path , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( output_numpy_mean_path, out )


## Detecting Room Occupancy using the Intel Movidius Stick

This repository consists of python scripts used to detect people within an office. This is done by capturing an overhead image, splitting the image into 5 regions and performing basic pre-processing on these 5 regions. Once processed, these images are run through a trained CNN that classifies each image as containing people (or not). The result of this classification can be saved for assessing the CNN's accuracy or broadcast over the network as a vector for other applications.
___
#### Prerequisites:
Within this repository, create a `data/` folder that contains an `image_mean.npy` file, and the compiled computational graph of your trained Neural Network,  named `graph`. The mean file can be generated using the `convert_mean.py` file, as explained below. The details of generating a `graph` file from a trained caffe model are outlined [here](https://movidius.github.io/blog/deploying-custom-caffe-models/).
___
#### An overview of each file:  

[convert_mean.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/convert_mean.py):  
Converts Caffe's binaryproto files used for the mean image into the npy format, which can be used by the Movidius API.  
*Usage*: Modify the hardcoded path for the mean binaryproto file within the program, and run the script (no args) to generate a local *image_mean.npy* file.

[image_process.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/image_process.py):  
Consists of image processing functions similar to those used in training of the Caffe Classification network. Note that the network requires a 227x227 image due to the constraints of transfer learning, and so the `transform_img` function is used for resizing the images and providing the necessary padding to the image.  
*Usage*: Should be used as a library, see *network_accuracy.py* and *occupancy_server.py* for example usages of the functions.

[network_accuracy.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/network_accuracy.py):  
Used to assess the accuracy of the CNN's predictions. Saves a captured image and its predicted occupancy label, which can then be validated manually to see the true accuracy of the network.  
*Usage*: There are 4 main paths that should be supplied to the program. The `HOME_PATH` in the script is the location of this repository. The `GRAPH_PATH` sh




   


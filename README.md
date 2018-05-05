## Detecting Room Occupancy using the Intel Movidius Stick

This repository consists of python scripts used for the task of detecting people within an office. This is done by capturing an overhead image, splitting the image into 5 smaller images and performing basic pre-processing on these 5 images. Once processed, these images are run through a trained CNN that classifies each image as containing people (or not). The result of this classification can be saved for assessing the CNN's accuracy or broadcast over the network as a binary vector (1's for occupied, else 0) for other applications.
___
#### Prediction vector format
Since the image of the room is split into 5 regions (4 corners and a middle region), the vector has a length of 5, with a 1 corresponding to an occupied region and 0 being unoccupied. The spatial positions corresponding to each vector index are as follows: \[top left corner, top right corner, bottom left corner, bottom right corner, middle of room\].
___
#### Prerequisites:
Within this repository, create a `data/` folder that contains an `image_mean.npy` file, and the compiled computational graph of your trained Neural Network,  named `graph`. The mean file can be generated using the `convert_mean.py` file, as explained below. The details of generating a `graph` file from a trained caffe model are outlined [here](https://movidius.github.io/blog/deploying-custom-caffe-models/).
___
#### An overview of each file:  

[convert_mean.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/convert_mean.py):  
Converts Caffe's binaryproto files used for the mean image into the npy format, which can be used by the Movidius API.  
*Usage*: Modify the hardcoded path for the mean binaryproto file within the program, and run the script (no args) to generate a local `image_mean.npy` file.

[image_process.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/image_process.py):  
Consists of image processing functions similar to those used in training of the Caffe Classification network. Note that the network requires a 227x227 image due to the constraints of transfer learning, and so the `transform_img` function is used for resizing the images and providing the necessary padding to the image.  
*Usage*: Should be used as a library, see `network_accuracy.py` and `occupancy_server.py` for example usages of the functions.

[network_accuracy.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/network_accuracy.py):  
Used to assess the accuracy of the CNN's predictions. Saves a captured image and its predicted occupancy label, which can then be validated manually to see the true accuracy of the network.  
*Usage*: There are 4 main paths that should be supplied to the program. The `HOME_PATH` in the script is the location of this repository. The `GRAPH_PATH` should point to the `data/graph` file within the repository. The `mean_img` is the path to the image mean in the npy format. The final path that must be supplied is the `output_path`, which will be populated with labelled folders for each day containing an image and its corresponding vector label in a text file. The `output_path` is supplied as a command line argument when the script is run.  
*Note*: Two variables may be useful for customizing the behaviour of the script.
* `captureRate` determines the period at which images are taken, in seconds. 
* `endTime` is the time at which the script will end on the current day, which is useful if the script is launched automatically on a daily basis (with a cronjob)

[occupancy_server.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/occupancy_server.py):  
Sends the predicted occupancy of the room as a binary vector.   
*Usage*: As with `network_accuracy.py`, necessary paths should be modified in the script. By default, the occupancy server broadcasts an occupancy vector over port `1234` upon receiving a `Read` message (case-sensitive) from the client. Thus the client should connect to the Raspberry Pi's IP address and broadcast a `Read` message when a prediction is desired. No command-line arguments are required when running the script.
   


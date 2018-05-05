## Detecting Room Occupancy using the Intel Movidius Stick

This repository consists of python scripts used to detect people within an office.


An overview of each file:  

   [convert_mean.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/convert_mean.py):  
Converts Caffe's binaryproto files used for the mean image into the npy format, which can be used by the Movidius API.  
*Usage*: Modify the hardcoded path for the mean binaryproto file within the program, and run the script (no args) to generate a local *image_mean.npy* file.

[image_process.py](https://github.com/sashaDoubov/MovidiusRoomOccupancy/blob/master/image_process.py):
   


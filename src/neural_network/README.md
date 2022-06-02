# Neural Network

The core of this project is based upon a regionbased convolutional neural network, or R-CNN in short. 
This project uses the familiar model based upon Mask R-CNN, which detects ROIs within images and outputs masks on trained inferences. 

## Training

The training of the final model has been conducted in multiple stages. These will be described below:

### Using .ipynb Python Notebooks 
Everything used to train the model is written in interactive python notebook files, or .ipynb files in short. 
These files allows for segmential execution of python code. This is quite useful when training a neural network.


### Using Google Colaboratory
At first the project was written and tested on Google Colab, however this was not suitable for the amount of required GPU power. 
The group was shut down on Google Colab for too exessive use. Thus we switched to CLAAUDIA.

### Using CLAAUDIA's AI-Cloud
CLAAUDIA is an AAU based service for primarily research using heavy amounts of computing power. 
In general the group has gained access to the AI-Cloud, consisting of two Nvidia DGX-2 servers, each with 16x Nvidia Tesla V100 GPUs.

About this project : 

I used transfer learning to train a neural network to predict the species of a flower from 102 different flowers based on the image provided by the user. This was then turned into a command line application where different parameters can be changed and used to specify how to train and which neural network to use. Pytorch was used to develop the neural network. The argparse module was used to build the command line interface.

The syntax for the command line interface can be explained by the following : 

1) To train a new network on a data set with train.py
	
Basic usage: python train.py data_directory (Prints out training loss, validation loss, and validation accuracy as the network trains)
        
Options:

*Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

*Choose architecture: python train.py data_dir --arch "vgg13"

*Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

*Use GPU for training: python train.py data_dir --gpu

eg: python train.py /home/ishan/DSND/Image_Classifier/flowers --save_dir /home/ishan/DSND/Image_Classifier --arch "vgg13" --learning_rate 0.005

2) Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image's /path/to/image and you will be returned the flower name and which were the most probable classes for that flower.
       
Basic usage: python predict.py /path/to/image checkpoint
        
Options:

*Return top K most likely classes: python predict.py input checkpoint --top_k 3

*Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

*Use GPU for inference: python predict.py input checkpoint --gpu

eg: python predict.py /home/ishan/DSND/Image_Classifier/flowers/train/1/image_06734.jpg /home/ishan/DSND/Image_Classifier/checkpoint.pth --top_k 2 --gpu 

This folder has 5 sub-folders/files : 

1) flowers : This contains the training, validating and testing data under train, valid and test respectively.

2) Image Classifier Project.html : This is the html screenshot of the jupyter notebook 'Image Classifier Project.ipynb' and acts as an easy alternative for reading through the code of the notebook.

3) Image Classifier Project.html : This is the Jupyter notebook in which our neural network has been developed, trained, validated and tested. All the code related to the development of the neural network can be found here.

4) train.py : Contains all code related to developing a command line interface for creating and training a neural network.

5) predict.py : Contains all code related to developing a command line interface for loading a trained neural network and using it to predict a given image.

	

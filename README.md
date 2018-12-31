# DSND

This is the repository for my Data Science Nanodegree projects from Udacity.
There are 3 projects currently and each folder contains their html files and .ipynb Jupyter notebooks along with any other files 
which are needed.

Finding_Donors_Supervised : I used supervised machine learning (classification) to find people who were most likely to be donors according to several publically available features from a dataset. The dataset for this project originates from the UCI Machine Learning Repository and was donated by Ron Kohavi and Barry Becker. I performed grid search to tune the model and explored feature importance using the AdaBoostClassifier. 
                            
Image_Classifier : I used transfer learning to train a neural network to predict the species of a flower from 102 different flowers based on the image provided by the user. This was then turned into a command line application where different parameters can be changed and used to specify how to train and which neural network to use. Pytorch was used to develop the neural network. The argparse module was used to build the command line interface.

Customer_Segmentation_Unsupervised : I worked with real-life data provided to us by the Bertelsmann partners AZ Direct and Arvato Finance Solution.
The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign.
I used unsupervised learning techniques to organize the general population into clusters, and then used those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, I also assessed and cleaned the data in order to convert the data into a usable form.  

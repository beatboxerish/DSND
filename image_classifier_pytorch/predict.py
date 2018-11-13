import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('path',help = 'Specify the path to the image')
parser.add_argument('checkpoint',help = 'Specify the location of the saved model you want to use')
parser.add_argument('--top_k',help = 'Returns the top "k" classes with maximum probability to which the image belongs to',type = int)
parser.add_argument('--category_names',help = 'Specify the dictionary/mapping of index of classes to names of classes')
parser.add_argument('--gpu',help = 'Specify this option if you want to use gpu for inference',action = 'store_true')
args = parser.parse_args()

#-------------------------------
# Code for predicting the image
# Importing libs
from torchvision import datasets,transforms,models
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import OrderedDict
from PIL import Image

# Setting up args values
checkpoint = args.checkpoint
path_image = args.path
if args.top_k:
    top_k = args.top_k
    
# Defining functions    
def model_load(model,checkpoint):
    ''' Only use this function if using transfer learning from one of the models provided under
        torchvision.models module.'''
    checkpoint = torch.load(checkpoint)
    model.name = checkpoint['name']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        model.class_labels = cat_to_name
    else:
        model.class_labels = checkpoint['class_labels']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs'] 
    return model
def process_image(path_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(path_image)
    im = im.resize((256,256))
    d = (256-224)/2
    im = im.crop((d,d,256-d,256-d))
    
    img_array = np.array(im)
    img_array = img_array.T
    img_array = (img_array - img_array.min())/(img_array.max() - img_array.min())
    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]
    
    for i,j in enumerate(img_array):
        img_array[i] = (j - mean_[i])/std_[i]
        
    return img_array
def predict(path_image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if args.top_k:
        topk = top_k
    model.eval()
    img = process_image(path_image)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.view(1,3,224,224).float()
    
    if args.gpu:
        model.to('cuda')
        img = img.cuda()
    else:
        img.to('cpu')
        model.to('cpu')
        
    with torch.no_grad():
        output = model.forward(img)

    probs = torch.exp(output.topk(topk)[0])
    classes = output.topk(topk)[1]
    return probs,classes

# Implementing functions and finally predicting the image
model = models.densenet121(pretrained=True)
if args.gpu:
    model.to('cuda')

model = model_load(model,checkpoint)
probs, classes = predict(path_image, model)

classes = [model.class_labels[str(int(i))] for i in classes[0]]

print('The topk classes are :',classes)
print('The topk probs are :',probs[0])
#-------------------------------
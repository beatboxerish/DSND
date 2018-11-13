import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',help ='Specify the parent folder in which the training data and validation data folders are located')
parser.add_argument('--save_dir',help = 'Specify the location where you want to save checkpoints')
parser.add_argument('--arch',help = 'Specify the architecture of the model you want to load')
parser.add_argument('--learning_rate',help = 'Specify the learning rate for training the model',type = float)
parser.add_argument('--hidden_units',help = 'Specify how many units you want in the hidden layer',type = int)
parser.add_argument('--epochs',help = 'Specify the number of epochs you want the model trained for',type = int)
parser.add_argument('--gpu',help = 'Use this option only if you want to use gpu to train your model',action = 'store_true')
args = parser.parse_args()

#--------------------------------
# Class to train a model
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

class train:
    def __init__(self,data_dir):
        self.data_dir = data_dir
    
    def train_now(self, save_dir = None, arch = 'densenet121', learning_rate = 0.001, hidden_units = 100,epochs = 10, gpu = False,print_every = 100):
        # Setting the args values
        save_dir = self.data_dir
    
        if args.arch:
                arch = args.arch     

        if args.learning_rate:
            learning_rate = args.learning_rate

        if args.hidden_units:
            hidden_units = args.hidden_units

        if args.epochs:
            epochs = args.epochs

        if args.gpu:
            gpu = args.gpu
            
        # Loading and dividing the data
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        train_transforms               =transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomRotation(35),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.458,0.456,0.406],[0.229,0.224,0.225])])
        valid_transforms= transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.458,0.456,0.406],[0.229,0.224,0.225])])
        train_datasets = datasets.ImageFolder(train_dir,transform = train_transforms)
        class_labels = train_datasets.classes
        valid_datasets = datasets.ImageFolder(valid_dir,transform = valid_transforms)
        trainloader = torch.utils.data.DataLoader(train_datasets,batch_size = 64,shuffle = True)
        validloader = torch.utils.data.DataLoader(valid_datasets,batch_size = 64)
        # Building the model
        if arch == 'vgg13':
            model = models.vgg13(pretrained = True)
        elif arch == 'alexnet':
            model = models.alexnet(pretrained = True)
        elif arch == 'densenet121':
            model = models.densenet121(pretrained = True) 
        else:
            raise ValueError('Try again. Enter one of these models : [alexnet, vgg13, densenet121]')
         
        try:
            num_input_units = model.classifier[0].in_features
        except :
            num_input_units = model.classifier.in_features
            
        classifier_layer = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_input_units,hidden_units)),
                                                      ('relu_1' ,nn.ReLU()),
                                                      ('dropout_1' , nn.Dropout(0.5)),
                                                      ('fc2' ,nn.Linear(hidden_units,hidden_units)),
                                                      ('relu_2' , nn.ReLU()),
                                                      ('dropout_2' , nn.Dropout(0.2)),
                                                      ('output', nn.Linear(hidden_units,102)),
                                                      ('softmax' , nn.LogSoftmax(dim =1 ))]))
        model.classifier = classifier_layer

        # Setting the training process and the hyperparameters
        model.epochs = 0
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(),lr =learning_rate  )
        
        steps = 0
        if gpu:
            model.to('cuda')
        model.learning_rate = learning_rate
        
        
        for epoch in range(epochs):
            model.epochs += 1
            print('The models current epoch :',model.epochs)
            running_loss = 0
            for img,lab in trainloader:
                model.train()
                steps += 1
                if gpu :
                    img,lab = img.to('cuda'),lab.to('cuda')

                optimizer.zero_grad()
                output = model.forward(img)
                loss = criterion(output,lab)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if steps%print_every == 0:

                    valid_loss = 0
                    n = 0
                    accuracy = 0
                    for i,j in validloader:
                        if gpu :
                            i,j = i.to('cuda'),j.to('cuda')
                        n += i.shape[0]
                        model.eval()
                        with torch.no_grad():
                            valid_output = model.forward(i)    
                        valid_loss += criterion(valid_output,j).item()
                        accuracy += (valid_output.max(dim = 1)[1] == j).sum().item()

                    print('Epoch : {}/{}'.format(epoch + 1,epochs))
                    print('Training Loss :',running_loss/(print_every*img.shape[0]))
                    print('Validation Loss :',valid_loss/n)
                    print('Accuracy :',accuracy*100/n,'%')

                    running_loss = 0
                    model.train()
        # Setting the model checkpoint and saving the model
        with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        model_checkpoint = {'name': arch,
                        'learning_rate' : model.learning_rate,
                        'state_dict' : model.state_dict(), 
                        'epochs' : model.epochs ,
                        'classifier' : model.classifier,
                        'norm_vectors' : [[0.458,0.456,0.406],[0.229,0.224,0.225]],
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'class_labels' : cat_to_name
                        }
    
        torch.save(model_checkpoint,save_dir + '/checkpoint.pth')
#-------------------------------
#Initialize an object and run method train_now()

train_object = train(args.data_dir)
train_object.train_now()

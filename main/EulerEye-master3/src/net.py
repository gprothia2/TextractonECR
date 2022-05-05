import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.feature_extract import FeatureExtractNet
from torchvision import models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractNet(nn.Module):
    """
    This network using extract features from the network
    """
    def __init__(self):
        """
        define a convolutional neural network to generate feature map
        
        Args:
            Hard code here, TODO: config network with config file
        """
        super(FeatureExtractNet, self).__init__()
        
        # conv1 downsample=1/2
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv2 downsample=1/4
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv3 downsample=1/8
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn3_2 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv4 downsample=1/16
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.conv4_1 = nn.Conv2d(16, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn4_2 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv5 downsample=1/32
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.conv5_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.conv5_2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv6 downsample=1/64
        self.dropout6 = nn.Dropout2d(p=0.3)
        self.conv6_1 = nn.Conv2d(128, 128, kernel_size = 5, stride = 1, padding = 2) # different with paper
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn6_2 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv7 
        self.dropout7 = nn.Dropout2d(p=0.3)
        self.conv7_1 = nn.Conv2d(256, 256, kernel_size = 5, stride = 1, padding = 2) # different with paper
        self.bn7_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            x(Variable): contains a batch of images, of dimension batch_size x channels x H x W .

        Returns:
            out(list): list of feature maps for each conv step 
            
        Note the dimensions after each step are provided
        """
        #input                                            batch_size x 2 x 256 x 256

        # block1
        x = self.dropout1(x)                            # batch_size x 2 x 256 x 256
        x = F.relu(self.bn1_1(self.conv1_1(x)))         # batch_size x 32 x 256 x 256
        x = F.relu(self.bn1_2(self.conv1_2(x)))         # batch_size x 16 x 256 x 256
        x = self.pool1(x)                               # batch_size x 16 x 128 x 128
        pool1 = x

        # block2
        x = self.dropout2(x)                            # batch_size x 16 x 128 x 128
        x = F.relu(self.bn2_1(self.conv2_1(x)))         # batch_size x 16 x 128 x 128
        x = F.relu(self.bn2_2(self.conv2_2(x)))         # batch_size x 16 x 128 x 128
        x = self.pool2(x)                               # batch_size x 16 x 64 x 64
        pool2 = x

        # block3
        x = self.dropout3(x)                            # batch_size x 16 x 64 x 64
        x = F.relu(self.bn3_1(self.conv3_1(x)))         # batch_size x 16 x 64 x 64
        x = F.relu(self.bn3_2(self.conv3_2(x)))         # batch_size x 16 x 64 x 64
        x = self.pool3(x)                               # batch_size x 16 x 32 x 32
        pool3 = x

        # block4
        x = self.dropout4(x)                            # batch_size x 16 x 32 x 32
        x = F.relu(self.bn4_1(self.conv4_1(x)))         # batch_size x 64 x 32 x 32
        x = F.relu(self.bn4_2(self.conv4_2(x)))         # batch_size x 64 x 32 x 32
        x = self.pool4(x)                               # batch_size x 64 x 16 x 16
        pool4 = x

        # block5
        x = self.dropout5(x)                            # batch_size x 64 x 16 x 16
        x = F.relu(self.bn5_1(self.conv5_1(x)))         # batch_size x 64 x 16 x 16
        x = F.relu(self.bn5_2(self.conv5_2(x)))         # batch_size x 128 x 16 x 16
        x = self.pool5(x)                               # batch_size x 128 x 8 x 8
        pool5 = x

        # block6
        x = self.dropout6(x)                            # batch_size x 128 x 8 x 8
        x = F.relu(self.bn6_1(self.conv6_1(x)))         # batch_size x 128 x 8 x 8
        x = F.relu(self.bn6_2(self.conv6_2(x)))         # batch_size x 256 x 8 x 8
        x = self.pool6(x)                               # batch_size x 256 x 4 x 4
        pool6 = x

        # block1
        x = self.dropout7(x)                            # batch_size x 256 x 4 x 4
        x = F.relu(self.bn7_1(self.conv7_1(x)))         # batch_size x 256 x 4 x 4
        pool7 = x

        return [pool1, pool2, pool3, pool4, pool5, pool6, pool7]

class FCN(nn.Module):
    """
    This network using to making upsampling and predicting for Fully Convolutional Neural Network
    """
    def __init__(self):
        """
        define a de-convolutional process for feature map
        """
        super(FCN, self).__init__()
        # self.num_class = num_class  #TODO: config with config file
        self.features = FeatureExtractNet()
        
        # transpose_conv8
        self.transpose_conv8 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 0)
        
        # transpose_conv9
        self.transpose_conv9 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)

        # transpose_conv10
        self.transpose_conv10 = nn.ConvTranspose2d(64, 16, kernel_size = 2, stride = 2, padding = 0)

        # transpose_conv11
        self.transpose_conv11 = nn.ConvTranspose2d(16, 16, kernel_size = 4, stride = 4, padding = 0)

        # refinement_conv12
        self.conv12_1 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_1 = nn.BatchNorm2d(32)
        self.conv12_2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_2 = nn.BatchNorm2d(32)
        self.conv12_3 = nn.Conv2d(32, 8, kernel_size = 1, stride = 1, padding = 0)
        self.dropout12 = nn.Dropout(p=0.3)
        self.conv12_4 = nn.Conv2d(8, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_4 = nn.BatchNorm2d(32)
        self.conv12_5 = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn12_5 = nn.BatchNorm2d(16)

        # classification13
        self.conv13 = nn.Conv2d(16, 1, kernel_size = 1, stride = 1, padding = 0)
        self.upscale_13 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            x(Variable): contains a batch of images with dimension batch_size x channels x H x W

        Returns:
            out(Variable): value of each pixel(probability) 
            
        Note the dimensions after each step are provided
        """
        #input                                                           batch_size x 2 x 256 x 256
        # feature map extraction
        features = self.features(x)

        # transpose_conv8 
        x = F.relu(self.transpose_conv8(features[-1]) + features[4])    # batch_size x 128 x 8 x 8
       
        # transpose_conv9
        x = F.relu(self.transpose_conv9(x) + features[3])               # batch_size x 64 x 16 x 16
        
        # transpose_conv10
        x = F.relu(self.transpose_conv10(x) + features[2])              # batch_size x 16 x 32 x 32
        
        # transpose_conv11
        x = F.relu(self.transpose_conv11(x))                            # batch_size x 16 x 128 x 128
    
        # refinement12
        x = F.relu(self.bn12_1(self.conv12_1(x)))                       # batch_size x 32 x 128 x 128
        x = F.relu(self.bn12_2(self.conv12_2(x)))                       # batch_size x 32 x 128 x 128
        x = torch.sigmoid(self.conv12_3(x))                                 # batch_size x 8 x 128 x 128
        x = self.dropout12(x)                                           # batch_size x 8 x 128 x 128
        x = F.relu(self.bn12_4(self.conv12_4(x)))                       # batch_size x 32 x 128 x 128
        x = F.relu(self.bn12_5(self.conv12_5(x)))                       # batch_size x 16 x 128 x 128
        

        # upsampling13
        # x = F.softmax(self.conv13(x), dim=1)                            # batch_size x 1 x 128 x 128
        x = torch.sigmoid(self.conv13(x)) 
        out = self.upscale_13(x)                                        # batch_size x 1 x 256 x 256
        
        return out
    
    
def pretrained_model(pretrained=True):
    fcn_model = models.segmentation.fcn_resnet101(pretrained=pretrained)
    newspaper_FCNHead_cls = models.segmentation.fcn.FCNHead(2048, 2)
    newspaper_FCNHead_auxcls = models.segmentation.fcn.FCNHead(1024, 2)
    fcn_model.classifier = newspaper_FCNHead_cls
    fcn_model.aux_classifier = newspaper_FCNHead_auxcls
    return fcn_model
 
    
    
def accuracy(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all images

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 --- log softmax output of the model
        labels: (np.ndarray) dimiension batch_size

    Returns:
        accuracy: (float) accuracy in [0, 1]
    '''
    # outputs = np.argmax(outputs, axis=1)
    outputs = outputs > 0.5
    return np.sum(outputs==labels)/float(labels.size)

def class_level_acc(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all images

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 --- log softmax output of the model
        labels: (np.ndarray) dimiension batch_size

    Returns:
        accuracy: (dict) accuracy for each class
    '''

    _class = np.unique(labels)
    outputs = np.argmax(outputs, axis=1)
    accuracy = {}
    for c in _class:
        idx = np.where(labels == c)
        idx = idx[0]
        label_c = len(idx)
        mask_c = np.sum(outputs[idx]==labels[idx])
        accuracy[c] = mask_c / float(label_c)
    return accuracy

def binary_class0_acc(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all images

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 x img_h x img_w
        labels: (np.ndarray) dimiension batch_size x 1 x img_h x img_w
 
    Returns:
        accuracy: (float) accuracy for class0
    '''
    outputs = outputs > 0.5
    idx = np.where(labels==0)
    accuracy = np.sum(outputs[idx] == labels[idx]) / (float(labels[idx].size) + 1)
    return accuracy

def binary_class1_acc(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all images

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 x img_h x img_w
        labels: (np.ndarray) dimiension batch_size x 1 x img_h x img_w
 
    Returns:
        accuracy: (float) accuracy for class1
    '''
    outputs = outputs > 0.5
    idx = np.where(labels==1)
    accuracy = np.sum(outputs[idx] == labels[idx]) / (float(labels[idx].size) + 1)
    return accuracy

# maintain all metrics required in this dictionary
metrics = {
    'accuracy': accuracy,
    'class0_accuracy':binary_class0_acc,
    'class1_accuracy':binary_class1_acc,
}
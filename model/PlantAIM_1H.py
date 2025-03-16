'''
This is pytorch impplementation of PlantAIM(1H) from
PlantAIM: A New Baseline Model Integrating Global Attention and Local Features for Enhanced Plant Disease Identification
https://www.sciencedirect.com/science/article/pii/S2772375525000474

'''

import pandas as pd
import numpy as np
import os
import torch
import timm
from tqdm import tqdm

import cv2
import torchvision
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

from torch.utils.tensorboard import SummaryWriter

from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_
from functools import partial
    
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, Rotate,
    ShiftScaleRotate, CenterCrop, Crop, Resize, Rotate, RandomShadow, RandomSizedBBoxSafeCrop,
    ChannelShuffle, MotionBlur
)

from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_score, recall_score, f1_score


class CustomDatasetFromImagesForalbumentation(Dataset):
    def __init__(self, csv_path, transforms, data_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # third column is the labels for plant disease
        self.label_arr_p = np.asarray(self.data_info.iloc[:, 3])

        
        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.data_path = data_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.data_path, single_image_name)
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get combined plant disease label of the image 
        single_image_label_p = self.label_arr_p[index]

        return (img_as_tensor, single_image_label_p)

    def __len__(self):
        return self.data_len

class Model_SL(nn.Module):
    def __init__(self,model_v, model_c, num_plant):
        super(Model_SL,self).__init__()
        
        # ViT and CNN model
        self.model_v = model_v
        self.model_c = model_c

        # Parameters for MLP and self-attention block
        self.dim = 768
        self.dim_a = 768
        self.num_features = 2048
        self.num_heads = 12
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.drop = 0.
        self.attn_drop = 0.
        self.drop_path = 0.
        self.act_layer = nn.GELU
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # To rescale the CNN features to match ViT features
        self.mlp = nn.Linear(self.num_features, self.dim, bias=True)
        
        self.block1 = Block(
                dim= self.dim_a,
                num_heads= self.num_heads,
                mlp_ratio= self.mlp_ratio,
                qkv_bias= self.qkv_bias,
                attn_drop= self.attn_drop,
                drop_path= self.drop_path,
                norm_layer= self.norm_layer,
                act_layer= self.act_layer
            )
        
        # Classifier layer
        self.fc_p = nn.Linear(self.dim, num_plant)

        
        # Apply weight initialization only to new layers
        self._init_weights(self.mlp)
        self._init_weights(self.block1)
        self._init_weights(self.fc_p)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def forward(self, i1):
        # Obtain ViT features
        self.f1 = self.model_v.forward_features(i1)
        
        # Obtain CNN features
        self.f2 = self.model_c.forward_features(i1)
        self.f2 =  self.avgpool(self.f2)
        self.f2 = torch.flatten(self.f2, 1)
        self.f2 = self.mlp(self.f2)

        # Feature summation / multiplication    
        # self.f2 = self.f1[:, 0] + self.f2
        self.f2 = torch.mul(self.f1[:, 0], self.f2)
        
        # attention
        self.f1t = self.f1[:, 1:]
        self.f3 = torch.cat((self.f2.unsqueeze(1), self.f1t), dim=1)
        self.f3 = self.block1(self.f3)

        # Predictions
        self.h1p =  self.fc_p(self.f3[:, 0] + self.f1[:, 0]) 
        return (self.h1p)        
    
# Hyperparameters and variables
num_classes = 38
# num_plant = 14
# num_disease = 21
num_epochs = 30
batch_size = 16
img_size = 224
learning_rate_layer = 0.001
pretrained = True
momentum = 0.9
weight_decay = 0.00001

# Default data augmentation
ori_train_transforms = Compose([
            RandomResizedCrop(img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)


ori_valid_transforms = Compose([
            Resize(img_size, img_size),
            CenterCrop(img_size, img_size, p=1.),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# Data path for datasets and metadatas
data_path1 = 'path to PV dataset'
data_path2 = 'path to IPM and Bing dataset'
data_path3 = 'path to PlantDoc dataset'

Save_model_path = "path to save ur model"

train_csv_path = 'path to PV_train_3L_10s.csv' 
test_csv_path1 = 'path to PV_test_seen_3L.csv' 
test_csv_path2 = 'path to PV_test_limited_3L.csv' 
test_csv_path3 = 'path to IPM_test_3L.csv' 
test_csv_path4 = 'path to Bing_test_3L.csv' 
test_csv_path5 = 'path to Plantdoc_test_3L.csv' 

# Dataset declarations
train_dataset = CustomDatasetFromImagesForalbumentation(train_csv_path, transforms = ori_train_transforms, data_path = data_path1)
test_dataset1 = CustomDatasetFromImagesForalbumentation(test_csv_path1, transforms = ori_valid_transforms, data_path = data_path1)
test_dataset2 = CustomDatasetFromImagesForalbumentation(test_csv_path2, transforms = ori_valid_transforms, data_path = data_path1)
test_dataset3 = CustomDatasetFromImagesForalbumentation(test_csv_path3, transforms = ori_valid_transforms, data_path = data_path2)
test_dataset4 = CustomDatasetFromImagesForalbumentation(test_csv_path4, transforms = ori_valid_transforms, data_path = data_path2)
test_dataset5 = CustomDatasetFromImagesForalbumentation(test_csv_path5, transforms = ori_valid_transforms, data_path = data_path3)

# Dataset loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, drop_last = False)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, drop_last = False)
test_loader3 = DataLoader(test_dataset3, batch_size=batch_size, shuffle=False, drop_last = False)
test_loader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=False, drop_last = False)
test_loader5 = DataLoader(test_dataset5, batch_size=batch_size, shuffle=False, drop_last = False)

# GPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

# ViT model from Timm with ImageNet pretrained weight
model_name = "vit_base_patch16_224"
model_v = timm.create_model(model_name, pretrained=False).to(device)
model_v.to(device)

# CNN model from torchvision with pretrained weight
model_c = models.resnet152(pretrained=True)
model_c.to(device)

# Tensorboard for progress monitoring
tb = SummaryWriter()

# print(model)

model_sl = Model_SL(model_v, model_c, num_classes)
model_sl.to(device)
# print(model_sl)


if pretrained:
    print("Using Pre-Trained ViT Model")
    MODEL_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/saved model/Pre-trained (official)/trained/jx_vit_base_p16_224-80ecf9dd (rwightman, ImageNet21k+ImageNet2012).pth"
    model_sl.model_v.load_state_dict(torch.load(MODEL_PATH),strict=True)

# Use this if for checkpoint or fine-tune for model      
# if pretrained:
#     print("Using Pre-Trained ViT+CNN model")
#     MODEL_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/ViT_and_CNN/SOTA/ViT+CNN_12e_98.8870_98.0562_0.2w+mul+c+0a+rv_PV38-model-S0-1.pth"
#     model_sl.load_state_dict(torch.load(MODEL_PATH),strict=True)

parameters = list(model_sl.parameters())

error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(parameters, lr=learning_rate_layer, weight_decay = weight_decay, momentum = momentum)

# Use this if for checkpoint or fine-tune for model  
# if pretrained:
#     print("Using Pre-Trained optimizer")
#     OPTIMIZER_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/ViT_and_CNN/SOTA/ViT+CNN_12e_98.8870_98.0562_0.2w+mul+c+0a+rv_PV38-opt-S0-1.pth"
#     optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))

for g in optimizer.param_groups:
    g['lr'] = learning_rate_layer

for epoch in range(num_epochs):
    print(f"Start of Epoch {epoch+1} of {num_epochs}")
    print('Current Learning rate: {0}'.format(optimizer.param_groups[0]['lr']))

    # Training counter for calculation or debug
    total_train = 0
    correct_train_p = 0
    correct_train_d = 0
    total_loss = 0
    correct_train_o_combine = 0
    model_sl.train()

    # Batch size (using gradient accumulation method)
    # Actual batch size = batch_size * iter_batch
    iter_batch = 8
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):        
        images, labels  = images.to(device), labels.to(device)

        # Obtain the output for plant disease prediction
        output_p = model_sl(images)
        
        # Calculate loss for plant disease prediction
        loss_p = error(output_p, labels)
        loss_t = loss_p
        total_loss+= loss_t.item()
        
        # Obtain the predicted label
        p_predictions = torch.max(output_p,1)[1].to(device)
        
        # Calculate accuracy
        correct_train_p += (p_predictions == labels).sum()
                    
        # Gradient calculation and model update
        loss_t = loss_t / iter_batch
        loss_t.backward()
        
        if ((batch_idx + 1) % iter_batch == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
    
        total_train += len(labels)
        
        # Tensorboard record 
        tb.add_scalar("Total comb loss", total_loss, epoch)
    
    # Total accuracy calculations
    accuracy_train_p = correct_train_p * 100 / total_train
    accuracy_train_d = correct_train_d * 100 / total_train
    accuracy_o = correct_train_o_combine  * 100 / total_train
    
    print("")
    print("SL training acc for Plant: {:.4f}".format(accuracy_train_p))
    print("")
    print("SL training acc for Disease: {:.4f}".format(accuracy_train_d))
    print("")
    print("SL training acc for total PD : {:.4f}".format(accuracy_o))
    print("")
    
    # Total losses calcuations 
    a_p = total_loss / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    tb.add_scalar("Training loss per epoch", (a_p), epoch)
    print('Total Loss: {0}'.format(a_p))
    print(f"\nEpoch {epoch+1} of {num_epochs} Done!")
    print('Current Learning rate: {0}'.format(optimizer.param_groups[0]['lr']))


    # Seen test dataset evaluation
    print(f"\nSeen Testing") 
    model_sl.eval()
    
    # testing counter for calculation or debug
    total = 0
    correct_p = 0
    total_loss_p_loss = 0

    with torch.no_grad():
        
        for images, labels in tqdm(test_loader1):
            images, labels  = images.to(device), labels.to(device)
            
            # Obtain the output for plant disease prediction
            output_p = model_sl(images)
            
            # Calculate loss for plant disease prediction
            loss_p = error(output_p, labels)
            total_loss_p_loss += loss_p.item()
            
            # Obtain the predicted label
            p_prediction = torch.max(output_p,1)[1].to(device)
            
            # Calculate accuracy
            correct_p += (p_prediction == labels).sum()
    
            total += len(labels)
        
        # Total accuracy calculations
        accuracy_p = correct_p * 100 / total

        # Total losses calcuations
        a_p = total_loss_p_loss / ((len(test_dataset1)//batch_size) + (len(test_dataset1) % batch_size > 0))
        
        # Tensorboard record 
        tb.add_scalar("Testing Seen loss per epoch", (a_p), epoch)
        # tb.close()
        print("")
        print('Total testing Loss: {0}'.format(a_p))    
        print("")
        print("Testing acc for seen Plant Disease : {:.4f}".format(accuracy_p))
        print("")
    

    print(f"\nFew show Testing") 
    model_sl.eval()
    
    total = 0
    correct_p = 0   
    total_loss_p_loss = 0
    
    for images, labels in tqdm(test_loader2):
        images, labels  = images.to(device), labels.to(device)

        output_p = model_sl(images)
        
        loss_p = error(output_p, labels)
        total_loss_p_loss += loss_p.item()
       
        p_prediction = torch.max(output_p,1)[1].to(device)
       
        correct_p += (p_prediction == labels).sum()

        total += len(labels)

    accuracy_p = correct_p * 100 / total
    a_p = total_loss_p_loss / ((len(test_dataset1)//batch_size) + (len(test_dataset1) % batch_size > 0))
    tb.add_scalar("Testing few shot loss per epoch", (a_p), epoch)
    # tb.close()
    print("")
    print('Total testing Loss: {0}'.format(a_p))    
    print("")
    print("Testing acc for Few show Plant Disease: {:.4f}".format(accuracy_p))
    print("")

    print(f"\nIPM Testing") 
    model_sl.eval()
    
    total = 0
    correct_p = 0
    total_loss_p_loss = 0
    
    for images, labels in tqdm(test_loader3):
        images, labels  = images.to(device), labels.to(device)

        output_p = model_sl(images)
        
        loss_p = error(output_p, labels)
        total_loss_p_loss += loss_p.item()

        p_prediction = torch.max(output_p,1)[1].to(device)

        correct_p += (p_prediction == labels).sum()

        total += len(labels)

    accuracy_p = correct_p * 100 / total
    a_p = total_loss_p_loss / ((len(test_dataset1)//batch_size) + (len(test_dataset1) % batch_size > 0))
    tb.add_scalar("Testing IPM loss per epoch", (a_p), epoch)
    # tb.close()
    print("")
    print('Total testing Loss: {0}'.format(a_p))    
    print("")
    print("Testing acc for IPM Plant Disease: {:.4f}".format(accuracy_p))
    print("")

    print(f"\nBing Testing") 
    model_sl.eval()
    
    total = 0
    correct_p = 0
    total_loss_p_loss = 0
    
    for images, labels in tqdm(test_loader4):
        images, labels  = images.to(device), labels.to(device)

        output_p = model_sl(images)
        
        loss_p = error(output_p, labels)
        total_loss_p_loss += loss_p.item()
        
        p_prediction = torch.max(output_p,1)[1].to(device)
        
        correct_p += (p_prediction == labels).sum()

        total += len(labels)

    accuracy_p = correct_p * 100 / total
    a_p = total_loss_p_loss / ((len(test_dataset1)//batch_size) + (len(test_dataset1) % batch_size > 0))
    tb.add_scalar("Testing Bing loss per epoch", (a_p), epoch)
    # tb.close()
    print("")
    print('Total testing Loss: {0}'.format(a_p))    
    print("")
    print("Testing acc for Bing Plant Disease: {:.4f}".format(accuracy_p))
    print("")

    print(f"\nPlantDoc Testing") 
    model_sl.eval()
    
    total = 0
    correct_p = 0
    total_loss_p_loss = 0
    
    for images, labels in tqdm(test_loader5):
        images, labels  = images.to(device), labels.to(device)

        output_p = model_sl(images)
        
        loss_p = error(output_p, labels)
        total_loss_p_loss += loss_p.item()
        
        p_prediction = torch.max(output_p,1)[1].to(device)

        correct_p += (p_prediction == labels).sum() 

        total += len(labels)

    accuracy_p = correct_p * 100 / total
    a_p = total_loss_p_loss / ((len(test_dataset1)//batch_size) + (len(test_dataset1) % batch_size > 0))
    tb.add_scalar("Testing PlantDoc loss per epoch", (a_p), epoch)
    tb.close()
    print("")
    print('Total testing Loss: {0}'.format(a_p))    
    print("")
    print("Testing acc for PlantDoc Plant : {:.4f}".format(accuracy_p))
    print("")    
   
    if ((epoch+1) % 1 == 0):
        print("Saving Model")
        torch.save(model_sl.state_dict(), os.path.join(Save_model_path,'ViT+CNN_1H_{}e_{:.4f}-model-S1-1.pth'
                                                                .format(epoch+1,accuracy_p)))

        print("Saving optimizer")
        torch.save(optimizer.state_dict(), os.path.join(Save_model_path,'ViT+CNN_1H_{}e_{:.4f}-opt-S1-1.pth'
                                                                    .format(epoch+1,accuracy_p)))

        print("Saving done")

print("Training done")    

































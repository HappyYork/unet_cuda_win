import torch
from torch import nn
from _datetime import datetime
from torchvision import transforms
from PIL import Image
import json
import numpy as np

def criterion(inputs, target):
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    losses = nn.functional.cross_entropy(inputs, target, ignore_index=255)

    return losses


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    i = 0
    train_loss,correct = 0.0,0.0
    num_batches = len(data_loader)
    size = len(data_loader.dataset)
    header = 'Epoch: {} [{}/{}] time:{}'
    print(header.format(epoch,i,num_batches,datetime.now()))
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.argmax(output,dim=1)
            correct += ((pred == target).type(torch.float).sum().item())
            train_loss += loss
        i = i + 1
        if(i % 45 == 0):
            print(header.format(epoch,i,num_batches,datetime.now()))
    train_loss /=  num_batches
    correct /= size*256**2

    return train_loss,correct


def test_one_epoch(model, optimizer, val_loader, device, epoch):
    model.eval()
    i = 0
    test_loss,test_correct = 0.0,0.0
    num_batches = len(val_loader)
    size = len(val_loader.dataset)
    header = 'Epoch_t: {} [{}/{}] time:{}'

    print(header.format(epoch,i,num_batches,datetime.now()))
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            pred = torch.argmax(output,dim=1)
            test_correct += ((pred == target).type(torch.float).sum().item())
            test_loss += loss
            i = i + 1
            if(i % 100 == 0):
                print(header.format(epoch,i,num_batches,datetime.now()))

        test_loss /=  num_batches
        test_correct /= size*256**2

    return test_loss,test_correct
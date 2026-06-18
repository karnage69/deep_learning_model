import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from torchvision import datasets, transforms
import torch.optim as optim
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)
transformer= transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])
#train set to create the classes and transform the dataset(load data)
train_dataset= datasets.ImageFolder(root=r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\train",transform=transformer)
test_dataset= datasets.ImageFolder(root=r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\test",transform=transformer)
#split train data into validation
train_size= int(0.8*len(train_dataset))
val_size= len(train_dataset) - train_size 
train_data, val_data = random_split(train_dataset,[train_size, val_size])
#data loader :- for dividing image set into batches 
batch_size = 32
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_data,batch_size=batch_size)
test_loader = DataLoader(test_dataset,batch_size=batch_size)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network= nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
    def forward(self,x):
        return self.network(x)
model=MLP().to(device)
loss=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epoch=10
total_loss=[]
val_loss=[]
for i in range(epoch):
    for images, labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        y_pred=model(images)
        loss_c=loss(y_pred,labels)
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        total_loss.append(loss_c.item())
    print(f"Epoch {epoch+1}: Loss = {loss_c.item():.4f}")
    model.eval()
    correct=0
    total=0
    running_val_loss=0
    with torch.no_grad():
     for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device)
        y_pred=model(images)
        loss_n=loss(y_pred,labels)
        running_val_loss += loss_n.item()
        val_loss.append(loss_n.item())
        val_preds = torch.argmax(y_pred, dim=1)
        val_acc = (val_preds == labels).sum().item() / len(labels)
        correct +=(val_preds==labels).sum().item()
        total+= labels.size(0)
    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    print(
           f"Epoch {epoch+1}/{epoch} | "
           f"Train Loss: {loss_c.item():.4f} | "
           f"Val Acc: {val_accuracy:.4f}" )
# TESTING
model.eval()
correct = 0
total = 0
running_test_loss = 0
with torch.no_grad():
 for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    y_pred = model(images)
    loss_t = loss(y_pred, labels)
    running_test_loss += loss_t.item()
    preds = torch.argmax(y_pred, dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
test_accuracy = correct / total
avg_test_loss = running_test_loss / len(test_loader)
print(
f"Test Loss: {avg_test_loss:.4f} | "
f"Test Accuracy: {test_accuracy:.4f}"
)
torch.save(model.state_dict(),"mlp_image_classifier.pth"
)
print("Model saved successfully")
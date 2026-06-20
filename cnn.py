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
device="cuda" if torch.cuda.is_available() else "cpu"
print("using",device)
transformers= transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])
train_dataset= datasets.ImageFolder(root=r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\train",transform=transformers)
test_dataset= datasets.ImageFolder(root=r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\test",transform=transformers)
train_size= int(0.8*len(train_dataset))
val_size= len(train_dataset) - train_size
train_data,val_data= random_split(train_dataset,[train_size,val_size])
batch_size = 40
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_data,batch_size=batch_size)
test_loader = DataLoader(test_dataset,batch_size=batch_size)
class CNN(nn.Module):
    def  __init__(self):
        super().__init__()
        self.network= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self,x):
        return self.network(x)
model=CNN().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs=20
for i in range (epochs):
    training_loss= 0
    correct=0
    total=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        pred=model(images)
        loss_c=loss_fn(pred,labels)
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        training_loss+=loss_c.item()
        predicted = pred.argmax(dim=1)
        correct += (predicted==labels).sum().item()
    train_loss = training_loss / len(train_loader)
    accuracy = (correct/len(train_data))*100
    print(
    f"Epoch={i+1}/{epochs} | "
    f"train_Loss={train_loss:.3f} | "
    f"Acc={accuracy:.2f}%"
    )
model.eval()
val_loss= []
correct=0
total=0
with torch.no_grad():
    for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device)
        pred=model(images)
        loss_c=loss_fn(pred,labels)
        val_loss.append(loss_c.item())
        predicted = pred.argmax(dim=1)
        correct += (predicted==labels).sum().item()
totalloss= sum(val_loss)/len(val_data)
accuracy = (correct/len(val_loader))*100
print(
    f"\nValidation | "
    f"val_Loss={totalloss:.3f} | "
    f"Acc={accuracy:.2f}%"
)

model.eval()
test_loss= []
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        pred=model(images)
        loss_c=loss_fn(pred,labels)
        test_loss.append(loss_c.item())
        predicted = pred.argmax(dim=1)
        correct += (predicted==labels).sum().item()
testloss= sum(test_loss)/len(test_loader)
accuracy = (correct/len(test_dataset))*100
print(
    f"Epoch={epochs+1} | "
    f"test_Loss={testloss:.3f} | "
    f"Acc={accuracy:.2f}%"
    )
torch.save(model.state_dict(), "cnn.pth")
print("model saved")
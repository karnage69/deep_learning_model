import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
df= pd.read_csv(r"D:\ML\archive (1)\diabetes.csv")
print(df.head())
x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
#converting dataset into float tensor
x_train= torch.FloatTensor(x_train.values)
x_test= torch.FloatTensor(x_test.values)
y_train= torch.LongTensor(y_train.values)
y_test= torch.LongTensor(y_test.values)
class ANN(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_feature=2):
        super().__init__()
        self.fc1= nn.Linear(input_features,hidden1)
        self.fc2= nn.Linear(hidden1,hidden2)
        self.out= nn.Linear(hidden2,out_feature)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x
model= ANN()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=500
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred= model.forward(x_train)
    loss_c= loss(y_pred,y_train)
    final_losses.append(loss_c.item())
    if i%10==1:
        print("epoch no.{} ,loss{}".format(i,loss_c.item()))
    optimizer.zero_grad()
    loss_c.backward()
    optimizer.step()

#plot loss graph
plt.plot(range(epochs),final_losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
##lets predict on test data and one this is that we do not require gradients (wieight,bias) as we are using test data,gradients are only required on train data
model.eval()
with torch.no_grad():
    y_pred=model(x_test)
    y_pred = torch.argmax(y_pred, dim=1)
correct = (y_pred == y_test).sum()
accuracy= correct/len(y_test)
test_loss = loss(model(x_test), y_test).item()
plt.figure(figsize=(10,6))
plt.plot(range(epochs),final_losses,label="Training Loss")
plt.axhline(y=test_loss,label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.show()
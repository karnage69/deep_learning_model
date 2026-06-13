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
from sklearn.preprocessing import StandardScaler,OneHotEncoder
df= pd.read_csv(r"D:\heartapk\heart.csv")
df.drop_duplicates(inplace=True)
x = df.drop('HeartDisease', axis=1)
y=df['HeartDisease']
numerical_col=['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']
cat_cols=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, random_state=42, test_size=0.25)
#numerical data pipeline
numerical_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
#categorical pipeline
cat_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder())
])
##column transformer
preprocessor=ColumnTransformer([
("num",numerical_pipeline,numerical_col),
("cat",cat_pipeline,cat_cols)
])
x_train = preprocessor.fit_transform(x_train)
x_val = preprocessor.transform(x_val)
x_test = preprocessor.transform(x_test)
#converting dataset into float tensor
x_train= torch.FloatTensor(x_train)
x_val = torch.FloatTensor(x_val)
x_test= torch.FloatTensor(x_test)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val = torch.tensor(y_val.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)
class ANN(nn.Module):
    def __init__(self,input_features ,hidden1=20,hidden2=20,out_feature=2):
        super().__init__()
        self.fc1=nn.Linear(input_features,hidden1)
        self.fc2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_feature)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x
input_features = x_train.shape[1]
model = ANN(input_features=input_features)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
final_losses= []
val_losses=[]
epochs=1000
for i in range (epochs):
    y_pred=model(x_train)
    loss_c=loss(y_pred,y_train)
    final_losses.append(loss_c.item())
    if i%10==1:
        print("epoch no.{} ,loss{}".format(i,loss_c.item()))
    optimizer.zero_grad()
    loss_c.backward()
    optimizer.step()
# --- VALIDATION PHASE ---
    model.eval() 
    with torch.no_grad():
        y_pred_val = model(x_val)
        loss_val = loss(y_pred_val, y_val)
        val_losses.append(loss_val.item())
        
        val_preds = torch.argmax(y_pred_val, dim=1)
        val_acc = (val_preds == y_val).sum().item() / len(y_val)
        
    # FIXED: Variable names in print statement corrected
    if i % 100 == 0: 
        print(f"Epoch {i:3d} | Train Loss: {loss_c.item():.4f} | Val Loss: {loss_val.item():.4f} | Val Acc: {val_acc*100:.2f}%")

print("\n--- Training Complete ---")
#on test data
model.eval()
with torch.no_grad():
    y_pred=model(x_test)
    y_pred = torch.argmax(y_pred, dim=1)
correct = (y_pred == y_test).sum()
accuracy= correct/len(y_test)
test_loss = loss(model(x_test), y_test).item()
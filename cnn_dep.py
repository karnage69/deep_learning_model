import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
# MODEL
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64*8*8,128),
            nn.ReLU(),

            nn.Linear(128,2)

        )

    def forward(self,x):
        return self.network(x)

# SAME TRANSFORM AS TRAINING
transform = transforms.Compose([

    transforms.Resize((64,64)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )

])
# LOAD CLASSES FROM IMAGEFOLDER
dataset = datasets.ImageFolder(root=r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\train",transform=transform)
classes = dataset.classes

# LOAD MODEL
model = CNN()
model.load_state_dict(
    torch.load(
        r"C:\Users\mathu\OneDrive\Desktop\deep learning\cnn.pth",
        map_location="cpu"
    )
)
model.eval()
# LOAD IMAGE
image = Image.open(
   r"C:\Users\mathu\OneDrive\Desktop\deep learning\archive (2)\PRC_142435379.webp").convert("RGB")

image = transform(image)

# ADD BATCH DIMENSION
image = image.unsqueeze(0)

# INFERENCE
with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )
# RESULT
predicted_class = prediction.item()

print("\nRaw Output:", output)

print(
    "\nPredicted Index:",
    predicted_class
)

print(
    "\nPredicted Class:",
    classes[predicted_class]
)
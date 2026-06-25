import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# MODEL
class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )

        )

    def forward(
        self,
        x
    ):

        return self.network(x)



# IMAGE TRANSFORM
transform = transforms.Compose([

    transforms.Resize(
        (
            64,
            64
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.5, 0.5, 0.5],

        [0.5, 0.5, 0.5]

    )

])


# CHANGE THESE TO YOUR REAL CLASS NAMES
classes = [

    "class_1",

    "class_2"

]


# LOAD MODEL
model = CNN()

model.load_state_dict(

    torch.load(

        "cnn.pth",

        map_location="cpu"

    )

)

model.eval()



# PREDICTION
def predict(image):

    if not isinstance(
        image,
        Image.Image
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    )

    with torch.no_grad():

        output = model(
            image
        )

        probs = torch.softmax(

            output,

            dim=1

        )

        confidence, prediction = torch.max(

            probs,

            dim=1

        )

    return f"""
Prediction: {classes[prediction.item()]}

Confidence: {round(confidence.item()*100,2)}%
"""
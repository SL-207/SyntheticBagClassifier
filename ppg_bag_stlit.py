import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.io import read_image


def create_model():
    class SyntheticBagModel(nn.Module):
        def __init__(self, input_channels: int, output_features: int, hidden_units: int):
            super().__init__()
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.clf_block = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*16*16, out_features=output_features)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.clf_block(self.conv_block2(self.conv_block1(x)))
    
    model = SyntheticBagModel(input_channels=3, output_features=3, hidden_units=10)    
    return model
    

def main():
    model = create_model()
    load_dict = torch.load('ppg_bag_final_model.pt')
    model.load_state_dict(load_dict)
    class_names = ['Garbage', 'Paper', 'Plastic']
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    st.title('Plastic, Paper, Garbage Bag Classifier')
    images = st.file_uploader(label='Bag Image', accept_multiple_files=True)
    cols = st.columns(3) # tuple of 3 columns
    imageT_list = []
    if images:
        for img in images:
            image = Image.open(img)
            imageT_list.append(data_transforms(image))

        imageT = torch.stack(imageT_list, dim=0)
        model.eval()
        with torch.inference_mode():
            y_probs = model(imageT).softmax(dim=1)
            y_preds = y_probs.argmax(dim=1).squeeze(dim=0)
            
        counter = 0
        try:
            while counter < len(y_preds):
                for col in cols:
                    if counter >= len(y_preds):
                        break
                    pred = y_preds[counter].item()
                    print(pred)
                    col.header(f'{class_names[pred]} bag: {y_probs[counter][pred]:.4f}')
                    col.image(images[counter])
                    counter += 1
        except:
            cols[0].header(f'{class_names[y_preds]} bag: {y_probs[0][y_preds]:.4f}')
            cols[0].image(images[0])

if __name__ == '__main__':
    main()
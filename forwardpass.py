import pickle
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st
import random
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, maps = 64, noise_size = 128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_size, maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(maps * 8),
            nn.ReLU(True),
            # state size. (maps*8) x 4 x 4
            nn.ConvTranspose2d(maps * 8, maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(maps * 4),
            nn.ReLU(True),
            # state size. (maps*4) x 8 x 8
            nn.ConvTranspose2d(maps * 4, maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(maps * 2),
            nn.ReLU(True),
            # state size. (maps*2) x 16 x 16
            nn.ConvTranspose2d(maps * 2, maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(maps),
            nn.ReLU(True),
            # state size. (maps) x 32 x 32
            nn.ConvTranspose2d(maps, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, x):
        return self.model(x)
    
# streamlit app will use this function to generate comic images
def main():
    st.set_page_config(page_title="Landing Page",
                       initial_sidebar_state="collapsed")
    st.title("Comic Generator")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process uploaded image and generate comic
        fake_samples = forward_pass(uploaded_file)
        
        # Display generated comic
        st.image(fake_samples, caption='Generated Comic', use_column_width=True)

def load_process_img(img_path):
    # Load and preprocess the image
    img = Image.open(img_path)

    # Resize the image to match the expected input size of the CNN model
    resize_transform = transforms.Resize((224, 224))
    img_resized = resize_transform(img)

    # Convert the image to a tensor and normalize it
    to_tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize_transform(to_tensor_transform(img_resized)).unsqueeze(0)

    # Load a pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)

    # Set the model to evaluation mode
    resnet.eval()

    # Extract features from the image using the ResNet model
    with torch.no_grad():
        features = resnet(img_tensor)

    # Use a linear layer to reduce the number of channels to 128
    linear_layer = torch.nn.Linear(in_features=features.size(1), out_features=128)
    features_reduced = linear_layer(features.squeeze()).unsqueeze(-1).unsqueeze(-1)

    # Add an extra dimension at the beginning
    features_reduced = features_reduced.unsqueeze(0)

    # Further processing or use of the feature map
    return features_reduced # Output: torch.Size([1, 128, 1, 1])

def forward_pass(img_path):

    # Load generator model
    generator = pickle.load(open('/Users/rashmipanse/Documents/Masters/MSDS631-DeepLearning/Comic-Generator/generator.pkl', 'rb'))

    features_reduced = load_process_img(img_path)

    fake_samples = generator(features_reduced).reshape(-1, 3, 64, 64)

    # return fake_samples as image
    fake_samples_img = Image.fromarray((np.transpose(fake_samples.detach().numpy(), (0, 2, 3, 1))[0] * 255).astype(np.uint8))

    return fake_samples_img

if __name__ == "__main__":
    main()

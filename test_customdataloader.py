# %%
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from CustomSegmentationDataset import CustomSegmentationDataset
import matplotlib.pyplot as plt

transform= transforms.Compose([

    #transforms.Normalize(), # open all images to get std and mean
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.ToPILImage(),
    transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    
])

# Create an instance of the custom test dataset
custom_test_dataset = CustomSegmentationDataset(os.getcwd(), transform=transform,  target_transform=target_transform)

# Create a DataLoader for the test dataset
batch_size = 1  # Use a batch size of 1 for testing to go through patches one at a time
test_dataloader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)
# %%
i=0
for batch in test_dataloader:  # Assuming you have a DataLoader named 'dataloader'
    images, labels = batch

    # Assuming 'images' is a tensor containing the images
    for image, label in zip(images,labels):
        # Convert the PyTorch tensor to a NumPy array
        image_np = image.squeeze()  # Convert from (C, H, W) to (H, W, C)
        label_np = label.squeeze()

        # Display the image using Matplotlib
        plt.imshow(image_np)
        plt.show()
        plt.imshow(label_np)
        plt.show()
        i+=1
        print(i)

# %%

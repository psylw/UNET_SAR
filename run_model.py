# %%
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision import transforms
import torch
from CustomSegmentationDataset import CustomSegmentationDataset
from model import UNet

import torch.optim as optim

# Specify the root directory where your dataset is located
root_dir = os.getcwd()

# get mean and std dev for images
image_folder = os.path.join(root_dir, 'images')
images = os.listdir(image_folder)

im = []
for i in range(len(images)):
    image_name = images[i]
    image = np.load(os.path.join(image_folder,image_name))
    image[np.isnan(image)] = 0
    im.append(image)

mean = np.mean(im)
std = np.std(im)

# define transforms
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std), # open all images to get std and mean
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    
])
# %%
# Split your data into training and testing sets
images_train = images[0:116]
images_test = images[117:]

# Create separate instances for training and testing datasets
train_dataset = CustomSegmentationDataset(
    root_dir=root_dir,
    transform=transform,
    target_transform=target_transform,
    image_files=images_train,
)

test_dataset = CustomSegmentationDataset(
    root_dir=root_dir,
    transform=transform,
    target_transform=target_transform,
    image_files=images_test,
)

# Create DataLoaders for training and testing
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# %%
model = UNet()


# %%
learning_rate = 1e-3
epochs = 5

# Initialize the loss function
def dice_loss(predicted, target):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
    return dice

criterion = dice_loss

# Define an optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
    test_loop(test_dataloader, model, criterion)
print("Done!")
# %%

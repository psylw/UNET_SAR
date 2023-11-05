
import os
import rioxarray as rxr

root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'images')

mask_folder = os.path.join(root_dir, 'masks')

images = os.listdir(image_folder)
masks = os.listdir(mask_folder)


for i in range(len(images)):
    im = images[i]
    t = rxr.open_rasterio(os.path.join(image_folder,im)).sel(band=1).values
    np.save(im.split('.tif')[0]+'.npy', t)


for i in range(len(masks)):
    im = masks[i]
    t = rxr.open_rasterio(os.path.join(mask_folder,im)).sel(band=1).values
    np.save(im.split('.tif')[0]+'.npy', t)

# %%
import numpy as np

# Create a sample NumPy array
original_array = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]])

# Define the cropping region
# In this example, we're cropping from row 1 to 2 (exclusive) and from column 1 to 3 (exclusive)
cropped_array = original_array[1:2, 1:3]

print(cropped_array)
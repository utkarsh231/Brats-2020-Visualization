# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps 

# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
!pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif 
import gif_your_nifti.core as gif2nif


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
#from tensorflow.keras.layers.experimental import preprocessing


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include

TRAIN_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')

# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')

# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')

shutil.copy2(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii', './test_gif_BraTS20_Training_001_flair.nii')
gif2nif.write_gif_normal('./test_gif_BraTS20_Training_001_flair.nii')

from IPython.display import Image

# Path to your GIF file
gif_path = '/kaggle/working/test_gif_BraTS20_Training_001_flair.gif'

# Display the GIF
Image(filename=gif_path)

import cv2
import imageio

# Path to your GIF file
gif_path = '/kaggle/working/test_gif_BraTS20_Training_001_flair.gif'  # Adjust this path accordingly

# Convert GIF to Video
video_path = '/kaggle/working/test_gif_BraTS20_Training_001_flair.mp4'

# Get GIF reader
gif_reader = imageio.get_reader(gif_path)

# Get first frame to get dimensions
frame = gif_reader.get_data(0)
height, width, _ = frame.shape

# Manually set frame rate (adjust as needed)
fps = 10  # Set your desired frame rate here

# Create video writer
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Write each frame to video
for frame in gif_reader:
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Release video writer
video_writer.release()

# Display the video in a Kaggle notebook
from IPython.display import Video
Video(video_path, embed=True)

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Define segmentation classes
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}

# Set the parameters for volume slices
VOLUME_SLICES = 100
VOLUME_START_AT = 22

# Set the dataset paths
TRAIN_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# Load a sample volume (flair, t1, t1ce, t2, seg)
flair_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii'
t1_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii'
t1ce_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii'
t2_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii'
seg_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii'

test_image_flair = nib.load(flair_path).get_fdata()
test_image_t1 = nib.load(t1_path).get_fdata()
test_image_t1ce = nib.load(t1ce_path).get_fdata()
test_image_t2 = nib.load(t2_path).get_fdata()
test_mask = nib.load(seg_path).get_fdata()

# Define a function to visualize the images and segmentation
def visualize_slice(image_data, slice_index, title, cmap='gray'):
    plt.imshow(image_data[:, :, slice_index], cmap=cmap)
    plt.title(title)
    plt.axis('off')

# Display the image slices
slice_w = 25
slice_index = test_image_flair.shape[0] // 2 - slice_w

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
ax1.imshow(test_image_flair[:, :, slice_index], cmap='gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:, :, slice_index], cmap='gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:, :, slice_index], cmap='gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:, :, slice_index], cmap='gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:, :, slice_index])
ax5.set_title('Mask')

plt.show()

# Function to create a montage of slices
def create_montage(volume, start_idx, end_idx):
    slices = volume[start_idx:end_idx, :, :]
    montage = np.concatenate([slices[i] for i in range(slices.shape[0])], axis=1)
    return montage

# Display montage of slices
montage_image = create_montage(test_image_t1, 50, test_image_t1.shape[0]-50)

fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(rotate(montage_image, 90), cmap='gray')
plt.show()

# Function to segment and display the regions
def display_segmented_slice(slice_data, seg_data, slice_index):
    slice_img = slice_data[:, :, slice_index]
    seg_img = seg_data[:, :, slice_index]
    
    # Define regions
    background = (slice_img == 0)
    foreground = (slice_img > 0) & (seg_img == 0)
    tumor_area = (seg_img > 0)
    surrounding_region = (foreground) & (~tumor_area)
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(background, cmap='gray')
    ax[0].set_title('Background')
    
    ax[1].imshow(foreground, cmap='gray')
    ax[1].set_title('Foreground')
    
    ax[2].imshow(tumor_area, cmap='gray')
    ax[2].set_title('Tumor Area')
    
    ax[3].imshow(surrounding_region, cmap='gray')
    ax[3].set_title('Surrounding Region')
    
    plt.show()

# Display the segmented slice
display_segmented_slice(test_image_flair, test_mask, slice_index)


from scipy.ndimage import binary_dilation

# Set the parameters for volume slices
VOLUME_SLICES = 100
VOLUME_START_AT = 22

# Set the dataset paths
TRAIN_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# Load a sample volume (flair, t1, t1ce, t2, seg)
flair_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii'
t1_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii'
t1ce_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii'
t2_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii'
seg_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii'

test_image_flair = nib.load(flair_path).get_fdata()
test_image_t1 = nib.load(t1_path).get_fdata()
test_image_t1ce = nib.load(t1ce_path).get_fdata()
test_image_t2 = nib.load(t2_path).get_fdata()
test_mask = nib.load(seg_path).get_fdata()

# Define a function to visualize the images and segmentation
def visualize_slice(image_data, slice_index, title, cmap='gray'):
    plt.imshow(image_data[:, :, slice_index], cmap=cmap)
    plt.title(title)
    plt.axis('off')

# Display the image slices
slice_w = 25
slice_index = test_image_flair.shape[0] // 2 - slice_w

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
ax1.imshow(test_image_flair[:, :, slice_index], cmap='gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:, :, slice_index], cmap='gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:, :, slice_index], cmap='gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:, :, slice_index], cmap='gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:, :, slice_index])
ax5.set_title('Mask')

plt.show()

# Function to create a montage of slices
def create_montage(volume, start_idx, end_idx):
    slices = volume[start_idx:end_idx, :, :]
    montage = np.concatenate([slices[i] for i in range(slices.shape[0])], axis=1)
    return montage

# Display montage of slices
montage_image = create_montage(test_image_t1, 50, test_image_t1.shape[0]-50)

fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(np.rot90(montage_image), cmap='gray')
plt.show()

# Function to segment and display the regions
def display_segmented_slice(slice_data, seg_data, slice_index):
    slice_img = slice_data[:, :, slice_index]
    seg_img = seg_data[:, :, slice_index]
    
    # Define regions
    background = (slice_img == 0)
    foreground = (slice_img > 0) & (seg_img == 0)
    tumor_area = (seg_img > 0)
    
    # Dilate the tumor area to get the surrounding region
    surrounding_region = binary_dilation(tumor_area, iterations=5) & ~tumor_area
    
    # Create a color image to display the regions
    color_img = np.zeros((*slice_img.shape, 3))
    
    # Set colors for each region
    color_img[background] = [0, 0, 0]          # Black for background
    color_img[foreground] = [0.5, 0.5, 0.5]    # Gray for foreground
    color_img[tumor_area] = [1, 0, 0]          # Red for tumor area
    color_img[surrounding_region] = [0, 1, 0]  # Green for surrounding region
    
    plt.imshow(color_img)
    plt.title('Segmented Regions')
    plt.axis('off')
    plt.show()

# Display the segmented slice
display_segmented_slice(test_image_flair, test_mask, slice_index)


import os
import h5py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['text.color'] = '#DDDDDD'

def load_nifti_file(file_path):
    return nib.load(file_path).get_fdata()

def display_image_channels(image, title='Image Channels'):
    channel_names = ['T1-weighted (T1)', 'T1-weighted post contrast (T1c)', 'T2-weighted (T2)', 'Fluid Attenuated Inversion Recovery (FLAIR)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(image[idx, :, :], cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
    plt.show()

def display_mask_channels_as_rgb(mask, title='Mask Channels as RGB'):
    channel_names = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    plt.show()

def overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0, :, :]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.zeros_like(rgb_image)
    color_mask[mask[0, :, :] > 0, 0] = 1  # Red for Necrotic
    color_mask[mask[1, :, :] > 0, 1] = 1  # Green for Edema
    color_mask[mask[2, :, :] > 0, 2] = 1  # Blue for Tumour
    overlay_image = np.where(color_mask > 0, color_mask, rgb_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_image)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')
    plt.show()

# Load the sample image and mask
flair_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii'
t1_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii'
t1ce_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii'
t2_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii'
seg_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii'

flair = load_nifti_file(flair_path)
t1 = load_nifti_file(t1_path)
t1ce = load_nifti_file(t1ce_path)
t2 = load_nifti_file(t2_path)
mask = load_nifti_file(seg_path)

# Select a slice index
slice_index = 75

# Prepare the image and mask data
image = np.stack([t1[:, :, slice_index], t1ce[:, :, slice_index], t2[:, :, slice_index], flair[:, :, slice_index]])
mask = np.stack([(mask[:, :, slice_index] == 1).astype(np.uint8), (mask[:, :, slice_index] == 2).astype(np.uint8), (mask[:, :, slice_index] == 3).astype(np.uint8)])

# View images using plotting functions
display_image_channels(image)
display_mask_channels_as_rgb(mask)
overlay_masks_on_image(image, mask)


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

# Load the MRI volume data
flair_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii'
t1_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii'
t1ce_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii'
t2_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii'
seg_path = TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii'

flair_data = nib.load(flair_path).get_fdata()
t1_data = nib.load(t1_path).get_fdata()
t1ce_data = nib.load(t1ce_path).get_fdata()
t2_data = nib.load(t2_path).get_fdata()
seg_data = nib.load(seg_path).get_fdata()

# Function to create a 3D MRI GIF
def create_mri_gif(volume_data, seg_data, output_file='mri_slices.gif', cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.tight_layout()

    frames = []
    for i in range(volume_data.shape[2]):
        axes[0].clear()
        axes[1].clear()
        
        # Display MRI slice
        axes[0].imshow(volume_data[:, :, i], cmap=cmap)
        axes[0].set_title('MRI Slice {}'.format(i))
        axes[0].axis('off')
        
        # Display Segmentation mask overlay
        axes[1].imshow(volume_data[:, :, i], cmap=cmap)
        axes[1].imshow(seg_data[:, :, i], alpha=0.3, cmap='viridis')  # Adjust alpha for transparency
        axes[1].set_title('Segmentation Overlay')
        axes[1].axis('off')
        
        # Append current frame to frames list
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    # Save frames as GIF using imageio
    imageio.mimsave(output_file, frames, duration=0.1)

# Example usage
create_mri_gif(flair_data, seg_data)


from IPython.display import Image

# Path to your GIF file
gif_path2 = '/kaggle/working/mri_slices.gif'  

# Display the GIF
Image(filename=gif_path2)


import cv2
import imageio

# Path to your GIF file
gif_path2 = '/kaggle/working/mri_slices.gif'  # Adjust this path accordingly

# Convert GIF to Video
video_path2 = '/kaggle/working/mri_slices.mp4'

# Get GIF reader
gif_reader = imageio.get_reader(gif_path2)

# Get first frame to get dimensions
frame = gif_reader.get_data(0)
height, width, _ = frame.shape

# Manually set frame rate (adjust as needed)
fps = 10  # Set your desired frame rate here

# Create video writer
video_writer = cv2.VideoWriter(video_path2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Write each frame to video
for frame in gif_reader:
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Release video writer
video_writer.release()

# Display the video in a Kaggle notebook
from IPython.display import Video
Video(video_path2, embed=True)

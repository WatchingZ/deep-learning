import torch
import torchvision

import matplotlib.pyplot as plt
import random


def generate_samples(dataset, n_samples: int) -> tuple:
  """
  Generates a list of samples from the given dataset randomly
  
  Args: 
  dataset (Any): dataset where samples will be extracted (must be unpackable)
  n_samples (int): amount of samples to be extracted from dataset

  Returns:
  A tuple consisting of random samples extracted from the dataset with a length of n_samples along with their corresponding label
  """
  
  samples = ()

  for sample, label in random.sample(list(dataset), k=n_samples):
    samples.append((sample, label))

  return samples

def plot_images(dataset, rows: int, columns: int, figsize: tuple, cmap: str) -> None:
  """
  Plots a certain amount of images from the given dataset

  Args:
  dataset (Any): dataset in which images are stored (must be unpackable, if length does not equal rows * columns, the first rows * columns images of the dataset will be plotted)
  rows (int): how many rows of images the figure will consist of 
  columns (int): how many columns of images the figure will consist of
  figsize (tuple): what the resulting figure size of the figure will be
  cmap (str): what the color maps of the plotted images will be

  Returns:
  A matplotlib plot consisting of rows * columns images
  """
  
  plt.figure(figsize=figsize)

  for i in range(rows * columns):
    image, label = dataset[i]
    
    plt.subplot(rows, columns, i+1)
    plt.title(label)
    plt.imshow(image.squeeze(), cmap=cmap)
  plt.show();

  

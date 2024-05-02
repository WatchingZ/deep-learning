import torch
import matplotlib.pyplot as plt

import random

from torch import nn

def generate_samples(dataset, n_samples: int) -> list:
    """
    Generates a list of samples from the given dataset randomly
    
    Args: 
    dataset (Any): dataset where samples will be extracted (must be unpackable)
    n_samples (int): amount of samples to be extracted from dataset
  
    Returns:
    A list consisting of random samples extracted from the dataset with a length of n_samples along with their corresponding label (can be unpacked)
    """
    
    samples = []
  
    for sample, label in random.sample(list(dataset), k=n_samples):
      samples.append((sample, label))
  
    return samples

def plot_images(dataset, rows: int, columns: int, figsize: tuple, cmap=None, title=True, fontsize=10) -> None:
    """
    Plots a certain amount of images from the given dataset
  
    Args:
    dataset (Any): dataset in which images are stored (must be unpackable, if length does not equal rows * columns, the first rows * columns images of the dataset will be plotted)
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of
    figsize (tuple): what the resulting figure size of the figure will be
    cmap (str, optional): what the color map of the plotted images will be
    title (bool, optional): whether or not a title will be displayed for each image (defaults to True)
    fontsize (int, optional): font size of the title (defaults to 10)
  
    Returns:
    A matplotlib plot consisting of rows * columns images
    """
    
    plt.figure(figsize=figsize)
  
    for i in range(rows * columns):
      image, label = dataset[i]
      
      plt.subplot(rows, columns, i+1)
  
      if title:
          plt.title(dataset.classes[label], fontsize=fontsize)
      plt.imshow(image.squeeze(), cmap=cmap)
    plt.show();

def plot_image_predictions(predictions, dataset, rows: int, columns: int, figsize: tuple, classes: dict, fontsize=10, cmap=None) -> None:
    """ Plots the image being predicted on and the models prediction on it 
  
    Args:
    predictions (Any): the model's predictions on the dataset (must be in probability form)
    dataset (Any): The dataset the model predicted on
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of 
    figsize (tuple): the resulting figure size of the plot
    classes (dict): the classes of the dataset
    fontsize (int, optional): the fontsize of the resulting plot (default is 10)
    cmap (str, optional): the colormap of the resulting images
  
    Returns:
    A matplotlib plot of all images and their respective predictions
    """
    
    plt.figure(figsize=figsize)
    
    for i in range(rows * columns):
      plt.subplot(rows, columns, i+1)
  
      image, label = dataset[i]
      prediction = predictions[i]
  
      title = f"Predicted: {classes[prediction.argmax()]}: {prediction.max(): .2f} | Truth: {classes[label]}"
      color = "green" if classes[prediction.argmax()] == classes[label] else "red"
  
      plt.title(title, fontsize=fontsize, c=color)
      plt.imshow(image.squeeze(), cmap=cmap)
  
    plt.show();
  

def plot_loss(results: dict, name: str, figsize=(15, 7)) -> None:
    """
    Plots a loss curve
  
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"epoch": [...],
             "loss": [...],}
        name (str): the resulting name of the plt
        figsize (tuple, optional): what the resulting figure size of the plot will be (defaults to (15, 7))
        
    Returns:
    A matplotlib plot of the loss curves
    """
  
    losses = results["loss"]
    epochs = results["epoch"]
  
    plt.plot(epochs, losses, label="Loss")
    plt.title(name)
    plt.xlabel("Epochs")
    plt.legend()
  
    plt.show();

def plot_linear_predictions(X: torch.Tensor, y: torch.Tensor, predictions=None, colors=['r','g'], figsize=(10, 7)) -> None:
    """
    Plots the linear predictions of a certain model
  
    Args:
    X (torch.Tensor): the data inputed through the model
    y (torch.Tensor): the labels of the data given
    predictions (torch.Tensor, optional): The predictions of the model
    colors (list, optional): The colors of each of scatter plots (defaults to [r, g])
    figsize (tuple, optional): The figure size of the returned plot
  
    Returns:
    A Matplotlib scatter plot consisting of the data, labels and the models predictions
    """
    plt.figure(figsize=figsize)
  
    plt.scatter(X, y, c=colors[0], s=4, label="Data")
    
    if predictions is not None:
        plt.scatter(X, predictions, c=colors[1], s=4, label="Predictions")
  
    plt.legend(prop={"size": 14})


def make_predictions(model: torch.nn.Module, dataset, device: str):
    """
    Makes predictions using the given model on a given dataset
  
    Args: 
    model (torch.nn.Module): the model predicting on the given dataset
    dataset (Any): the dataset in which the model will be predicting on
    device (str): specifies what device the data should be on
  
    Returns:
    A tensor of predictions from the model (in raw logit form)
    """
    model.eval()
  
    with torch.inference_mode():
      predictions = []
      
      for X, y in dataset:
        X = X.to(device)
        
        prediction = model(X)
  
        predictions.append(prediction)
  
    return torch.cat(predictions)

def train(epochs: int, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader, device:str, test_dataloader=None, train=True, test=True, return_loss_plot=False):
    """
    Performs a training & optional test loop over an iterable dataset with the given model

    Args:
    epochs (int): how many iterations the model will be trained for
    model (torch.nn.Module): the model being trained
    loss_fn (torch.nn.Module): the loss function being used for training
    optimizer (torch.optim.Optimizer): the optimizer for training the model
    train_dataloader (torch.utils.data.DataLoader): the dataloader used when training the model for the training loop
    test_dataloader (torch.utils.data.DataLoader, optional): dataloader being used for the training loop (optional but highly recommended)
    device (str): the desired device for the data that the model will be trained on (must be the same as the device of the model)
    train (bool, True): if a training loop will be used (is always True unless specifically said so)
    test (bool, True): if a testing loop will be used (is always True unless specifically said so, when set to True a test_dataloader must be used)

    Returns:

    None
    
    """
    
    train_results = {"epoch":[], "loss":[]}
    test_results = {"epoch":[], "loss":[]}
    
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch} \n----------")
        train_loss = 0

        if train:
            for batch, (X, y) in enumerate(train_dataloader):
                model.train()
    
                X, y = X.to(device), y.to(device)
    
                y_logits = model(X)
                loss = loss_fn(y_logits, y)
    
                optimizer.zero_grad()
    
                loss.backward()
    
                optimizer.step()
    
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            train_results["epoch"].append(epoch)
            train_results["loss"].append(train_loss)
            
            print(f"Train Loss: {train_loss: .5f}")

        if test:       
            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(device), y.to(device)

                    model.eval()

                    y_logits = model(X)
                    loss = loss_fn(y_logits, y)

                    test_loss += loss
                test_loss /= len(test_dataloader)

                test_results["epoch"].append(epoch)
                test_results["loss"].append(test_loss)
                

                print(f"Test Loss: {test_loss: .5f}")
                
    if return_loss_plot:
        if train:
            plot_loss(results=train_results, name='Train Loss')
        if test:
            plot_loss(results=test_results, name='Test Loss')

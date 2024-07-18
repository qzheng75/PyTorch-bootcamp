import gif
import numpy as np
import matplotlib.pyplot as plt

import torch

@gif.frame
def plot_frame(X, y, model, epoch, title=None):
  plt.figure(figsize=(10, 6))
  plt.scatter(X, y, color='blue')
  X_numpy = X.cpu().numpy()
  y_numpy = y.cpu().numpy()
  x_lim = np.array([X_numpy.min(), X_numpy.max()])
  y_lim = np.array([y_numpy.min(), y_numpy.max()])
  with torch.no_grad():
    model.eval()
    ypred = model(X).cpu().numpy()
  plt.plot(X_numpy, ypred, color='red')
  if title is None:
    title = f'Epoch {epoch + 1}'
  plt.xlim(left=x_lim[0] - 0.2, right=x_lim[1] + 0.2)
  plt.ylim(bottom=y_lim[0] - 0.2, top=y_lim[1] + 0.2)
  plt.title(title)
  plt.xlabel('X')
  plt.ylabel('Y')

@gif.frame
def plot_losses(train_losses, val_losses, epoch):
  plt.figure(figsize=(10, 6))
  plt.plot(train_losses, label='Training Loss', color='blue')
  plt.plot(val_losses, label='Validation Loss', color='red')
  plt.title('Training and Validation Losses')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  max_epochs = len(train_losses)
  plt.xlim(0, max_epochs)
  
  y_min = min(min(train_losses), min(val_losses))
  y_max = max(max(train_losses), max(val_losses))
  plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

  # Add text to show current epoch and loss values
  current_epoch = len(train_losses) - 1
  plt.text(0.02, 0.95, f'Epoch: {epoch}', transform=plt.gca().transAxes)
  plt.text(0.02, 0.90, f'Train Loss: {train_losses[-1]:.4f}', transform=plt.gca().transAxes)
  plt.text(0.02, 0.85, f'Val Loss: {val_losses[-1]:.4f}', transform=plt.gca().transAxes)
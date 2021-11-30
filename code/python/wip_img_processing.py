import os
import re
import argparse
from pandas.core.frame import DataFrame
import skimage.measure
import cv2 as cv
import numpy as np
import pandas as pd
from minisom import MiniSom
from fnmatch import fnmatch
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score


# Needs generalization
files = []
pattern = "*r.jpg.jpg"  # Search for recto of all folios
main_path = "D:\SktClaraRes"
for (dirpath, dirnames, filenames) in os.walk(main_path):
    for name in filenames:
        if fnmatch(name, pattern):
            files.append(os.path.join(dirpath, name))


# Loads images from list of paths
def read_images(paths):
    images = [cv.imread(i, 0) for i in paths] # Read each path in array-like object into list of grey-scale images
    return images


def entropy_patches(image,  entropy_limit, patch_size=10):
    # _, binary = cv.threshold(image, 10, 255, cv.THRESH_BINARY_INV)
    patches = []
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Follows contours, extracting every coordinate of contours
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour) # Finds each letter by bounding contours
        square = image[x:x + patch_size, y:y + patch_size] # Extracts each letter, but maintains a specific patch size
        if skimage.measure.shannon_entropy(square) > entropy_limit: # Keeps patches iff they carry enough information according to Shannon
            square = square.flatten() # Flattens the images to 1D so they are prepared as direct input to convolutional operations
            patches.append(square)
    return patches


def soms(patches, neurons, width, height, sigma=0.1, learning_rate=0.2): # width and height has no default, only because we want separate one-call full-functionality functions and modifiable functions
    som = MiniSom(x=width, y=height, input_len=neurons, sigma=sigma, learning_rate=learning_rate) # Defines the neural network (Kohonen self-organizing map)
    som.random_weights_init(patches) # Initialize weights
    starting_weights = som.get_weights().copy() # Copy the weights to prepare initial 
    som.train_random(patches, neurons)
    qnt = som.quantization(patches)
    return qnt


def train_test_split(data, test_percent):
    split_indices =  np.random.random_integers(0, range(len(data)))
    train = split_indices(min(range(split_indices)), range(len(split_indices)) * 1 - test_percent)
    test = split_indices(range(len(split_indices))*test_percent, max(range(len(split_indices))))
    return train, test

def full_auto_train(starting_input_path): # Takes only input of 
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(starting_input_path):
        for name in filenames:
                paths.append(os.path.join(dirpath, name))
    images = read_images(paths)

import os
import re
import argparse
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


files = []
pattern = "*r.jpg.jpg"  # Search for recto of all folios
main_path = "D:\SktClaraRes"
for (dirpath, dirnames, filenames) in os.walk(main_path):
    for name in filenames:
        if fnmatch(name, pattern):
            files.append(os.path.join(dirpath, name))


# Loads images from list of paths
def read_images(paths):
    images = [cv.imread(i, 0) for i in paths]
    return images


def entropy_patches(image, patch_size, neurons):
    # _, binary = cv.threshold(image, 10, 255, cv.THRESH_BINARY_INV)
    patches = []
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        square = image[x:x + patch_size, y:y + patch_size]
        if skimage.measure.shannon_entropy(square) > 2:
            square = square.flatten()
            patches.append(square)

    som = MiniSom(x=10, y=10, input_len=neurons, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(patches)
    starting_weights = som.get_weights().copy()
    som.train_random(patches, neurons)
    qnt = som.quantization(patches)
    return qnt




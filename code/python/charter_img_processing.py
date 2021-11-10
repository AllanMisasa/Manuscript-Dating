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
from swtloc import SWTLocalizer
from run_lengths_encoding import rle
from classification import svc, svr
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, StratifiedShuffleSplit
from skimage.feature import hog, corner_fast, corner_peaks, corner_orientations, corner_subpix, corner_harris
from skimage.morphology import octagon

from operator import itemgetter

data = pd.read_csv("meta.txt", delimiter='|', skipinitialspace=True)
data.columns = data.columns.str.replace(' ', '')
data['fasc.'] = data['fasc.'].str.strip()
data = data[data['year'].notna()]


def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def image_paths():
    files = []
    pattern = "*r.jpg.jpg"  # Search for recto of all folios
    main_path = "D:\SktClaraRes"
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, pattern):
                files.append(os.path.join(dirpath, name))
    return files


def get_targets(paths):
    integers = []
    doc_ids = []
    for path in paths:
        i = path[-16:]
        integer = int(re.search(r'\d+', i).group())
        n = path[-20:-9]
        roman = re.findall('[LXVI]', n)
        roman = ''.join(roman)
        facsid = int(data.loc[(data['fasc.'] == roman) & (data['num.'] == integer)]['number'])  # row where document is.
        year = int(data.loc[(data['fasc.'] == roman) & (data['num.'] == integer)]['year'])
        integers.append(year)
        doc_ids.append(facsid)
    return integers, doc_ids


def bin_years(y):
    binned_years = []
    for year in y:
        if 1250 <= year < 1300:
            binned_years.append(1250)
        if 1300 <= year < 1350:
            binned_years.append(1300)
        if 1350 <= year < 1400:
            binned_years.append(1350)
        if 1400 <= year < 1450:
            binned_years.append(1400)
        elif 1450 <= year < 1500:
            binned_years.append(1450)
        elif 1500 <= year < 1550:
            binned_years.append(1500)
        elif year >= 1550:
            binned_years.append(1550)
    return binned_years


paths = image_paths()
years, docids = get_targets(paths)
binned_years = bin_years(years)


def swt_edge_load_images():
    swt = []
    edge = []
    edge_pattern = "edge*"  # Search for recto of all folios
    swt_pattern = "swt*"
    main_path = "D:/swtres/"
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, edge_pattern):
                edge.append(os.path.join(dirpath, name))
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, swt_pattern):
                swt.append(os.path.join(dirpath, name))
    return swt, edge


swt_paths, edge_paths = swt_edge_load_images()


def extract_features(edge_paths, swt_paths):
    edge_features = []
    swt_features = []
    for edge_image in edge_paths:  # can be omitted
        edge = cv.imread(edge_image, 0)
        edge_features.append(edge)
    for swt_image in swt_paths:
        swt = cv.imread(swt_image, 0)
        swt_features.append(swt)
    features = pd.DataFrame(zip(edge_features, swt_features))
    return features


image_features = extract_features(edge_paths, swt_paths)


def entropy_patches(image_features):
    Y = []
    years = []
    codebook = []
    doc_wise = []
    # codebook1250 = []
    # codebook1300 = []
    # codebook1350 = []
    # codebook1400 = []
    # codebook1450 = []
    # codebook1500 = []
    # codebook1550 = []
    count = 0
    doc = []
    for swt in image_features[1]:
        # _, binary = cv.threshold(swt, 10, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(swt, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            square = swt[x:x + 10, y:y + 10]
            if skimage.measure.shannon_entropy(square) > 2:
                square = square.flatten()
                year = binned_years[count]
                # codebook.append([square, year])
                if year == 1250:
                    codebook.append(square)
                    Y.append(1250)
                if year == 1300:
                    codebook.append(square)
                    Y.append(1300)
                if year == 1350:
                    codebook.append(square)
                    Y.append(1350)
                if year == 1400:
                    codebook.append(square)
                    Y.append(1400)
                if year == 1450:
                    codebook.append(square)
                    Y.append(1450)
                if year == 1500:
                    codebook.append(square)
                    Y.append(1500)
                if year == 1550:
                    codebook.append(square)
                    Y.append(1550)
                else:
                    continue
        doc.append(codebook)
        y = np.mean(Y)
        years.append(y)
        codebook = []
        Y = []
        count += 1
    return np.array(doc), np.array(years)


# return np.array(codebook1250), np.array(codebook1300), np.array(codebook1350), np.array(codebook1400),
# np.array(codebook1450), np.array(codebook1500), np.array(codebook1550), np.array(y)
# codebook1250, codebook1300, codebook1350, codebook1400, codebook1450, codebook1500, codebook1550,
# y = entropy_patches(image_features)
X, y = entropy_patches(image_features)

'''
train1250, test1250, = train_test_split(codebook1250, test_size=0.2, random_state=42)
train1300, test1300, = train_test_split(codebook1300, test_size=0.2, random_state=42)
train1350, test1350, = train_test_split(codebook1350, test_size=0.2, random_state=42)
train1400, test1400, = train_test_split(codebook1400, test_size=0.2, random_state=42)
train1450, test1450, = train_test_split(codebook1450, test_size=0.2, random_state=42)
train1500, test1500, = train_test_split(codebook1500, test_size=0.2, random_state=42)
train1550, test1550, = train_test_split(codebook1550, test_size=0.2, random_state=42)
'''

# unique, counts = np.unique(y, return_counts=True)
# bleh = dict(zip(unique, counts))
# sorted_years_indices = y.argsort()
# sorted_years = y[sorted_years_indices]
# sorted_X = X[sorted_years_indices]


def soms(codebook):
    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(codebook)
    starting_weights = som.get_weights().copy()
    som.train_random(codebook, 100)
    qnt = som.quantization(codebook)
    return qnt


def tag_docs(subsets):
    doc_nest = []
    for subset in subsets:
        length = len(subset)
        doc_nest.append(length)
    return doc_nest


def arrange():
    indices1250 = np.where(y == 1250)
    indices1300 = np.where(y == 1300)
    indices1350 = np.where(y == 1350)
    indices1400 = np.where(y == 1400)
    indices1450 = np.where(y == 1450)
    indices1500 = np.where(y == 1500)
    indices1550 = np.where(y == 1550)

    q1250 = X[indices1250]
    q1300 = X[indices1300]
    q1350 = X[indices1350]
    q1400 = X[indices1400]
    q1450 = X[indices1450]
    q1500 = X[indices1500]
    q1550 = X[indices1550]
    # concatenated arrays of shape n patches x 100.
    n1250 = np.concatenate(q1250)
    n1300 = np.concatenate(q1300)
    n1350 = np.concatenate(q1350)
    n1400 = np.concatenate(q1400)
    n1450 = np.concatenate(q1450)
    n1500 = np.concatenate(q1500)
    n1550 = np.concatenate(q1550)

    x1250 = soms(n1250)
    x1300 = soms(n1300)
    x1350 = soms(n1350)
    x1400 = soms(n1400)
    x1450 = soms(n1450)
    x1500 = soms(n1500)
    x1550 = soms(n1550)
    quantized_x = np.vstack((x1250, x1300, x1350, x1400, x1450, x1500, x1550))
    conc_y = [1250] * len(n1250) + [1300] * len(n1300) + [1350] * len(n1350) + [1400] * len(n1400) + [1450] * len(
        n1450) + [1500] * len(n1500) + [1550] * len(n1550)
    return quantized_x, conc_y


# Creates an array of the same length of the concatenated features, showing which document the patch belongs to
def index_arrays():
    my_arrays = [np.array(x) for x in X]
    index_arrays = [np.ones(x.shape, dtype=int) * i for i, x in enumerate(my_arrays)]
    index_array = np.concatenate(index_arrays)
    l = [int(np.mean(xi)) for xi in index_array]
    return l


# Use the index array to shuffle the features on a document level
def custom_train_test_split(features, targets, array_of_indices):
    indices = []
    possibilities = np.unique(array_of_indices)  # All the permutations
    choices = np.random.choice(possibilities, 86)  # Picking 20% of the documents at random
    for i in range(len(possibilities)):
        inds = np.where(array_of_indices == possibilities[i])[0]
        indices.append(inds)
    indices = np.array(indices)
    test_indices = indices[choices]
    test_targets = targets[choices]
    train_indices = np.delete(indices, choices)
    train_targets = np.delete(targets, choices)
    return train_indices, test_indices, train_targets, test_targets


def som_quantize(Xs, ys, year):
    indis = np.where(ys == year)
    XS = Xs[indis]
    XS = np.concatenate(XS)
    data[XS] = soms(data[XS])


doc_indices = index_arrays()

# Prepare index arrays for train/test split
Xtrain, Xtest, ytrain, ytest = custom_train_test_split(0, y, doc_indices)

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Prepare the data that is indexed
data, conc_y = arrange()
conc_y = np.asarray(conc_y)

# for i in np.unique(y):
#     som_quantize(data, conc_y, i)

# Quantize training data independently and separately from test data.
# for i in np.unique(y):
#     som_quantize(Xtrain, ytrain, i)  # Quantize training data
#     som_quantize(Xtest, ytest, i)  # Quantize test data

train_features = data[np.concatenate(Xtrain)]
train_targets = conc_y[np.concatenate(Xtrain)]

# X_train, X_test, y_train, y_test = train_test_split(data, conc_y, test_size=0.2, random_state=42)
# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, fit_intercept=True, class_weight='balanced', multi_class='ovr', max_iter=5000)
# clf = RandomForestClassifier(random_state=42)
clf = LogisticRegression(fit_intercept=True, multi_class='ovr', n_jobs=-1, random_state=42)
clf.fit(train_features, train_targets)


def predict():
    predictions = []
    for vector in Xtest:
        test_vector = data[vector]
        vector_predictions = clf.predict(test_vector)
        predictions.append(vector_predictions)
    return predictions


pred = predict()
final_predictions = []
for i in pred:
    candidate = np.bincount(i).argmax()
    final_predictions.append(candidate)

print('Accuracy: ', accuracy_score(ytest, final_predictions))
print('F1 score: ', f1_score(ytest, final_predictions, average='weighted'))


def originals():
    indices1250 = np.where(y == 1250)
    indices1300 = np.where(y == 1300)
    indices1350 = np.where(y == 1350)
    indices1400 = np.where(y == 1400)
    indices1450 = np.where(y == 1450)
    indices1500 = np.where(y == 1500)
    indices1550 = np.where(y == 1550)

    q1250 = X[indices1250]
    q1300 = X[indices1300]
    q1350 = X[indices1350]
    q1400 = X[indices1400]
    q1450 = X[indices1450]
    q1500 = X[indices1500]
    q1550 = X[indices1550]
    # concatenated arrays of shape n patches x 100.
    n1250 = np.concatenate(q1250)
    n1300 = np.concatenate(q1300)
    n1350 = np.concatenate(q1350)
    n1400 = np.concatenate(q1400)
    n1450 = np.concatenate(q1450)
    n1500 = np.concatenate(q1500)
    n1550 = np.concatenate(q1550)

    quantized_x = np.vstack((n1250, n1300, n1350, n1400, n1450, n1500, n1550))
    conc_y = [1250] * len(n1250) + [1300] * len(n1300) + [1350] * len(n1350) + [1400] * len(n1400) + [1450] * len(
        n1450) + [1500] * len(n1500) + [1550] * len(n1550)
    return quantized_x, conc_y


Xx, yy = originals()


def comparison(patch, quantized_patch):
    patch.shape = (10, 10)
    quantized_patch.shape = (10, 10)
    return patch, quantized_patch


n += 10
a, b = comparison(Xx[n], data[n])
plt.imshow(b, 'gray')

# predictions = clf.predict(X_test)
# clf.fit(X_train, y_train)  # .decision_function(X_test)
# print('Accuracy: ', accuracy_score(y_test, predictions))
# print('F1 score: ', f1_score(y_test, predictions, average='weighted'))

# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# scores = cross_val_score(clf, data, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
# print('Weighted F1 score averaged over 10 folds: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Logistic regression
# Accuracy:  0.7965295270081164
# F1 score:  0.7895171413047696
# Weighted F1 score averaged over 10 folds: 0.788 (0.004)

# SVC With intercept, max_iter=5000
# Accuracy: 0.723
# F1 Score: 0.730
# Weighted F1 score averaged over 10 folds 0.688 (0.053)
# Scores of folds 0.67947359 0.60726653 0.76001223 0.62534808 0.7183125  0.64988554
# 0.66293596 0.78099526 0.71681328 0.68318524

# SVC No intercept, max_iter=5000
# Acc 0.76945
# F1 score: 0.76309
# Weighted F1 score averaged over 10 folds: 0.663 (0.079)

# Random forest
# Accuracy:  0.8041561712846348
# F1 score:  0.7985359534963257
# Weighted F1 score averaged over 10 folds: 0.799 (0.004)


'''
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
from swtloc import SWTLocalizer
from run_lengths_encoding import rle
from classification import svc, svr
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, StratifiedShuffleSplit
from skimage.feature import hog, corner_fast, corner_peaks, corner_orientations, corner_subpix, corner_harris
from skimage.morphology import octagon


from operator import itemgetter

data = pd.read_csv("meta.txt", delimiter='|', skipinitialspace=True)
data.columns = data.columns.str.replace(' ', '')
data['fasc.'] = data['fasc.'].str.strip()
data = data[data['year'].notna()]


def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def image_paths():
    files = []
    pattern = "*r.jpg.jpg"  # Search for recto of all folios
    main_path = "D:\SktClaraRes"
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, pattern):
                files.append(os.path.join(dirpath, name))
    return files


def get_targets(paths):
    integers = []
    for path in paths:
        i = path[-16:]
        integer = int(re.search(r'\d+', i).group())
        n = path[-20:-9]
        roman = re.findall('[LXVI]', n)
        roman = ''.join(roman)
        year = int(data.loc[(data['fasc.'] == roman) & (data['num.'] == integer)]['year'])
        integers.append(year)
    return integers


def bin_years(y):
    binned_years = []
    for year in y:
        if 1250 <= year < 1300:
            binned_years.append(1250)
        if 1300 <= year < 1350:
            binned_years.append(1300)
        if 1350 <= year < 1400:
            binned_years.append(1350)
        if 1400 <= year < 1450:
            binned_years.append(1400)
        elif 1450 <= year < 1500:
            binned_years.append(1450)
        elif 1500 <= year < 1550:
            binned_years.append(1500)
        elif year >= 1550:
            binned_years.append(1550)
    return binned_years


paths = image_paths()
years = get_targets(paths)
binned_years = bin_years(years)


def swt_edge_load_images():
    swt = []
    edge = []
    edge_pattern = "edge*"  # Search for recto of all folios
    swt_pattern = "swt*"
    main_path = "D:/swtres/"
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, edge_pattern):
                edge.append(os.path.join(dirpath, name))
    for (dirpath, dirnames, filenames) in os.walk(main_path):
        for name in filenames:
            if fnmatch(name, swt_pattern):
                swt.append(os.path.join(dirpath, name))
    return swt, edge


swt_paths, edge_paths = swt_edge_load_images()


def extract_features(edge_paths, swt_paths):
    edge_features = []
    swt_features = []
    for edge_image in edge_paths:  # can be omitted
        edge = cv.imread(edge_image, 0)
        edge_features.append(edge)
    for swt_image in swt_paths:
        swt = cv.imread(swt_image, 0)
        swt_features.append(swt)
    features = pd.DataFrame(zip(edge_features, swt_features))
    return features


image_features = extract_features(edge_paths, swt_paths)


def entropy_patches(image_features):
    Y = []
    years = []
    codebook = []
    count = 0
    doc = []
    for swt in image_features[1]:
        # _, binary = cv.threshold(swt, 10, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(swt, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            square = swt[x:x+10, y:y+10]
            if skimage.measure.shannon_entropy(square) > 2:
                square = square.flatten()
                year = binned_years[count]
                # codebook.append([square, year])
                if year == 1250:
                    codebook.append(square)
                    Y.append(1250)
                if year == 1300:
                    codebook.append(square)
                    Y.append(1300)
                if year == 1350:
                    codebook.append(square)
                    Y.append(1350)
                if year == 1400:
                    codebook.append(square)
                    Y.append(1400)
                if year == 1450:
                    codebook.append(square)
                    Y.append(1450)
                if year == 1500:
                    codebook.append(square)
                    Y.append(1500)
                if year == 1550:
                    codebook.append(square)
                    Y.append(1550)
                else:
                    continue
        doc.append(np.asarray(codebook))
        y = np.mean(Y)
        years.append(y)
        codebook = []
        Y = []
        count += 1
    return np.asarray(doc), np.array(years)


X, y = entropy_patches(image_features)


def soms(codebook):
    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(codebook)
    starting_weights = som.get_weights().copy()
    som.train_random(codebook, 100)
    qnt = som.quantization(codebook)
    return qnt


def q(x):
    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(x)
    starting_weights = som.get_weights().copy()
    som.train_random(x, 100)
    return som.quantization(x)


def maps(n, indices):
    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(n)
    starting_weights = som.get_weights().copy()
    som.train_random(n, 100)
    # for i in X[indices]:
    #    X[i] = som.quantization(i)
    X[indices] = [q(xi) for xi in X[indices]]


def arrangeX():
    indices1250 = np.where(y == 1250)
    indices1300 = np.where(y == 1300)
    indices1350 = np.where(y == 1350)
    indices1400 = np.where(y == 1400)
    indices1450 = np.where(y == 1450)
    indices1500 = np.where(y == 1500)
    indices1550 = np.where(y == 1550)

    q1250 = X[indices1250]
    q1300 = X[indices1300]
    q1350 = X[indices1350]
    q1400 = X[indices1400]
    q1450 = X[indices1450]
    q1500 = X[indices1500]
    q1550 = X[indices1550]
    # concatenated arrays of shape n patches x 100.
    n1250 = np.concatenate(q1250)
    n1300 = np.concatenate(q1300)
    n1350 = np.concatenate(q1350)
    n1400 = np.concatenate(q1400)
    n1450 = np.concatenate(q1450)
    n1500 = np.concatenate(q1500)
    n1550 = np.concatenate(q1550)

    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(n1250)
    starting_weights = som.get_weights().copy()
    som.train_random(n1250, 100)
    for i in X[indices1250]:
        X[i] = som.quantization(i)
    som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
    som.random_weights_init(n1300)
    starting_weights = som.get_weights().copy()
    som.train_random(n1250, 100)
    for i in X[indices1250]:
        X[i] = som.quantization(i)

    maps(n1250, indices1250)
    maps(n1300, indices1300)
    maps(n1350, indices1350)
    maps(n1400, indices1400)
    maps(n1450, indices1450)
    maps(n1500, indices1500)
    maps(n1550, indices1550)


    X[indices1250] = [q(xi) for xi in q1250]
    X[indices1300] = [q(xi) for xi in q1300]
    X[indices1350] = [q(xi) for xi in q1350]
    X[indices1400] = [q(xi) for xi in q1400]
    X[indices1450] = [q(xi) for xi in q1450]
    X[indices1500] = [q(xi) for xi in q1500]
    X[indices1550] = [q(xi) for xi in q1550]


arrangeX()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_vector():
    indices1250 = np.where(y_train == 1250)
    indices1300 = np.where(y_train == 1300)
    indices1350 = np.where(y_train == 1350)
    indices1400 = np.where(y_train == 1400)
    indices1450 = np.where(y_train == 1450)
    indices1500 = np.where(y_train == 1500)
    indices1550 = np.where(y_train == 1550)

    q1250 = X_train[indices1250]
    q1300 = X_train[indices1300]
    q1350 = X_train[indices1350]
    q1400 = X_train[indices1400]
    q1450 = X_train[indices1450]
    q1500 = X_train[indices1500]
    q1550 = X_train[indices1550]
    q1250 = np.concatenate(q1250)
    q1300 = np.concatenate(q1300)
    q1350 = np.concatenate(q1350)
    q1400 = np.concatenate(q1400)
    q1450 = np.concatenate(q1450)
    q1500 = np.concatenate(q1500)
    q1550 = np.concatenate(q1550)
    xs_train = np.vstack((q1250, q1300, q1350, q1400, q1450, q1500, q1550))
    ys_train = [1250] * len(q1250) + [1300] * len(q1300) + [1350] * len(q1350) + [1400] * len(q1400) + [1450] * len(q1450) + [1500] * len(q1500) + [1550] * len(q1550)
    return xs_train, ys_train

def length(docs):
    l = []
    for i in docs:
        le = len(i)
        l.append(le)
    return l

def test_vector():
    indices1250 = np.where(y_test == 1250)
    indices1300 = np.where(y_test == 1300)
    indices1350 = np.where(y_test == 1350)
    indices1400 = np.where(y_test == 1400)
    indices1450 = np.where(y_test == 1450)
    indices1500 = np.where(y_test == 1500)
    indices1550 = np.where(y_test == 1550)

    q1250 = X_test[indices1250]
    q1300 = X_test[indices1300]
    q1350 = X_test[indices1350]
    q1400 = X_test[indices1400]
    q1450 = X_test[indices1450]
    q1500 = X_test[indices1500]
    q1550 = X_test[indices1550]
    len1250 = length(q1250)
    len1300 = length(q1300)
    len1350 = length(q1350)
    len1400 = length(q1400)
    len1450 = length(q1450)
    len1500 = length(q1500)
    len1550 = length(q1550)
    q1250 = np.concatenate(q1250)
    q1300 = np.concatenate(q1300)
    q1350 = np.concatenate(q1350)
    q1400 = np.concatenate(q1400)
    q1450 = np.concatenate(q1450)
    q1500 = np.concatenate(q1500)
    q1550 = np.concatenate(q1550)
    doc_lengths = len1250 + len1300 + len1350 + len1400 + len1450 + len1500 + len1550
    xs_test = np.vstack((q1250, q1300, q1350, q1400, q1450, q1500, q1550))
    ys_test = [1250] * len(q1250) + [1300] * len(q1300) + [1350] * len(q1350) + [1400] * len(q1400) + [1450] * len(q1450) + [1500] * len(q1500) + [1550] * len(q1550)
    return xs_test, ys_test, doc_lengths


X_train, y_train = train_vector()
X_test, y_test, doc_lengths = test_vector()
# clf = RandomForestClassifier(random_state=42)
clf = LogisticRegression(fit_intercept=True, multi_class='ovr', n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions, average='weighted'))


def vote(doc):
    prediction = clf.predict(doc)
    return prediction

votes = []
for i in X_test:
    v = (vote(i))
    votes.append(v)


def max_voting():
    votes = []
    predictions = []
    for doc in X_test:
        for i in doc:
            vote = clf.predict(i)
            votes.append(vote)
    for v in votes:
        prediction = np.bincount(v).argmax()
        predictions.append(prediction)
    return predictions

# clf = LogisticRegression(fit_intercept=True, multi_class='ovr', n_jobs=-1, random_state=42)


# predictions = max_voting()

# print('Accuracy: ', accuracy_score(y_test, predictions))
# print('F1 score: ', f1_score(y_test, predictions, average='weighted'))



# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, fit_intercept=True, class_weight='balanced', multi_class='ovr', max_iter=5000)
clf = RandomForestClassifier(random_state=42)
# clf = LogisticRegression(fit_intercept=True, multi_class='ovr', n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
clf.fit(X_train, y_train)  # .decision_function(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions, average='weighted'))

# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# scores = cross_val_score(clf, data, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
# print('Weighted F1 score averaged over 10 folds: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Logistic regression
# Accuracy:  0.7965295270081164
# F1 score:  0.7895171413047696
# Weighted F1 score averaged over 10 folds: 0.788 (0.004)

# SVC With intercept, max_iter=5000
# Accuracy: 0.723
# F1 Score: 0.730
# Weighted F1 score averaged over 10 folds 0.688 (0.053)
# Scores of folds 0.67947359 0.60726653 0.76001223 0.62534808 0.7183125  0.64988554
# 0.66293596 0.78099526 0.71681328 0.68318524

# SVC No intercept, max_iter=5000
# Acc 0.76945
# F1 score: 0.76309
# Weighted F1 score averaged over 10 folds: 0.663 (0.079)

# Random forest
# Accuracy:  0.8041561712846348
# F1 score:  0.7985359534963257
# Weighted F1 score averaged over 10 folds: 0.799 (0.004)

som1250 = soms(train1250)
som1300 = soms(train1300)
som1350 = soms(train1350)
som1400 = soms(train1400)
som1450 = soms(train1450)
som1500 = soms(train1500)
som1550 = soms(train1550)

tests1250 = soms(test1250)


train1250, test1250, = train_test_split(codebook1250, test_size=0.2, random_state=42)
train1300, test1300, = train_test_split(codebook1300, test_size=0.2, random_state=42)
train1350, test1350, = train_test_split(codebook1350, test_size=0.2, random_state=42)
train1400, test1400, = train_test_split(codebook1400, test_size=0.2, random_state=42)
train1450, test1450, = train_test_split(codebook1450, test_size=0.2, random_state=42)
train1500, test1500, = train_test_split(codebook1500, test_size=0.2, random_state=42)
train1550, test1550, = train_test_split(codebook1550, test_size=0.2, random_state=42)

# for pattern, year in codes:
    # tests['all']

# predictions, y_test = svc(X, binned_years)
clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, fit_intercept=True, class_weight='balanced', multi_class='ovr', max_iter=100000)
# random_forest = RandomForestClassifier(random_state=42)
# random_forest.fit(X_train, y_train)
# predictions = random_forest.predict(X_test)
clf.fit(X_train, y_train)  # .decision_function(X_test)
predictions = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions, average='weighted'))



img = image_features[1][0]
_, binary = cv.threshold(img, 10, 255, cv.THRESH_BINARY_INV)
# corners = corner_peaks(corner_harris(img, 9), min_distance=1)
# orientations = corner_orientations(img, corners, octagon(3, 2))
# print(np.rad2deg(orientations))


# return np.array(codebook1250), np.array(codebook1300), np.array(codebook1350), np.array(codebook1400),
# np.array(codebook1450), np.array(codebook1500), np.array(codebook1550), np.array(y)
# codebook1250, codebook1300, codebook1350, codebook1400, codebook1450, codebook1500, codebook1550,
# y = entropy_patches(image_features)
contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# draw all contours
x,y,w,h = cv.boundingRect(contours[5])
rectangle = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
image = cv.drawContours(img, contours, -1, (0, 127, 0), 1)

def entropy_patches(image_features):
    r = 4  # radius of window
    best_patch = []
    histograms = []
    best_patches = []
    for swt in image_features[1]:
        img = cv.adaptiveThreshold(swt, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        corners = corner_peaks(corner_harris(img, 9), min_distance=1)
        for corner in corners:
            try:
                window = img[corner[0]-r+1:corner[0]+r, corner[1]-r+1:corner[1]+r]
            except:
                pass
            
            
            
            
        # orientations = corner_orientations(img, corners, octagon(3, 2))
        # radians = orientations[-800:]
        # best_patches.append(radians)
    return best_patches


best_patches = entropy_patches(image_features)
best_patches = np.vstack(best_patches)
X_train, X_test, y_train, y_test = train_test_split(best_patches, binned_years, test_size=0.2, random_state=42)

clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, fit_intercept=False, class_weight='balanced', multi_class='ovr', max_iter=100000)
# random_forest = RandomForestClassifier(random_state=42)
# random_forest.fit(X_train, y_train)
# predictions = random_forest.predict(X_test)
clf.fit(X_train, y_train)  # .decision_function(X_test)
predictions = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions, average='weighted'))



fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(corners[:, 1], corners[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=1)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 310, 200, 0))
plt.show()

def hogs(image_features):
    hogs = []
    for img in image_features[1]:
        hist = hog(img, orientations=9, feature_vector=True, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=False, multichannel=None)
        hist = hist[hist != 0]
        # hist = hist.reshape(-1, 1)
        # hist = hist.tolist()
        hogs.append(hist[0:11000])
    return hogs


best_patches = hogs(image_features)

# X = normalize(X, norm='l2')
# X = scaler.fit_transform(X) Worked horrendously. F1 score of 0.0329. Definitely because 0 counts are also normalized.

X_train, X_test, y_train, y_test = train_test_split(best_patches, binned_years, test_size=0.2, random_state=42)

# predictions, y_test = svc(X, binned_years)
clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, fit_intercept=True, class_weight='balanced', multi_class='ovr', max_iter=100000)
# random_forest = RandomForestClassifier(random_state=42)
# random_forest.fit(X_train, y_train)
# predictions = random_forest.predict(X_test)
clf.fit(X_train, y_train)  # .decision_function(X_test)
predictions = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions, average='weighted'))
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(clf, best_patches, binned_years, scoring='f1_weighted', cv=cv, n_jobs=-1)
print('Weighted F1 score averaged over 10 folds: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))





# bah = np.array([q(xi) for xi in quantize])



def training():
    unique, counts = np.unique(y, return_counts=True)
    bleh = dict(zip(unique, counts))
    data1250 = X[0:bleh[1250]]
    X1250 = soms(data1250)
    data1300 = X[bleh[1250]:bleh[1300]+bleh[1250]]
    X1300 = soms(data1300)
    data1350 = X[bleh[1250]+bleh[1300]:bleh[1350]+bleh[1300]+bleh[1250]]
    X1350 = soms(data1350)
    data1400 = X[bleh[1250]+bleh[1300]+bleh[1350]:bleh[1400]+bleh[1350]+bleh[1300]+bleh[1250]]
    X1400 = soms(data1400)
    data1450 = X[bleh[1250]+bleh[1300]+bleh[1350]+bleh[1400]:bleh[1450]+bleh[1400]+bleh[1350]+bleh[1300]+bleh[1250]]
    X1450 = soms(data1450)
    data1500 = X[bleh[1250]+bleh[1300]+bleh[1350]+bleh[1400]+bleh[1450]:bleh[1500]+bleh[1450]+bleh[1400]+bleh[1350]+bleh[1300]+bleh[1250]]
    X1500 = soms(data1500)
    data1550 = X[bleh[1250]+bleh[1300]+bleh[1350]+bleh[1400]+bleh[1450]+bleh[1500]:bleh[1550]+bleh[1500]+bleh[1450]+bleh[1400]+bleh[1350]+bleh[1300]+bleh[1250]]
    X1550 = soms(data1550)
    x = np.vstack((X1250, X1300, X1350, X1400, X1450, X1500, X1550))
    return x


data = training()



unique, counts = np.unique(y, return_counts=True)

bleh = dict(zip(unique, counts))
sorted_years_indices = y.argsort()
sorted_years = y[sorted_years_indices]
sorted_X = X[sorted_years_indices]
data1250 = X[0:bleh[1250]]
data1300 = X[bleh[1250]:bleh[1300] + bleh[1250]]
data1350 = X[bleh[1250] + bleh[1300]:bleh[1350] + bleh[1300] + bleh[1250]]
data1400 = X[bleh[1250] + bleh[1300] + bleh[1350]:bleh[1400] + bleh[1350] + bleh[1300] + bleh[1250]]
data1450 = X[bleh[1250] + bleh[1300] + bleh[1350] + bleh[1400]:bleh[1450] + bleh[1400] + bleh[1350] + bleh[1300] + bleh[
    1250]]
data1500 = X[bleh[1250] + bleh[1300] + bleh[1350] + bleh[1400] + bleh[1450]:bleh[1500] + bleh[1450] + bleh[1400] + bleh[
    1350] + bleh[1300] + bleh[1250]]
data1550 = X[bleh[1250] + bleh[1300] + bleh[1350] + bleh[1400] + bleh[1450] + bleh[1500]:bleh[1550] + bleh[1500] + bleh[
    1450] + bleh[1400] + bleh[1350] + bleh[1300] + bleh[1250]]

data1250 = np.concatenate(data1250, axis=0)
data1300 = np.concatenate(data1300, axis=0)
data1350 = np.concatenate(data1350, axis=0)
data1400 = np.concatenate(data1400, axis=0)
data1450 = np.concatenate(data1450, axis=0)
data1450 = np.concatenate(data1500, axis=0)
data1450 = np.concatenate(data1550, axis=0)

som = MiniSom(x=10, y=10, input_len=100, sigma=0.1, learning_rate=0.2)
som.random_weights_init(data1250)
starting_weights = som.get_weights().copy()
som.train_random(data1250, 100)

def entropy_patches(image_features):
    best_patch = []
    histograms = []
    best_patches = []
    for swt in image_features[1]:
        patches = extract_patches_2d(swt, (16, 16), max_patches=1000)
        entropies = [skimage.measure.shannon_entropy(patch) for patch in patches]
        meta_lst = list(enumerate(entropies))
        sorted_meta_lst = sorted(meta_lst, key=itemgetter(1), reverse=True)
        for i in range(500):
            img = sorted_meta_lst[i][0]
            # print(img)
            best_patch.append(patches[img])
        for image in best_patch:
            # image = cv.equalizeHist(image)
            corners = corner_peaks(corner_harris(image, 9), min_distance=1)
            orientations = corner_orientations(image, corners, octagon(3, 2))
            # fd = hog(image, orientations=9, feature_vector=True, pixels_per_cell=(4, 4),
            #                    cells_per_block=(2, 2), visualize=False, multichannel=None)
            # hist = cv.calcHist([image], [0], None, [10], [0, 255])
            # cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            # hister = list(chain.from_iterable(fd))
            fd = orientations[0:2]
            histograms.append(fd)
        hists = np.concatenate(histograms)
        hists = hists[0:300]
        best_patches.append(hists)
        best_patch = []
        histograms = []
    return best_patches


Commented out because we only need to create the SWTs once.

def swt(paths):
    c = 1
    swtl = SWTLocalizer()
    for i in paths:
        img = cv.imread(i)
        swt = swtl.swttransform(image=img, save_results=True, save_rootpath='swtres/',
              edge_func = 'ac', ac_sigma = 1.0, text_mode = 'lb_df',
              gs_blurr=True, blurr_kernel = (5,5), minrsw = 3,
              maxCC_comppx = 10000, maxrsw = 200, max_angledev = np.pi/6,
              acceptCC_aspectratio = 5.0)
        new_name_swt = 'D:/swtres/swt_img' + str(c) + '.png'
        new_name_edge = 'D:/swtres/edge_img' + str(c) + '.png'
        img_swt = cv.imread('swtres/Trasnformed_Result/swtpruned3C_img.png')
        img_edge = cv.imread('swtres/Trasnformed_Result/edge_img.png')
        cv.imwrite(new_name_swt, img_swt)
        cv.imwrite(new_name_edge, img_edge)
        c += 1


swt(paths)



# Obsolete - merged into get_images()
def otsu(gray_image):
    blur = cv.gaussianblur(gray_image, (3, 3), 0)
    ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th



def runs(otsu):
    rot_otsu = cv.rotate(otsu, cv.ROTATE_90_CLOCKWISE)
    h_runs = np.array(rle(otsu))
    v_runs = np.array(rle(rot_otsu))
    h_runs_black = sum(h_runs[1::2])/(len(h_runs)/2)
    h_runs_white = sum(h_runs[0::2])/(len(h_runs)/2)
    v_runs_black = sum(v_runs[1::2])/(len(v_runs)/2)
    v_runs_white = sum(v_runs[0::2])/(len(v_runs)/2)
    return h_runs_black, h_runs_white, v_runs_black, v_runs_white


def get_images(paths):
    images = []
    h_runs_black = []
    h_runs_white = []
    v_runs_black = []
    v_runs_white = []
    for image in paths:
        img = cv.imread(image, 0)
        blur = cv.GaussianBlur(img, (3, 3), 0)
        ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        one, two, three, four = runs(th)
        images.append(th)
        h_runs_black.append(one)
        h_runs_white.append(two)
        v_runs_black.append(three)
        v_runs_white.append(four)
    return images, h_runs_black, h_runs_white, v_runs_black, v_runs_white


# blur = cv.GaussianBlur(images[400], (3, 3), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# box, area, centroid = cv.connectedComponentsWithStats(image_features[1][1])
img = image_features[1][1]
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--connectivity", type=int, default=4, help="connectivity for connected component analysis")
args = vars(ap.parse_args())
output = cv.connectedComponentsWithStats(img, args["connectivity"], cv.CV_32S)
(numLabels, labels, stats, centroids) = output


mask = np.zeros(img.shape, dtype="uint8")

def extract_connected_components(img, numLabels, labels, stats, centroids):
    comps = []
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        mask = np.zeros(img.shape, dtype="uint8")

        # ensure the width, height, and area are all neither too small
        # nor too big
        keepWidth = w > 5 and w < 300
        keepHeight = h > 5 and h < 100
        keepArea = area > 500 and area < 3000
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepHeight, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv.bitwise_or(mask, componentMask)
            comps.append(componentMask)
            if len(comps) > 10:
                comps = comps[0:9]
    return comps


components = []
for image in image_features[1]:
    output = cv.connectedComponentsWithStats(image, args["connectivity"], cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    comp = extract_connected_components(image, numLabels, labels, stats, centroids)
    components.append(comp)

Takes a list of images as input. Makes 200 random samples of 10x10 patches and calculates the Shannon
    entropy of each patch. The patches with the highest entropies are chosen and are returned.

def entropy_patches(image_features):
    best_patch = []
    histograms = []
    best_patches = []
    for swt in image_features[1]:
        patches = extract_patches_2d(swt, (8, 8), max_patches=1000)
        entropies = [skimage.measure.shannon_entropy(patch) for patch in patches]
        meta_lst = list(enumerate(entropies))
        sorted_meta_lst = sorted(meta_lst, key=itemgetter(1), reverse=True)
        for i in range(500):
            img = sorted_meta_lst[i][0]
            # print(img)
            best_patch.append(patches[img])
        for image in best_patch:
            # image = cv.equalizeHist(image)
            fd = hog(image, orientations=9, feature_vector=True, pixels_per_cell=(4, 4),
                                cells_per_block=(2, 2), visualize=False, multichannel=None)
            # hist = cv.calcHist([image], [0], None, [10], [0, 255])
            # cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            # hister = list(chain.from_iterable(fd))
            histograms.append(fd)
        hists = np.concatenate(histograms)
        best_patches.append(hists)
        best_patch = []
        histograms = []
    return best_patches


best_patches = entropy_patches(image_features)


def entropy_patches(image_features):
    best_patch = []
    histograms = []
    best_patches = []
    for swt in image_features[1]:
        hogs = hog(swt, orientations=9, feature_vector=True, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=False, multichannel=None)
        # patches = extract_patches_2d(hogs, (8, 8), max_patches=1000)
        patches = np.split(hogs, 36)
        entropies = [skimage.measure.shannon_entropy(patch) for patch in patches]
        meta_lst = list(enumerate(entropies))
        sorted_meta_lst = sorted(meta_lst, key=itemgetter(1), reverse=True)
        for i in range(20):
            img = sorted_meta_lst[i]
        # print(img)
            best_patch.append(patches[img[0]])

        entropies = []

        # entropies = [skimage.measure.shannon_entropy(patch) for patch in patches]
        # meta_lst = list(enumerate(entropies))
        # sorted_meta_lst = sorted(meta_lst, key=itemgetter(1), reverse=True)
        # for i in range(500):
        #    img = sorted_meta_lst[i][0]
            # print(img)
        #    best_patch.append(patches[img])
        # for image in best_patch:
            # image = cv.equalizeHist(image)
        #     fd = hog(image, orientations=9, feature_vector=True, pixels_per_cell=(4, 4),
           #                     cells_per_block=(2, 2), visualize=False, multichannel=None)
            # hist = cv.calcHist([image], [0], None, [10], [0, 255])
            # cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            # hister = list(chain.from_iterable(fd))
        #    histograms.append(fd)
        #hists = np.concatenate(histograms)
        # best_patches.append(hists)
        best = np.concatenate(best_patch)
        best_patches.append(best)
        # histograms = []
    return best_patches


best_patches = entropy_patches(image_features)
'''
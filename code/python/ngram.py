# -*- coding:utf-8 -*-
# This code analyzes N-Grams of the lemmata in the Danish charters

import sys
import nltk
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import numpy as np
import os
import re
import csv

scope = 20
n = 3

# Find relative paths
sys.path.append(os.path.realpath('../..'))
directory = sys.path[-1] + '/transcriptions/org/working/'

print("Reading Files: ", end="")


# The following function reads the texts
def read_files_into_string(filenames):
    strings = []
    for filename in filenames:
        working_file = directory + filename
        data = open(working_file, encoding='utf-8').read()
        mo = re.search("Transcription\n.*\n(\n|$)", data, re.S)
        mytable = mo.group(0)
        d = csv.DictReader(mytable.splitlines(), delimiter="|")
        for row in d:
            if row[None][0].strip() == "w":  # rows containing word will be kept
                lemma = row[None][7].strip()  # 7th column is the facsimile level transcription - take those as lemmas
                strings.append(lemma)
    return '\n'.join(strings)


texts = {}
files = []
y1250 = []
y1300 = []
y1350 = []
y1400 = []
y1450 = []
y1500 = []
y1550 = []

for filename in os.listdir(directory):
    if filename.endswith(".org"):
        files.append(filename)
        with open(directory + filename, 'r', encoding='utf8') as this_file:
            for line in this_file:
                if "year-min" in line:
                    full_line = line.split()
                    year = full_line[3]
                    year = year[0:4]
                    if 1250 <= int(year) < 1300:
                        y1250.append(filename)
                    if 1300 <= int(year) < 1350:
                        y1300.append(filename)
                    if 1350 <= int(year) < 1400:
                        y1350.append(filename)
                    if 1400 <= int(year) < 1450:
                        y1400.append(filename)
                    elif 1450 <= int(year) < 1500:
                        y1450.append(filename)
                    elif 1500 <= int(year) < 1550:
                        y1500.append(filename)
                    elif int(year) >= 1550:
                        y1550.append(filename)

texts["all"] = files
texts["y1250"] = y1250
texts["y1300"] = y1300
texts["y1350"] = y1350
texts["y1400"] = y1400
texts["y1450"] = y1450
texts["y1500"] = y1500
texts["y1550"] = y1550

text_types = ["all", "y1250", "y1300", "y1350", "y1400", "y1450", "y1500", "y1550"]
year_types = ["y1250", "y1300", "y1350", "y1400", "y1450", "y1500", "y1550"]

# Half-centuries
# 1250-1249
# 1300-1349
# 1350-1399
# 1400-1449
# 1450-1499
# 1500-1549
# 1550-1599

for text_type, files in texts.items():
    texts[text_type] = read_files_into_string(files)
print("Complete")

# Build Frequency Distributions
print("Building Frequency Distributions: ", end="")
tokens = {'total': []}
grams = {'total': []}
dist = {'total': []}
tokenizer = RegexpTokenizer(r'\w+')

for item in text_types:
    tokens[item] = tokenizer.tokenize(texts[item])
    grams[item] = ngrams(tokens[item], n)


for item in text_types:
    tokens[item] = tokenizer.tokenize(texts[item])
    tokens['total'].extend(tokens[item])
    grams[item] = ngrams(tokens[item], n)
    dist[item] = nltk.FreqDist(grams[item])
#    for key, count in dist[item].most_common(20):
#        print(key, count)

texts_for_grams = texts['all']

'''
# Useful for if any trailing newlines persist
def remove_trailing(t):
    text = t.replace("\n", " ")
    return text


to_write = remove_trailing(texts_for_grams)
'''

'''


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.


# Histogram with half-century bins
def histo():
    
    x = ['1250-1299', '1300-1349', '1350-1399', '1400-1449', '1450-1499', '1500-1549', '1550-1599']
    count = [len(y1250), len(y1300), len(y1350), len(y1400), len(y1450), len(y1500), len(y1550)]
    freq_series = pd.Series(count)
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_xlabel('Half-century')
    ax.set_ylabel('Count')
    ax.set_title('Number of transcriptions for every half-century')
    ax.set_xticklabels(x)
    add_value_labels(ax)


histo()

'''

'''
# Graph N-Grams
control = "all"
#for text_group in text_types:
#    x = [ ]
#    for key in dist[control].most_common(scope):
#        label = ""
#        for item in range(0, len(key[0])):
#            label = label + key[0][item] + ' '
#        label = label + ' (' + str(key[1]) + ')'
#        x.append(label)
#    y = [ ]
#    for key in dist[control].most_common(scope):
#        if key[0] in dist[text_group]:
#            frequency = dist[text_group].freq(key[0]) * 100
#        else:
#            frequency = 0
#        y.append(frequency)
#    x = np.array(x)
#    y = np.array(y)
#    plt.plot(x,y,label=text_group)

#plt.legend()
#plt.title("N-Grams")
#plt.ylabel("Frequency (%)")
#plt.xticks(ha="right", rotation = 45)
#plt.tight_layout()
#plt.show()
        
# Attempt to graph backwards
traces = []
for key in dist[control].most_common(20):
    print(key[0])
    x = []
    y = []
    for years in year_types:
        x.append(years)
        if key[0] in dist[years]:
            frequency = dist[years].freq(key[0]) * 100
        else:
            frequency = 0
        y.append(frequency)
    trace = plotly.graph_objs.Scatter(
        x=x,
        y=y,
        name=str(key[0]),
        mode="lines+markers"
        )
    traces.append(trace)


layout = dict(title="Rise and fall of N-Grams by half-century", xaxis_title="Years", yaxis_title="Frequency (%)")

fig = plotly.graph_objs.Figure(data=traces, layout=layout)
plotly.offline.plot(fig)
'''
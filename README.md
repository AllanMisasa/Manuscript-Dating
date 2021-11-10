# Multi-Modal Manuscript Dating
## Multimodal manuscript dating of the St. Clara corpus - code for a Master Thesis Project
Work in progress; will be updated to work with custom inputs.
Takes as input a list of documents containing transcriptions of charters (but can be any type of document in principle), as well as stroke-width transformed images of handwritten documents.
* Performs TF-IDF vectorization on the input documents, and returns each document represented by TF-IDF vectors.
* Identifies each handwritten character and extracts them as square 32x32 images - each charter is then represented as x character images.
* Performs SOM Quantization per Kohonen's method for implicit clustering and processing of each character image, on a per-document basis.
* SVM classifies each image and casts a vote per image - resulting in x votes. Majority vote becomes the uni-modal image prediction for the images
* SVM classifies each TF-IDF vector matrix to make a prediction per document
* Random forest learns under which conditions the different classifiers predict best, and then makes a final prediction.

Compared to simple TF-IDF classification with SVM, this approach yielded an 10-fold accuracy increase from 74.6% to over 90% on the St. Clara Guldkorpus.

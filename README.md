# cellClassifier
ML model to classify cells based on gene expression vectors 

keras_tests.ipynb is the important file, showing fetching / processing the data, how to compile a keras model for training, the training and evaluation of the model, and various other parts of the workflow (This will be modularized / cleaned up later).

test_model.h5 is a saved keras model that performed okay-ish on the data (it's a little overfit, getting about 85% training accuracy but only 75% accuracy on the test set)

models.py is a module with functions for generating keras models with related / similar architectures.

utils.py is a module with some utilities I wrote ad-hoc in this workflow

low_level_tf_implementation has a bunch of junk using tensorflow instead of the high level Keras API, I'll reimplement the final architecture using tensorflow once its finalized.

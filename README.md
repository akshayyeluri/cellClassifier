# cellClassifier
We have here a deep-learning model to classify cells based on single cell RNA (scRNA) sequencing data. More specifically, our model inputs a gene expression vector listing the counts of mRNA for each gene in a cell, and outputs one of 10 cell types it is predicted the cell belongs too.

---
## Overview of Files
Keras_tests.ipynb is the important file, showing fetching / processing the data, how to compile a keras model for training, the training and evaluation of the model, and various other parts of the workflow (This will be modularized / cleaned up later).

test_model.h5 is a saved keras model that performed well, getting about 80% testing accuracy, which is exciting given that this is a multi-class classification problem with 10 classes

models.py is a module with functions for generating keras models with related / similar architectures.

utils.py is a module with some utilities I wrote ad-hoc in this workflow

low_level_tf_implementation is a work in progress, but will soon be the same architecture, implemented in tensorflow for an increase in sustainability / speed

---
## Model summary
```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 19232, 1, 16)      176       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4808, 1, 16)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4808, 1, 32)       5152      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1202, 1, 32)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1201, 1, 64)       4160      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 300, 1, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 300, 1, 128)       41088     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 100, 1, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 12800)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               6554112   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 6,609,818
Trainable params: 6,609,818
Non-trainable params: 0
_________________________________________________________________
```

---
## Performance Metrics
Here we include an example of loading our model and evaluating a test set with it
```python
from keras.models import load_model
model = load_model("test_model.h5")
loss, acc = model.evaluate(test_data, test_labels)
print(f"Test loss: {loss}")
print(f"Test accuracy: {acc}")
```
    2000/2000 [==============================] - 0s 240us/step
    Test loss: 0.6971381943225861
    Test accuracy: 0.802

```
preds = model.predict(test_data)
conf = utils.confusions(preds, test_labels)
utils.plot_confusions(conf);
```

![test_confusions](assets/confusions.png)

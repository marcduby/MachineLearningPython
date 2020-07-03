# imports
import kipoi
import kipoiseq
import torch
import pybedtools

# load the model
model = kipoi.get_model('Basset')

# make the prediction
prediction = model.pipeline.predict_example()



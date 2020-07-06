# imports
import kipoi
import kipoiseq
import torch
import pybedtools

# load the model
model = kipoi.get_model('Basset')

# make the prediction
prediction = model.pipeline.predict_example()

# print
print("the prediction of type {} and shape {} is \n{}".format(type(prediction), prediction.shape, prediction))



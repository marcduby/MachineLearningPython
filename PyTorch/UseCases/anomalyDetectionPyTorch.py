
# imports
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import numpy as np
import pandas as pd 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# constants
file_ekg = "/home/javaprog/Data/Personal/TensorFlow/EKG/ecg.csv"
num_epochs = 25
LEARNING_RATE = 0.1

# model class
class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        self.encoder = nn.Sequential (
            nn.Linear(140, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential (
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 140),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        # return
        return x

def get_model():
    model = AnomalyDetectionModel()
    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.L1Loss()

    # return
    return model, loss, optimizer

# load the ekg data
df_ekg = pd.read_csv(file_ekg, header=None)
raw_data = df_ekg.values
print(df_ekg.head())

# get the labels
labels = raw_data[:,-1]
features = raw_data[:,0:-1]
print("the features have size {} and labels have shape {}".format(features.shape, labels.shape))

# normalize the feature data
# scaler = preprocessing.StandardScaler()
scaler = preprocessing.MinMaxScaler()
print(features[0,])
features = scaler.fit_transform(features)
print(features[0,])

# divide data into regular and irregular EKGs
regular_data = features[labels == 1]
regular_labels = labels[labels == 1]
irregular_data = features[labels == 0]
print("the regular features have shape {} and irregular EKG features have shape {}".format(regular_data.shape, irregular_data.shape))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(regular_data, regular_labels, test_size=.15, random_state=42)
print("the regular train have shape {} and regular test have shape {}".format(X_train.shape, X_test.shape))

# get the model and the loss
model, loss, optimizer = get_model()
X_train = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
X_irregular = Variable(torch.from_numpy(irregular_data).float(), requires_grad=False)

# train the model
for epoch in range(num_epochs):
    # x = torch.unsqueeze(x, dim=1)
    # print("the stept {} x is of type {} and shape {} and requires grad {}".format(step, type(x), x.shape, x.requires_grad))
    prediction = model(X_train)
    step_loss = loss(prediction, X_train) 
    optimizer.zero_grad()
    step_loss.backward()
    optimizer.step()

    # print progress
    # test_prediction = model(X_test)
    # label_prediction = torch.max(test_prediction, 1)[1].data.squeeze
    accuracy = 0
    # accuracy = (label_prediction == test_y).sum().item() / float(test_y.size(0))
    print("epoch: {} with train loss {:.4f} and accuracy {:.4f}".format(epoch, step_loss.item(), accuracy))     


# get the loss
predictions = model(X_train)
train_loss = loss(predictions, X_train)

# get the threshold, one deviation from the mean
good_threshold = np.mean(train_loss.detach().numpy()) + np.std(train_loss.detach().numpy())
print("got loss threshold of {}".format(good_threshold))

# test on irregular data
print("\nirregular has shape {}".format(X_irregular.shape))
X_irregular_subset = X_irregular[1:7, :]
print("irregular subset has shape {}".format(X_irregular_subset.shape))
predictions_irregular = model(X_irregular_subset)
print("irregular predictions has shape {}".format(predictions_irregular.shape))
irregular_loss = loss(predictions_irregular, X_irregular_subset)
print("irregular loss has shape {}".format(irregular_loss.shape))
print("got irregular ECG losses: {}".format(irregular_loss))
print("got irregular ECG results: {}".format(irregular_loss > good_threshold))


# test on regular data
X_regular_subset = X_train[1:7, :]
predictions_regular = model(X_regular_subset)
regular_loss = loss(predictions_regular, X_regular_subset)
print("got regular ECG losses: {}".format(regular_loss.detach().numpy()))
print("got regular ECG results: {}".format(regular_loss.detach().numpy() > good_threshold))

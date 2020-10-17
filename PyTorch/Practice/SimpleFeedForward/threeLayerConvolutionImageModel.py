# imports
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

print("the torch version is {}".format(torch.__version__))

# import the image data
scratch_dir = "/home/javaprog/Data/Personal/Scratch/"
cifar10 = datasets.CIFAR10(scratch_dir, train=True, download=True)
cifar10_validation = datasets.CIFAR10(scratch_dir, train=False, download=True)

# print the method resolution order of the dataset
type(cifar10).__mro__

# get the length of the dataset
len(cifar10)

# get the class names
class_names = cifar10.classes

# print an image
index = 77
image, label = cifar10[index]
image, label, class_names[label]

# print the images
# plt.imshow(image)
# plt.show()

# reload cifar10 into tensor format
cifar10_tensor = datasets.CIFAR10(scratch_dir, train=True, download=False, transform=transforms.ToTensor())
print("the dataset length is {}".format(len(cifar10_tensor)))

# load all the images into a batch tensor
images_stacked = torch.stack([image_temp for image_temp, _ in cifar10_tensor], dim=3)
print("the stacked images are of type {} and shape {}".format(type(images_stacked), images_stacked.shape))

# build a tensor to compute the mean and std
images_calc = images_stacked.view(3, -1)
print("the stacked mean/std images are of type {} and shape {}".format(type(images_calc), images_calc.shape))

# get the mean and std tensors
images_mean = images_calc.mean(dim=1)
images_std = images_calc.std(dim=1)
print("the mean tensor is {}".format(images_mean))
print("the std tensor is {}".format(images_std))
print("the mean numpy is {}".format(images_mean.numpy()))
print("the std numpy is {}".format(images_std.numpy()))

# reload the tensor images normalized
cifar10_normalized = datasets.CIFAR10(scratch_dir, train=True, download=False, 
                        transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(images_mean, images_std)]))
print("the normalized images are of length {}".format(len(cifar10_normalized)))

# reload the tensor images validation normalized
cifar10_normalized_val = datasets.CIFAR10(scratch_dir, train=False, download=False, 
                        transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(images_mean, images_std)]))
print("the normalized validation images are of length {}".format(len(cifar10_normalized_val)))

# print the transformad sample image
image_normalized, label = cifar10_normalized[index]
# plt.imshow(image_normalized.permute(1, 2, 0))
# plt.show()

# print the classes
print("the class names are {}".format(class_names))

# filter down to deer and ship
bi_label_map = {4:0, 9:1}
bi_class_names = [class_names[4], class_names[9]]
print("the binary classes are {}".format(bi_class_names))

# create binary dataset
cifar10_binary = [(temp_image, bi_label_map[label]) for temp_image, label in cifar10_normalized if label in [4, 9]]
cifar10_binary_val = [(temp_image, bi_label_map[label]) for temp_image, label in cifar10_normalized_val if label in [4, 9]]
print("got binary dataset for training of size {} and validation dataset of size {}".format(len(cifar10_binary), len(cifar10_binary_val)))

# build an NN class
class ThreeHiddenLayerConvolutional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_01_input = nn.Sequential (
            nn.Conv2d(
                in_channels=3,   # the one grayscale channel
                out_channels=16,  # will output 16 channels
                kernel_size=3,   # the size of the moving filter 3x3
                stride=1,        # how much the filter moves by step
                padding=1
            ),                   # output will be (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),     # height and width reduced by factor of 2, output will be (16, 16, 16)
        )
        # self.conv_01_input = nn.Conv2d(
        #         in_channels=3,   # the one grayscale channel
        #         out_channels=16,  # will output 16 channels
        #         kernel_size=3,   # the size of the moving filter 3x3
        #         stride=1,        # how much the filter moves by step
        #         padding=1
        #     ),                   # output will be (16, 32, 32)
        # self.conv_01_activation = nn.Tanh()
        # self.conv_01_maxpool = nn.MaxPool2d(kernel_size=2),     # height and width reduced by factor of 2, output will be (16, 16, 16)

        self.flatten_01 = nn.Flatten(start_dim=1)

        self.linear_02 = nn.Linear(4096, 1024)
        self.linear_02_activation = nn.Tanh()
        self.linear_03 = nn.Linear(1024, 128)
        self.linear_03_activation = nn.Tanh()
        self.linear_04_output = nn.Linear(128, 2)
        self.linear_04_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # input
        x = self.conv_01_input(x)
        # x = self.conv_01_activation(x)
        # x = self.conv_01_maxpool(x)

        # flatten
        x = self.flatten_01(x)

        # hidden layers
        x = self.linear_02(x)
        x = self.linear_02_activation(x)
        x = self.linear_03(x)
        x = self.linear_03_activation(x)

        # output
        x = self.linear_04_output(x)
        x = self.linear_04_activation(x)

        # return
        return x

three_hidden_model = ThreeHiddenLayerConvolutional()
print(three_hidden_model)

# first set the trining parameters
batch_size = 32
learning_rate = 1e-2
number_epochs = 10

# create the data loader
train_loader = data.DataLoader(cifar10_binary, batch_size=batch_size, shuffle=True)

# create the optimizer
optimizer = optim.SGD(three_hidden_model.parameters(), lr=learning_rate)
loss_function = nn.NLLLoss()

# train the network
for epoch in range(number_epochs):
    for temp_images, labels in train_loader:
        temp_size = temp_images.shape[0]
        # print("got images of size {}".format(temp_images.shape))
        outputs = three_hidden_model(temp_images)
        loss = loss_function(outputs, labels)

        # backward gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # print data for each epoch
    print("Epoch: {} with loss {:f}".format(epoch, loss))


# validate the model
validation_loader = data.DataLoader(cifar10_binary_val, batch_size=batch_size, shuffle=False)
number_correct = 0
number_total = 0

with torch.no_grad():
    for temp_images, labels in validation_loader:
        temp_size = temp_images.shape[0]
        outputs = three_hidden_model(temp_images)
        # print("outputs is of type {} and shape {}".format(type(outputs), outputs.shape))
        _, prediction = torch.max(outputs, dim=1)
        number_total += labels.shape[0]
        number_correct += int((prediction == labels).sum())

print("the model accuracy is {:f}".format(number_correct/number_total))





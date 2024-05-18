import random
import numpy as np
# from keras.datasets import mnist
from matplotlib import pyplot as plt
import scipy


def preprocess_matrix(matrix):
    angle = (random.random() - 0.5) * 40
    vector = (np.random.rand(2) - 0.5) * 7
    # scale = random.normalvariate(1, 0.1)
    rotated_matrix = scipy.ndimage.rotate(matrix, angle, reshape=False)
    rotated_shifted_matrix = scipy.ndimage.shift(rotated_matrix, vector)
    # rotated_shifted_zoomed_matrix =  scipy.ndimage.zoom(matrix, scale)
    # print(rotated_shifted_matrix.shape)
    return rotated_shifted_matrix

# (train_X, train_y), (test_X, test_y) = mnist.load_data()
# np.save("Training Images",train_X)
# np.save("Training Labels",train_y)
# np.save("Test Images",test_X)
# np.save("Test Labels",test_y)
train_X = np.load("Data Set/Digits/Training Images.npy")
train_y = np.load("Data Set/Digits/Training Labels.npy")
test_X = np.load("Data Set/Digits/Test Images.npy")
test_y = np.load("Data Set/Digits/Test Labels.npy")
# print('X_train: ' + str(train_X.shape))

for i in range(len(train_X)):
# for i in range(9):
    train_X[i] = preprocess_matrix(train_X[i])
    gaussian_noise = np.random.normal(10, 25, train_X[i].shape)
    noisy_img = train_X[i] + gaussian_noise
    train_X[i] = np.clip(noisy_img, 0, 255).astype(np.uint8)
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    
# print(train_y[:9])
# plt.show()
for i in range(len(test_X)):
    test_X[i] = preprocess_matrix(test_X[i])
    gaussian_noise = np.random.normal(10, 25, test_X[i].shape)
    noisy_img = test_X[i] + gaussian_noise
    test_X[i] = np.clip(noisy_img, 0, 255).astype(np.uint8)

np.save("Processed Training Images",train_X)
np.save("Processed Training Labels",train_y)
np.save("Processed Test Images",test_X)
np.save("Processed Test Labels",test_y)

# mnist_data = scipy.io.loadmat('matlab/emnist-letters.mat')['dataset']

# print(len(mnist_data))
# X_train = mnist_data['train'][0,0]['images'][0,0]
# y_train = mnist_data['train'][0,0]['labels'][0,0]
# X_test = mnist_data['test'][0,0]['images'][0,0]
# y_test = mnist_data['test'][0,0]['labels'][0,0]

# X_train = X_train.reshape( (X_train.shape[0], 28, 28), order='F')
# y_train = y_train.reshape( (y_train.shape[0], 28, 28), order='F')
# X_test = X_test.reshape( (X_test.shape[0], 28, 28), order='F')
# y_test = y_test.reshape( (y_test.shape[0], 28, 28), order='F')

# np.save("Training Images",X_train)
# np.save("Training Labels",y_train)
# np.save("Test Images",X_test)
# np.save("Test Labels",y_test)
# train_X = np.load("Data Set/letters/Training Images.npy")
# train_y = np.load("Data Set/letters/Training Labels.npy")
# test_X = np.load("Data Set/letters/Test Images.npy")
# test_y = np.load("Data Set/letters/Test Labels.npy")

# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))

# print(train_X[0])

# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i].reshape( (28, 28), order='F'), cmap=plt.get_cmap('gray'))
    
# print(train_y[:9])
# plt.show()

# data_arr = np.random.rand(2,500)
# print(data_arr[:,:5])
# label_arr = np.zeros(500)
# for i in range(len(label_arr)):
#     data = data_arr[:,i]
#     check = (data[0] * data[0] + data[1] * data[1]) > 0.4
#     label_arr[i] = check

# print(label_arr[:5])

# fig, ax = plt.subplots()
# # plt.plot(data_arr[0],data_arr[1], 'ro')
# col = np.where(label_arr==0, "b", "r")
# ax.scatter(data_arr[0],data_arr[1], c=col, s=5)
# plt.show()
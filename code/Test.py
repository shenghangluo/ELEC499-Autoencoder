
from __future__ import print_function
import tensorflow as tf
import scipy.io
import numpy as np
import random
import commpy as cp
from scipy.io import loadmat

a = np.ones(1000) # create an array of 1000 1's for the example
np.save('outfile_name', a) # save the file as "outfile_name.npy"

"""
# Create Upsampling matrix
a = np.zeros((104, 416))
row_index = np.arange(104)
colum_index = np.arange(0, 416, 4)
print("column is: ", colum_index)
a[row_index, colum_index] = 1
print("a is: ", a)
"""

"""
# Pytorch
import torch.nn as nn
import torch

inputs = torch.tensor([1, 0, 2, 3, 0, 1, 1], dtype=torch.float32)
filters = torch.tensor([2, 1, 3], dtype=torch.float32)

inputs = inputs.unsqueeze(0).unsqueeze(0)                   # torch.Size([1, 1, 7])
filters = filters.unsqueeze(0).unsqueeze(0)                 # torch.Size([1, 1, 3])
conv_res = F.conv1d(inputs, filters, padding=0, groups=1)   # torch.Size([1, 1, 5])
pad_res = F.pad(conv_res, (1, 1), mode='constant', value=0) # torch.Size([1, 1, 7])
"""

"""
# load matfile
filt = loadmat('rrc.mat')['rrt']
filt = np.array(filt)
filt = filt.flatten()[62-15:62+16]
filt = tf.convert_to_tensor(filt, dtype=tf.float32)
#filt = tf.reshape(filt, [-1])

with tf.Session() as sess:
    print(sess.run(filt))
"""



"""
#Convlution Test
i = tf.constant([1, 2, 3, 4, 5, 6, 7], dtype=tf.float32, name='i')
k = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32, name='k')

#print(i.shape)
#print(k.shape)

data = tf.reshape(i, [1, 1, 7], name='data')
kernel = tf.reshape(k, [5, 1, 1], name='kernel')
#kernel = tf.zeros([1, 6, 1])

data = tf.transpose(data, [0, 2, 1])
#print("x shape is: ", x.shape)

print("data shape is: ", data.shape)
print("kernel shape is: ", kernel.shape)

out = tf.nn.conv1d(data, kernel, 1, padding='VALID')
out = tf.transpose(out, [0, 2, 1])
print("output shape is: ", out.shape)
res = tf.squeeze(out)
with tf.Session() as sess:
    print("data is: ", sess.run(data))
    print("kernel is: ", sess.run(kernel))
    print("output from convolution is: ", sess.run(out))
    print("res is: ", sess.run(res))
"""



"""
indices = tf.constant([[1], [2], [4], [7]])
updates = tf.constant([[5, 5, 5, 5]])
shape = tf.constant([4, 4])
scatter = tf.scatter_nd(indices, updates, shape)
print(scatter)
session = tf.Session()
print(session.run(scatter))
"""
#dim = x.get_shape()[0].value
#print(dim)
#n = 5
#y = np.empty(dim * n, dtype=complex)
#y = tf.convert_to_tensor(y)
#y[0::n] = x
#zero_array = np.zeros(dim, dtype=complex)
#for i in range(1, n):
#    y[i::n] = zero_array

#print(y)
#a = cp.utilities.upsample(a, 5)
#print(tf.shape(a))

#print("The .numpy() method explicitly converts a Tensor to a numpy array")
#print(tensor.numpy())


#real, imag = tf.split(a, 2, 0)

#a = tf.complex(real, imag)

#areal = tf.math.real(a)
#aimag = tf.math.imag(a)
#final = tf.concat([areal, aimag], 0)

#sess = tf.Session()
#print(sess.run(a))
#print(sess.run(imag))
#print(sess.run(a))
#print(sess.run(areal))
#print(sess.run(aimag))
#print(sess.run(final))

# self._mul = CToR.get_r(self._mul)
# inputs = CToR.get_r(inputs)
# inputs = tf.dtypes.cast(inputs, tf.float32)
# print("mul is: ", tf.shape(self._mul))


# self._concat = tf.concat([self._mul, self._fe], 0)
# self._concat = CToR.get_r(self._concat)
# self._fe = tf.reshape(self._fe, [-1])
# out = tf.reshape(self._pe, [-1])
# out = RToC.get_c(out)
# self._fe = RToC.get_c(self._fe)

# Cinputs = RToC.get_c(inputs)
# Cinputs = tf.reshape(Cinputs, [-1])
# print("Cinput is: ", tf.shape(Cinputs))
# print("out is: ", tf.shape(out))
# Multiply
# self._mul = tf.multiply(Cinputs, out)
# self._mul = tf.dtypes.cast(self._mul, tf.int32)
# self._fe = tf.dtypes.cast(self._fe, tf.int32)
# print("fe is: ", tf.shape(self._fe))
# print("mul is: ", tf.shape(self._mul))
# Concatenate
# print("mul shape: ", self._mul.get_shape())
# print("fe shape: ", self._fe.get_shape())
# print("pe is: ", tf.shape(self._pe))
# print("inputs is: ", tf.shape(inputs))

# self._pe = tf.reshape(self._pe, [-1])
# real = tf.slice(self._pe, [0, 0], [-1, 1])
# imag = tf.slice(self._pe, [0, 1], [-1, 1])
# self._pe = tf.complex(real, imag)
# print("pe become: ", self._pe.get_shape())
# self._fe = RToC.get_c(self._fe)
# self._fe = CToR.get_r(self._fe)
# print("fe become: ", self._fe.get_shape())
# print("fe: ", tf.shape(self._fe))
# print("fe become: ", self._fe.get_shape())
# print("pe shape: ", self._pe.get_shape())
# self._pe = RToC.get_c(self._pe)

# print("inputs become: ", tf.shape(inputs))
# print("pe become: ", tf.shape(self._pe))


# print("pe extrend to: ", self._pe)
# self._mul = tf.multiply(inputs, self._pe)
# self._mul = CToR.get_r(self._mul)
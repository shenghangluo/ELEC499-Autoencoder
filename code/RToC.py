import tensorflow as tf


def get_c(inputs):
    real = tf.slice(inputs, [0, 0], [-1, tf.dtypes.cast(inputs.get_shape().as_list()[1]/2, tf.int32)])
    imag = tf.slice(inputs, [0, tf.dtypes.cast(inputs.get_shape().as_list()[1]/2, tf.int32)], [-1, tf.dtypes.cast(inputs.get_shape().as_list()[1]/2, tf.int32)])
    output = tf.complex(real, imag)
    return output

    #real = tf.slice(inputs, [0, 0], [2, 2])
    #imag = tf.slice(inputs, [0, 2], [2, 2])


#
#a = tf.constant([[[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]],
#                 [[11., 12., 13., 14., 15., 16.], [17., 18., 19., 20., 21., 22.]]])
# b= tf.constant([[1., 2., 3., 4., 5., 6.],
#                 [7., 8., 9., 10., 11., 12.]])
# out = tf.complex(a, b)
#
#print("a shape is: ", a.shape)
#ca = tf.slice(a, [0, 1, 0], [-1, 1, -1])
#print("ca shape is: ", ca.shape)
# print("ca shape is: ", ca.shape)
# #d = tf.constant([[1.0, 2.0]])
# ca = get_c(a)
# #cb = get_c(b)
#
# #mul = tf.multiply(ca, cb)
#
#
# print("ca is: ", tf.shape(ca))
# #print("b is: ", tf.shape(b))
# #print("cb is: ", tf.shape(cb))
# #print("d is: ", d.get_shape())
# #
#sess = tf.Session()
# #print(sess.run(mul))
#print(sess.run(ca))
#print(a.get_shape().as_list()[1])

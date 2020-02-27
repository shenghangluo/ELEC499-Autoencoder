#import tensorflow as tf
# a = [[1, 10, 26.9, 2.8, 166.32, 62.3], [2, 3, 4, 5, 6, 7]]
# d = [[1, 2, 3, 4, 8, 6], [3, 4, 5, 6, 7, 8]]
# b = tf.math.argmax(a,1)
# c = tf.keras.backend.eval(b)
#
# w = tf.math.argmax(d,1)
# u = tf.keras.backend.eval(w)
#
#
# out = tf.equal(b, w)
#
#
# print("C outcome is: ", c)
# print("D outcome is:", u)
# print("is equal:", tf.keras.backend.eval(out))

import tensorflow as tf
# a = tf.get_variable("a", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(a))

#
# tensor = tf.constant([[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
#                       [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
# print("tensor shape is: ", tensor.shape)
# b = tf.math.l2_normalize(tensor, axis=2)
# c = tf.dtypes.cast(b.get_shape()[2], tf.float32)
# d = tf.math.sqrt(c)
# e = tf.multiply(d, b)

tensor = tf.constant([[4.0, 5.0, 6.0],
                       [10.0, 11.0, 12.0]])
print("tensor shape is: ", tensor.shape)

a = tf.reshape(tensor, [-1,1,3])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
#     print(sess.run(e))

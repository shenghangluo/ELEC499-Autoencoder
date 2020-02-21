import tensorflow as tf
import RToC

def get_r(inputs):
    real = tf.math.real(inputs)
    imag = tf.math.imag(inputs)
    output = tf.concat([real, imag], 1)
    return output


#a = tf.constant([[1., 2., 3., 4., 5., 6.],
#                 [7., 8., 9., 10., 11., 12.]])
#ca = RToC.get_c(a)
#ra = get_r(ca)
#a = tf.Variable([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
#                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

#b = tf.Variable([1.0, 2.0])

#real, imag = tf.split(a, 2, 0)
#ca = tf.complex(real, imag)

#real, imag = tf.split(b, 2, 0)
#cb = tf.complex(real, imag)

#cc = tf.multiply(ca, cb)

#answer = get_r(cc)

#answer = tf.contrib.layers.fully_connected(f, num_outputs=2,
#                                           activation_fn=None)
#print("a is: ", tf.shape(a))
#print("ca is: ", tf.shape(ca))
#print("answer is: ", tf.shape(answer))

#sess = tf.Session()
#print(sess.run(ca))
#print(sess.run(ra))

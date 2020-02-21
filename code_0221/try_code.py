import tensorflow as tf
a = [[1, 10, 26.9, 2.8, 166.32, 62.3], [2, 3, 4, 5, 6, 7]]
d = [[1, 2, 3, 4, 8, 6], [3, 4, 5, 6, 7, 8]]
b = tf.math.argmax(a,1)
c = tf.keras.backend.eval(b)

w = tf.math.argmax(d,1)
u = tf.keras.backend.eval(w)


out = tf.equal(b, w)


print("C outcome is: ", c)
print("D outcome is:", u)
print("is equal:", tf.keras.backend.eval(out))

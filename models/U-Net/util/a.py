import tensorflow as tf

x1 = tf.constant(1.0, shape=[1,3,3,1])
kernel = tf.constant(1.0, shape=[3,3,3,1])
x2 = tf.constant(1.0, shape=[1,6,6,3])  
x3 = tf.constant(1.0, shape=[1,5,5,3])
y2 = tf.nn.conv2d(x3, kernel, strides=[1,2,2,1], padding="SAME")
sess = tf.Session()

print(sess.run(y2))


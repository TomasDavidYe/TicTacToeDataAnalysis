import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


features = pd.read_csv("./resources/features.txt")
labels = pd.read_csv("./resources/labels.txt")
learning_rate = 3
input_layer_dim = features.shape[1]
hidden_layer_1_dim = 50
hidden_layer_2_dim = 50
hidden_layer_3_dim = 50
output_layer_dim = labels.shape[1]
print("Feature shape = ", features.shape)
print("Labels shape = ", labels.shape)

given_weights = {
    'h1': tf.Variable(tf.truncated_normal([input_layer_dim, hidden_layer_1_dim])),
    'h2': tf.Variable(tf.truncated_normal([hidden_layer_1_dim,hidden_layer_2_dim])),
    'h3': tf.Variable(tf.truncated_normal([hidden_layer_2_dim, hidden_layer_3_dim])),
    'out': tf.Variable(tf.truncated_normal([hidden_layer_3_dim, output_layer_dim]))
}

# given_biases = {
#     'b1': tf.Variable(tf.truncated_normal([hidden_layer_1_dim])),
#     'b2': tf.Variable(tf.truncated_normal([hidden_layer_2_dim])),
#     'b3': tf.Variable(tf.truncated_normal([hidden_layer_3_dim])),
#     'out': tf.Variable(tf.truncated_normal([output_layer_dim]))
# }


session = tf.Session()
X = tf.placeholder(tf.float32, [None, input_layer_dim])
y = tf.placeholder(tf.float32, [None, output_layer_dim])


# def apply_net(x, weights, biases):
#     layer_1_matrix = x
#
#     layer_2_matrix = tf.add(tf.matmul(layer_1_matrix, weights['h1']), biases['b1'])
#     layer_2_matrix = tf.nn.sigmoid(layer_2_matrix)
#
#     layer_3_matrix = tf.add(tf.matmul(layer_2_matrix, weights['h2']), biases['b2'])
#     layer_3_matrix = tf.nn.sigmoid(layer_3_matrix)
#
#     layer_4_matrix = tf.add(tf.matmul(layer_3_matrix, weights['h3']), biases['b3'])
#     layer_4_matrix = tf.nn.sigmoid(layer_4_matrix)
#
#     output_layer_matrix = tf.add(tf.matmul(layer_3_matrix, weights['out']), biases['out'])
#     return tf.nn.sigmoid(output_layer_matrix)



def apply_net_only_weights(x, weights):
    layer_1_matrix = x

    layer_2_matrix = tf.matmul(layer_1_matrix, weights['h1'])
    layer_2_matrix = tf.nn.sigmoid(layer_2_matrix)

    layer_3_matrix = tf.matmul(layer_2_matrix, weights['h2'])
    layer_3_matrix = tf.nn.sigmoid(layer_3_matrix)

    layer_4_matrix = tf.matmul(layer_3_matrix, weights['h3'])
    layer_4_matrix = tf.nn.sigmoid(layer_4_matrix)

    output_layer_matrix = tf.matmul(layer_4_matrix, weights['out'])
    return tf.nn.sigmoid(output_layer_matrix)


init = tf.global_variables_initializer()
session.run(init)

computedY = apply_net_only_weights(X, given_weights)
cost = tf.reduce_mean(tf.square(y - computedY))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

temp = tf.constant(0.5, shape=labels.shape, dtype=tf.float32)
prediction = tf.cast(tf.greater_equal(computedY, temp), tf.float32)
accuracy_vector = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(accuracy_vector)


#File_Writer = tf.summary.FileWriter('/Users/i354640/Desktop/Home_Projects/PythonProjects/TensorFlowPlaying/graph', session.graph)
#

for i in range(5000):
    print("Iteration: ", i, "Cost: ", session.run(cost, {X: features, y: labels}), "Accuracy: ", session.run(accuracy, {X: features, y: labels}))
    session.run(train, {X: features, y: labels})

[w1, w2, w3, w4, y] = session.run([
    given_weights['h1'],
    given_weights['h2'],
    given_weights['h3'],
    given_weights['out'],
    computedY
], {X: features, y: labels})
# print(y)
np.savetxt("./resources/weights1.txt", w1, delimiter=",")
np.savetxt("./resources/weights2.txt", w2, delimiter=",")
np.savetxt("./resources/weights3.txt", w3, delimiter=",")
np.savetxt("./resources/weights4.txt", w4, delimiter=",")
# np.savetxt("./resources/bias1.txt", b1, delimiter=",")
# np.savetxt("./resources/bias2.txt", b2, delimiter=",")
# np.savetxt("./resources/bias3.txt", b3, delimiter=",")
# np.savetxt("./resources/bias4.txt", b4, delimiter=",")
np.savetxt("./resources/computedY.txt", y, delimiter=",")
#
#



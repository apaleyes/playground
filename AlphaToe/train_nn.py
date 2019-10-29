import tensorflow as tf
import numpy as np


board_size = 3

dropout_rate = tf.placeholder(tf.float32)

def build_model(x, dropout_rate, board_size):
    n_input = board_size * board_size + 2
    n_hidden1 = 50
    n_out = 2

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden1]), name='w_h_1'),
        'out1': tf.Variable(tf.random_normal([n_hidden1, 1]), name='w_out_1'),
        'out2': tf.Variable(tf.random_normal([n_hidden1, 1]), name='w_out_2')
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden1]), name='b_h_1'),
        'out1': tf.Variable(tf.random_normal([1]), name='b_out_1'),
        'out2': tf.Variable(tf.random_normal([1]), name='b_out_2')
    }

    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='hidden_layer_1')
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, rate=dropout_rate)

    output_1 = tf.add(tf.matmul(layer_1[0, None, :], weights['out1']), biases['out1'], name='output_layer_1')
    output_1 = tf.sigmoid(output_1)
    output_2 = tf.add(tf.matmul(layer_1[1, None, :], weights['out2']), biases['out2'], name='output_layer_2')
    output_2 = tf.math.tanh(output_2)
    output_layer = tf.concat(1, [output_1, output_2])

    return output_layer


def position_to_input(position):
    marker_to_value = {
        'x': 1,
        '.': 0,
        'o': -1
    }

    input_vector = [marker_to_value[marker] for marker in position]

    return input_vector

def move_to_input(marker, next_move_index):
    move_input = []
    if marker == 'x':
        move_input.append(1)
    else:
        move_input.append(-1)

    move_input.append(next_move_index)

    return move_input

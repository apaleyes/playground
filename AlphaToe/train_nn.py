import tensorflow as tf
import numpy as np

import math


from common import FIRST_PLAYER_WIN, DRAW, SECOND_PLAYER_WIN, TreeNode, TreeSearch

UCB_C = 1.0

BOARD_SIZE = 3

def build_model(x, board_size):
    n_input = 2 * board_size * board_size
    n_hidden1 = 50

    weights = {
        'h1': tf.Variable(tf.random.normal([n_input, n_hidden1]), name='w_h_1'),
        'out1': tf.Variable(tf.random.normal([n_hidden1, 1]), name='w_out_1'),
        'out2': tf.Variable(tf.random.normal([n_hidden1, 1]), name='w_out_2')
    }

    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden1]), name='b_h_1'),
        'out1': tf.Variable(tf.random.normal([1]), name='b_out_1'),
        'out2': tf.Variable(tf.random.normal([1]), name='b_out_2')
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='hidden_layer_1')
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, rate=0.1)

    output_1 = tf.add(tf.matmul(layer_1, weights['out1']), biases['out1'], name='output_layer_1')
    output_1 = tf.sigmoid(output_1)
    output_2 = tf.add(tf.matmul(layer_1, weights['out2']), biases['out2'], name='output_layer_2')
    output_2 = tf.math.tanh(output_2)
    output_layer = tf.concat([output_1, output_2], 1)

    return output_layer


def position_to_input(position):
    marker_to_value = {
        'x': 1,
        '.': 0,
        'o': -1
    }

    input_vector = [marker_to_value[marker] for marker in position]

    return input_vector

def move_to_input(marker, next_move_index, board_size):
    move_input = [0] * (board_size ** 2)
    if marker == 'x':
        move_input[next_move_index] = 1
    else:
        move_input[next_move_index] = -1

    return move_input


def test_nn():
    test_board_size = 3
    test_position = ['.'] * (test_board_size ** 2)
    test_input = position_to_input(test_position) + move_to_input('x', 0, test_board_size)

    x = tf.placeholder("float", [None, 2 * test_board_size * test_board_size])
    model = build_model(x, test_board_size)

    with tf.Session() as tf_session:
        tf_session.run(tf.global_variables_initializer())
        test_output = tf_session.run(model, feed_dict={x: [test_input]})
    
    print(test_output)

if __name__ == '__main__':
    test_nn()

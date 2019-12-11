import tensorflow as tf
import numpy as np

import math

from mcts import MonteCarloTreeSearch
from game import FIRST_PLAYER_WIN, DRAW, SECOND_PLAYER_WIN, TicTacToeGame


UCB_C = 1.0

BOARD_SIZE = 3

def build_model_position_and_move(x, n_input):
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

def build_model_position(x, n_input):
    n_hidden1 = 50

    weights = {
        'h1': tf.Variable(tf.random.normal([n_input, n_hidden1]), name='w_h_1'),
        'out': tf.Variable(tf.random.normal([n_hidden1, 1]), name='w_out')
    }

    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden1]), name='b_h_1'),
        'out': tf.Variable(tf.random.normal([1]), name='b_out')
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='hidden_layer_1')
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, rate=0.1)

    output_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'], name='output_layer')
    output_layer = tf.math.tanh(output_layer)

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
    test_n_input = 2 * test_board_size * test_board_size

    x = tf.placeholder("float", [None, test_n_input])
    model = build_model(x, test_n_input)

    with tf.Session() as tf_session:
        tf_session.run(tf.global_variables_initializer())
        test_output = tf_session.run(model, feed_dict={x: [test_input]})
    
    print(test_output)


class Model():
    def __init__(self, input_size, tf_session):
        self.n_input = input_size
        self.x = tf.placeholder("float", [None, self.n_input])
        self.tf_session = tf_session
        self.model = build_model_position(self.x, self.n_input)

    def evaluate(self, model_input):
        output = self.tf_session.run(self.model, feed_dict={self.x: [model_input]})
        position_value = output[0][0]

        return position_value


class ModelBaseMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, game, model):
        super().__init__(game)
        self.model = model

    def simulate(self, node):
        outcome = self.game.find_outcome(node.position)
        if outcome is not None:
            return outcome

        position_model_input = self.game.position_to_model_input(node.position)
        possible_moves = self.game.get_possible_moves(node.position, node.is_first_player_move)
        position_values = []
        for move in possible_moves:
            model_input = position_model_input + self.game.move_to_model_input(move)
            position_value = self.model.evaluate(model_input)
            position_values.append(position_value)
        return max(position_values)

    def backpropagate(self, node, value):
        node.n_visits += 1
        node.value += value

        if node.parent is not None:
            self.backpropagate(node.parent, value)

if __name__ == '__main__':
    with tf.Session() as tf_session:
        board_size = 3
        game = TicTacToeGame(board_size)
        model = Model(2 * board_size * board_size, tf_session)
        mcts = ModelBaseMonteCarloTreeSearch(game, model)


        y = tf.placeholder('float')
        loss = tf.reduce_mean(tf.square(model.model - y))
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        tf_session.run(tf.global_variables_initializer())

        n_training_games = 1000
        for game_i in range(n_training_games):
            training_data = []
            outcome = None
            position = game.get_initial_position()
            while outcome is None:
                new_position = mcts.loop(3, initial_position=position)
                move = game.get_move(position, new_position)

                model_input = game.position_to_model_input(position) + game.move_to_model_input(move)
                training_data.append(model_input)

                position = new_position
                outcome = game.find_outcome(position)

            _, current_loss = tf_session.run([optimizer, loss], feed_dict={model.x: training_data, y: [outcome]*len(training_data)})

            if game_i % 200 == 0:
                print("Played {} games, current loss is {}".format(game_i, current_loss))

        print()
        print("Now let's play a test game")
        outcome = None
        position = game.get_initial_position()
        while outcome is None:
            position = mcts.loop(3, initial_position=position)
            print()
            print(position[0:3], '\n', position[3:6], '\n', position[6:9])
            outcome = game.find_outcome(position)


import tensorflow as tf
import numpy as np


from common import FIRST_PLAYER_WIN, DRAW, SECOND_PLAYER_WIN, TreeNode, TreeSearch

UCB_C = 1.0

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


class ModelTreeNode(TreeNode):
    def __init__(self, position):
        super().__init__(position)
        self.move = None

    def ucb(self):
        v = self.value
        n = self.n_visits
        N = self.parent.n_visits

        if n == 0:
            n = 1

        return v / n + UCB_C * math.sqrt(math.log(N) / n)


class ModelTreeSearch():
    def __init__(self, game, model):
        super().__init__(game)
        self.model = model

    def loop(self, n_iterations, initial_position=None, debug=False):
        if initial_position is None:
            initial_position = self.game.get_initial_position()
        root = self.create_node(initial_position)

        for i in range(n_iterations):
            print("Starting iteration " + str(i))

            leaf_node = self.select(root)
            if leaf_node is None:
                print("All possible positions were evaluated")
                break

            new_child = self.expand(leaf_node)
            value = self.simulate(new_child)
            self.backpropagate(new_child, value)

        if debug:
            self.print_tree(root)

        next_move_node = max(self.root.children, lambda node: node.n_visits)
        return next_move_node.move

    def create_node(self, position):
        return ModelTreeNode(position)

    def select(self, node):
        if len(node.children) == 0:
            # found a leaf, return it
            return node

        non_terminal_children = [child for child in node.children if not self.game.is_terminal(child.position)]
        # sort in decsending order
        non_terminal_children.sort(key=lambda c: c.ucb(), reverse=True)

        # iterate over sorted children, finish when
        # we found a non-terminal leaf
        for child in non_terminal_children:
            leaf = self.select(child)
            if leaf is not None:
                return leaf

        # we haven't found a non-terminal leaf in this subtree
        return None


    def expand(self, leaf_node):
        next_positions = self.game.create_children(leaf_node.position, leaf_node.is_first_player_move)

        for next_position in next_positions:
            new_node = self.create_node(next_position)
            new_node.parent = leaf_node
            new_node.is_first_player_move = not leaf_node.is_first_player_move

            move = next_position - leaf_node.position
            new_node.move = move
            move_value, _ = self.model.evaluate(leaf_node.position, move)
            new_node.value = move_value

            leaf_node.children.append(new_node)

        next_child = random.choice(leaf_node.children)
        return next_child


    def simulate(self, node):
        if node.parent is None:
            # we are at the root
            return 0

        _, position_value = self.model.evaluate(node.parent.position, node.move)
        return position_value


    def backpropagate(self, node, value):
        node.n_visits += 1
        node.value += value

        if node.parent is not None:
            self.backpropagate(node.parent, value)

    def select_move(self):
        selected_node = self.root.children[0]
        for node in self.root.children:
            if node.n_visits > selected_node.n_visits:
                selected_node = node

        return selected_node

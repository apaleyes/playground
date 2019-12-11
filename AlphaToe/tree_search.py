import math


class TreeNode:
    def __init__(self, position):
        self.position = position
        self.value = 0
        self.n_visits = 0
        self.children = []
        self.parent = None
        self.is_first_player_move = True


class TreeSearch:
    def __init__(self, game):
        self.game = game


    def loop(self, n_iterations, initial_position=None, debug=False):
        if initial_position is None:
            initial_position = self.game.get_initial_position()
        root = self.create_node(initial_position)

        for i in range(n_iterations):
            if debug:
                print("Starting iteration " + str(i))

            leaf_node = self.select(root)
            if leaf_node is None:
                if debug:
                    print("All possible positions were evaluated")
                break

            new_child = self.expand(leaf_node)
            outcome = self.simulate(new_child)
            self.backpropagate(new_child, outcome)

        if debug:
            self.print_tree(root)

        next_position = self.select_next_position(root)
        return next_position


    def select_next_position(self, node):
        raise NotImplementedError


    def create_node(self, position):
        raise NotImplementedError


    def expand(self, leaf_node):
        raise NotImplementedError


    def select(self, node):
        raise NotImplementedError


    def simulate(self, node):
        raise NotImplementedError


    def backpropagate(self, node, outcome):
        raise NotImplementedError


    def print_tree(self, node, level=0):
        print('{}{}'.format('\t' * level, node.position))
        for child in node.children:
            self.print_tree(child, level + 1)

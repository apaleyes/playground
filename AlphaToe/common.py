import math

FIRST_PLAYER_WIN = 1
SECOND_PLAYER_WIN = -1
DRAW = 0

UCB_C = 1.0


class Node:
    def __init__(self, position):
        self.position = position
        self.value = 0
        self.n_visits = 0
        self.children = []
        self.parent = None
        self.is_first_player_move = True

    def ucb(self):
        v = self.value
        n = self.n_visits
        N = self.parent.n_visits

        if n == 0:
            n = 1

        return v / n + UCB_C * math.sqrt(math.log(N) / n)


class TreeSearch:
    def __init__(self, game):
        self.game = game


    def loop(self, n_iterations, debug=False):
        root = Node(self.game.get_initial_position())

        for i in range(n_iterations):
            print("Starting iteration " + str(i))

            leaf_node = self.select(root)
            if leaf_node is None:
                print("All possible positions were evaluated")
                break

            new_child = self.expand(leaf_node)
            outcome = self.simulate(new_child)
            self.backpropagate(new_child, outcome)

        if debug:
            self.print_tree(root)


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

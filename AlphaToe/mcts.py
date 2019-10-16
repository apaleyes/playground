# implementation of Monte Carlo search tree accroding to
# http://mcts.ai/about/index.html

import math
import random

from common import FIRST_PLAYER_WIN, DRAW, SECOND_PLAYER_WIN, Node, TreeSearch
from game import TicTacToeGame


class MonteCarloTreeSearch(TreeSearch):
    def __init__(self, game):
        super().__init__(game)


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
            new_node = Node(next_position)
            new_node.parent = leaf_node
            new_node.is_first_player_move = not leaf_node.is_first_player_move
            leaf_node.children.append(new_node)

        next_child = random.choice(leaf_node.children)
        return next_child


    def simulate(self, node):
        outcome = self.game.simulate_to_the_end(node.position, node.is_first_player_move)
        return outcome


    def backpropagate(self, node, outcome):
        node.n_visits += 1
        if node.is_first_player_move and outcome == FIRST_PLAYER_WIN:
            node.value += 1
        if not node.is_first_player_move and outcome == SECOND_PLAYER_WIN:
            node.value += 1
        if outcome == DRAW:
            node.value += 0.5

        if node.parent is not None:
            self.backpropagate(node.parent, outcome)


if __name__ == "__main__":
    game = TicTacToeGame()
    mcts = MonteCarloTreeSearch(game)
    mcts.loop(5, True)

    # tttg = TicTacToeGame(board_size=4)
    # print(tttg.simulate_to_the_end(tttg.get_initial_position(), True))
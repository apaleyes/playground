# implementation of Monte Carlo search tree accroding to
# http://mcts.ai/about/index.html

import math
import random

from common import FIRST_PLAYER_WIN, DRAW, SECOND_PLAYER_WIN
from common import Game, Node, TreeSearch

class MockGame(Game):
    def __init__(self, n_children=3):
        self.n_children = n_children

    def get_initial_position(self):
        return "init"

    def create_children(self, position, is_first_player_move):
        return [position + " c" + str(i) for i in range(self.n_children)]

    def simulate_to_the_end(self, position, is_first_player_move):
        rnd = random.random()
        if rnd < 0.4:
            return FIRST_PLAYER_WIN
        elif rnd < 0.6:
            return DRAW
        else:
            return SECOND_PLAYER_WIN

    def is_terminal(self, position):
        n_moves = position.count('c')
        if n_moves < 3:
            return False

        return True


class TicTacToeGame(Game):
    def __init__(self, board_size=3, win_count=3):
        self.board_size = board_size
        self.win_count = win_count

        self.first_player_marker = 'x'
        self.second_player_marker = 'o'
        self.empty_marker = '.'

    def get_initial_position(self):
        position = [self.empty_marker] * (self.board_size * self.board_size)
        return position

    def create_children(self, position, is_first_player_move):
        children = []
        next_marker = self.first_player_marker if is_first_player_move else self.second_player_marker
        for i, marker in enumerate(position):
            if marker == self.empty_marker:
                child = position[:]
                child[i] = next_marker
                children.append(child)

        return children

    def simulate_to_the_end(self, position, is_first_player_move):
        sim_position = position[:]
        #self.print_board(sim_position)

        outcome = self.find_outcome(sim_position)
        while outcome is None:
            next_marker = self.first_player_marker if is_first_player_move else self.second_player_marker
            empty_spaces = [i for i, m in enumerate(sim_position) if m == self.empty_marker]

            # make tje move
            next_cell = random.choice(empty_spaces)
            sim_position[next_cell] = next_marker

            # prepare next move
            is_first_player_move = not is_first_player_move
            outcome = self.find_outcome(sim_position)
            # self.print_board(sim_position)

        return outcome

    def if_board_full(self, position):
        return (self.empty_marker not in position)

    def find_outcome(self, position):
        get_winner_by_marker = lambda marker: FIRST_PLAYER_WIN if marker == self.first_player_marker else SECOND_PLAYER_WIN

        board = [position[i:i+self.board_size] for i in range(0, len(position), self.board_size)]

        # we are looking at squares of self.win_count size
        # therefore a winning row, column or diagonal should contain only one symbol
        for top in range(self.board_size - self.win_count + 1):
            bottom = top + self.win_count - 1;

            for left in range(self.board_size - self.win_count + 1):
                right =  left + self.win_count - 1

                # Check each row
                for row in range(top, bottom + 1):
                    if board[row][left] == self.empty_marker:
                        # if row contains empty marker, it cannot be a winning row
                        continue

                    all_markers = set(board[row][left:right+1])
                    if len(all_markers) != 1:
                        # not all markers are the same, so cannot be a winning row
                        continue


                    return get_winner_by_marker(board[row][left])

                # Check each column.
                for col in range(left, right + 1):
                    if board[top][col] == self.empty_marker:
                        # if column contains empty marker, it cannot be a winning column
                        continue

                    all_markers = set([board[i][col] for i in range(top, bottom + 1)])
                    if len(all_markers) != 1:
                        # not all markers are the same, so cannot be a winning column
                        continue

                    return get_winner_by_marker(board[top][col])

                # Check top-left to bottom-right diagonal.
                if board[top][left] != self.empty_marker:
                    all_markers = set([board[top + i][left + i] for i in range(0, self.win_count)])
                    if len(all_markers) == 1:
                        return get_winner_by_marker(board[top][left])

                # Check top-right to bottom-left diagonal.
                if board[top][right] != self.empty_marker:
                    all_markers = set([board[top + i][right - i] for i in range(0, self.win_count)])
                    if len(all_markers) == 1:
                        return get_winner_by_marker(board[top][right])


        # Check for a completely full board.
        if self.if_board_full(position):
            return DRAW

        # there is no winner yet
        return None

    def is_terminal(self, position):
        outcome = self.find_outcome(position)
        return (outcome is not None)

    def print_board(self, position):
        board = [position[i:i+self.board_size] for i in range(0, len(position), self.board_size)]
        for line in board:
            print(''.join(line))
        print('\n')


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
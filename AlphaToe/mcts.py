# implementation of Monte Carlo search tree accroding to
# http://mcts.ai/about/index.html

import math
import random


FIRST_PLAYER_WIN = 1
SECOND_PLAYER_WIN = -1
DRAW = 0

UCB_C = 1.0


class MockGame:
    def __init__(self, n_children=3):
        self.n_children = n_children

    @property
    def initial_position(self):
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


class TicTacToeGame:
    def __init__(self, board_size=3, win_count=3):
        self.board_size = board_size
        self.win_size = win_size

        self.first_player_marker = 'x'
        self.second_player_marker = 'o'
        self.empty_marker = '.'

    @property
    def initial_position(self):
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

    def is_terminal(self, position):
        for top in range(self.board_size - self.win_count + 1)
            bottom = top + self.win_count - 1;

            for left in range(self.board_size - self.win_count + 1):
                rigth =  left + self.win_count - 1

                # Check each row.
                for row in range(top, bottom + 1)
                    if position[row][left] == self.empty_marker:
                        continue

                    for (int col = left; col <= right; ++col) {
                        if (board[row][col] != board[row][left]) {
                            continue nextRow;
                        }
                    }

                    return board[row][left];
                }

                // Check each column.
                nextCol: for (int col = left; col <= right; ++col) {
                    if (board[top][col] == Square.NONE) {
                        continue;
                    }

                    for (int row = top; row <= bottom; ++row) {
                        if (board[row][col] != board[top][col]) {
                            continue nextCol;
                        }
                    }

                    return board[top][col];
                }

                // Check top-left to bottom-right diagonal.
                diag1: if (board[top][left] != Square.NONE) {
                    for (int i = 1; i < lengthToWin; ++i) {
                        if (board[top+i][left+i] != board[top][left]) {
                            break diag1;
                        }
                    }

                    return board[top][left];
                }

                // Check top-right to bottom-left diagonal.
                diag2: if (board[top][right] != Square.NONE) {
                    for (int i = 1; i < lengthToWin; ++i) {
                        if (board[top+i][right-i] != board[top][right]) {
                            break diag2;
                        }
                    }

                    return board[top][right];
                }
            }
        }

        // Check for a completely full board.
        boolean isFull = true;

        full: for (int row = 0; row < board.length; ++row) {
            for (int col = 0; col < board.length; ++col) {
                if (board[row][col] == Square.NONE) {
                    isFull = false;
                    break full;
                }
            }
        }

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


def loop(game, n_iterations):
    root = Node(game.initial_position)

    for i in range(n_iterations):
        print("Starting iteration " + str(i))
        
        leaf_node = select(root, game)
        if leaf_node is None:
            print("All possible positions were evaluated")
            break

        new_child = expand(leaf_node, game)
        outcome = simulate(new_child, game)
        backpropagate(new_child, outcome)

    print_tree(root)


def select(node, game):
    if len(node.children) == 0:
        # found a leaf, return it
        return node

    non_terminal_children = [child for child in node.children if not game.is_terminal(child.position)]
    # sort in decsending order
    non_terminal_children.sort(key=lambda c: c.ucb(), reverse=True)

    # iterate over sorted children, finish when
    # we found a non-terminal leaf
    for child in non_terminal_children:
        leaf = select(child, game)
        if leaf is not None:
            return leaf

    # we haven't found a non-terminal leaf in this subtree
    return None


def expand(leaf_node, game):
    next_positions = game.create_children(leaf_node.position, leaf_node.is_first_player_move)

    for next_position in next_positions:
        new_node = Node(next_position)
        new_node.parent = leaf_node
        new_node.is_first_player_move = not leaf_node.is_first_player_move
        leaf_node.children.append(new_node)

    next_child = random.choice(leaf_node.children)
    return next_child


def simulate(node, game):
    outcome = game.simulate_to_the_end(node.position, node.is_first_player_move)
    return outcome


def backpropagate(node, outcome):
    node.n_visits += 1
    if node.is_first_player_move and outcome == FIRST_PLAYER_WIN:
        node.value += 1
    if not node.is_first_player_move and outcome == SECOND_PLAYER_WIN:
        node.value += 1
    if outcome == DRAW:
        node.value += 0.5

    if node.parent is not None:
        backpropagate(node.parent, outcome)


# def print_tree(node):
#     print(node.position)
#     for c in node.children:
#         print_tree(c)

def print_tree(node, level=0):
    print('\t' * level + node.position)
    for child in node.children:
        print_tree(child, level + 1)


if __name__ == "__main__":
    game = MockGame()
    loop(game, 5)

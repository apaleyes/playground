# board is a square of .
# first player is 'x'
# second player is 'o'

import math

class Game:
    def __init__(self, board_size=6, win_size=4):
        self.board_size = board_size
        self.win_size = win_size

        self.is_first_player_turn = True
        self.first_player_marker = 'x'
        self.second_player_marker = 'o'
        self.empty_marker = '.'

        self.board = [self.empty_marker] * (board_size * board_size)
        self.moves = []


    @property
    def current_marker(self):
        if self.is_first_player_turn:
            return self.first_player_marker
        else:
            return self.second_player_marker

    def get_possible_moves(self):
        moves = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                ind = i * self.board_size + j
                if self.board[ind] == '.':
                    moves.append((i, j))

        return moves

    def get_possible_positions(self):
        positions = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                ind = i * self.board_size + j
                if self.board[ind] == '.':
                    board_copy = self.board[:]
                    board_copy[ind] = self.current_marker
                    positions.append(board_copy)

        return positions

    def make_move(self, i, j):
        ind = i * self.board_size + j

        if self.board[ind] == '.':
            self.board[ind] = self.current_marker
            self.is_first_player_turn = not self.is_first_player_turn
            self.moves.append((i, j))
        else:
            raise IndexError('Position {},{} is already occupied by {}'.format(i, j, self.board[ind]))

    def check_game_over(self):
        

    @classmethod
    def print_board(cls, board):
        n = int(math.sqrt(len(board)))

        for i in range(n):
            for j in range(n):
                print(board[i * n + j], end = '')
            print("\n")

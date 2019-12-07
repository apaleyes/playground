import random


FIRST_PLAYER_WIN = 1
SECOND_PLAYER_WIN = -1
DRAW = 0


class Game:
    def get_initial_position(self):
        raise NotImplementedError

    def create_children(self, position, is_first_player_move):
        raise NotImplementedError

    def simulate_to_the_end(self, position, is_first_player_move):
        raise NotImplementedError

    def is_terminal(self, position):
        raise NotImplementedError

    def get_possible_moves(self, position, is_first_player_move):
        raise NotImplementedError

    def position_to_model_input(self, position):
        raise NotImplementedError

    def move_to_model_input(self, move):
        raise NotImplementedError

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

    def get_possible_moves(self, position, is_first_player_move):
        moves = []
        next_marker = self.first_player_marker if is_first_player_move else self.second_player_marker
        for i, marker in enumerate(position):
            if marker == self.empty_marker:
                moves.append({"marker": next_marker, "index": i})

        return moves

    def simulate_to_the_end(self, position, is_first_player_move):
        sim_position = position[:]
        #self.print_board(sim_position)

        outcome = self.find_outcome(sim_position)
        while outcome is None:
            next_marker = self.first_player_marker if is_first_player_move else self.second_player_marker
            empty_spaces = [i for i, m in enumerate(sim_position) if m == self.empty_marker]

            # make the move
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
            bottom = top + self.win_count - 1

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
        board = [position[i:(i + self.board_size)] for i in range(0, len(position), self.board_size)]
        for line in board:
            print(''.join(line))
        print('\n')

    def position_to_model_input(self, position):
        marker_to_value = {
            self.first_player_marker: 1,
            self.empty_marker: 0,
            self.second_player_marker: -1
        }

        input_vector = [marker_to_value[marker] for marker in position]

        return input_vector

    def move_to_model_input(self, move):
        move_input = [0] * (self.board_size ** 2)
        if move["marker"] == self.first_player_marker:
            move_input[move["index"]] = 1
        else:
            move_input[move["index"]] = -1

        return move_input

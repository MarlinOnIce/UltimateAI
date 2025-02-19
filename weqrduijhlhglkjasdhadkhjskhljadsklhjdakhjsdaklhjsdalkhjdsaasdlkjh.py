import random
import copy
import numpy
import time
from collections import defaultdict
from multiprocessing import Pool
import tkinter as tk

positions = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

positions_array = numpy.array(positions)

weights = numpy.array([2, 1.25, 2, 1.25, 2.75, 1.25, 2, 1.25, 2])


def evaluate_line(line):
    x_count = numpy.sum(line == 1)
    o_count = numpy.sum(line == -1)
    empty_count = 3 - (x_count + o_count)
    if x_count == 2 and empty_count == 1:
        return 20
    elif o_count == 2 and empty_count == 1:
        return -20
    elif x_count == 1 and empty_count == 2:
        return 6
    elif o_count == 1 and empty_count == 2:
        return -6
    else:
        return 0


def board2hash(board):
    board_fixed = numpy.where(numpy.isnan(board), 0, board)
    return numpy.dot(board_fixed.reshape(-1) + 1, 3 ** numpy.arange(9))


def hash2board(hash_value):
    board = numpy.zeros(9, dtype=numpy.int8)
    for i in range(9):
        board[i] = hash_value % 3
        hash_value //= 3
    return board.reshape(3, 3) - 1


def check_win(matrix):
    for i in range(3):

        row_sum = numpy.sum(matrix[i, :])
        col_sum = numpy.sum(matrix[:, i])
        if abs(row_sum) == 3:
            return int(numpy.sign(row_sum))
        if abs(col_sum) == 3:
            return int(numpy.sign(col_sum))
    diag1 = numpy.sum(numpy.diag(matrix))
    diag2 = numpy.sum(numpy.diag(numpy.fliplr(matrix)))
    if abs(diag1) == 3:
        return int(numpy.sign(diag1))
    if abs(diag2) == 3:
        return int(numpy.sign(diag2))
    return 0


def evaluate_state(hash_value):
    matrix = hash2board(hash_value)
    winner = check_win(matrix)
    if winner == 1:
        return 100
    elif winner == -1:
        return -100
    total_value = 0
    for i in range(3):
        total_value += evaluate_line(matrix[i, :]) + evaluate_line(matrix[:, i])
    total_value += evaluate_line(matrix.diagonal()) + evaluate_line(numpy.fliplr(matrix).diagonal())
    return total_value


def generate_state_dict():
    state_dict = {}
    for i in range(3 ** 9):
        hash_value = i
        if hash_value not in state_dict:
            total_value = evaluate_state(hash_value)
            state_dict[hash_value] = total_value
    return state_dict


def fieldvalues(fieldliste):
    fieldliste = numpy.asarray(fieldliste)
    weighted_fieldliste = fieldliste*weights

    line_values = numpy.sum(weighted_fieldliste[positions_array], axis=1)

    score = (numpy.mean(line_values) + numpy.max(line_values) - numpy.min(line_values)) / 2

    return score

def rollout_simulation_random_wrapper(state, _):
    return rollout_simulation_random(state)
def rollout_simulation_random(state):
    current_state = copy.deepcopy(state)
    while not current_state.game_over():
        possible_moves = current_state.get_possible_moves()
        if not possible_moves:
            break
        action = random.choice(possible_moves)
        current_state = current_state.make_move_sim(action)
    return current_state.game_result()


def rollout_simulation_ai(state, state_dictionary):
    POW3 = [3 ** i for i in range(9)]
    current_state = copy.deepcopy(state)

    b2h = board2hash
    s_d = state_dictionary

    while not current_state.game_over():
        moves = current_state.get_possible_moves()
        if not moves:
            break

        action1 = random.choice(moves)
        moves.remove(action1)
        action2 = random.choice(moves) if moves else action1

        fieldliste = [s_d[b2h(current_state.which_board((i, j)))] for i in range(3) for j in range(3)]

        fieldliste1 = fieldliste.copy()
        fieldliste1[action1[0][0] * 3 + action1[0][1]] = s_d[
            b2h(current_state.which_board(action1[0])) + POW3[action1[1][0] * 3 + action1[1][1]]]
        single_value1 = fieldvalues(fieldliste1)

        fieldliste2 = fieldliste.copy()
        fieldliste2[action2[0][0] * 3 + action2[0][1]] = s_d[
            b2h(current_state.which_board(action2[0])) + POW3[action2[1][0] * 3 + action2[1][1]]]
        single_value2 = fieldvalues(fieldliste2)

        chosen_action = action1 if single_value1 >= single_value2 else action2

        current_state = current_state.make_move_sim(chosen_action)

    return current_state.game_result()

class MCTSNode:
    def __init__(self, state, rollout = rollout_simulation_random, state_dictionary=None, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.results = defaultdict(int)
        self.untried_actions = self.get_possible_moves()
        self.rollout = rollout
        self.state_dictionary = state_dictionary

    def get_possible_moves(self):
        return self.state.get_possible_moves()

    def game_result(self):
        return self.state.game_result()

    def game_over(self):
        return self.state.game_over()

    def expand(self):
        action = self.untried_actions.pop()
        statecopy = copy.deepcopy(self.state)
        next_state = statecopy.make_move_sim(action)
        child_node = MCTSNode(next_state, rollout=self.rollout,  parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param=1):
        player = self.state.current_player
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float("inf")
            else:
                if player == 1:
                    value = (child.results[1] - child.results[-1]) / child.visits
                else:
                    value = (child.results[-1] - child.results[1]) / child.visits
                weight = value + exploration_param * numpy.sqrt((2 * numpy.log(self.visits)) / child.visits)
            choices_weights.append(weight)
        best_index = numpy.argmax(choices_weights)
        return self.children[best_index]

    def tree_policy(self):
        current_node = self
        while not current_node.game_over():
            if not current_node.fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number=1000, batch_size=10):
        sims_remaining = simulations_number
        with Pool() as pool:
            while sims_remaining > 0:
                nodes = []
                states_for_rollout = []
                for _ in range(min(batch_size, sims_remaining)):
                    node = self.tree_policy()
                    nodes.append(node)
                    states_for_rollout.append((node.state, self.state_dictionary))
                if self.rollout == rollout_simulation_random:
                    results = pool.starmap(rollout_simulation_random_wrapper, states_for_rollout)
                else:
                    results = pool.starmap(self.rollout, states_for_rollout)
                for node, reward in zip(nodes, results):
                    node.backpropagate(reward)
                sims_remaining -= batch_size
        best = max(self.children, key=lambda c: c.visits)
        return best

wincombs = numpy.matrix([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1, 0, 1, 0, 0]], dtype=int)

def check_winner(board):
    nozeroboard = numpy.where(numpy.isnan(board), 0, board)
    wins = wincombs * nozeroboard.reshape(9, 1)
    if (wins == 3).any():
        return 1
    if (wins == -3).any():
        return -1
    if not numpy.any(numpy.isnan(board)):
        return 0
    return numpy.nan

class UltimateBoard:
    def __init__(self):
        self.boards = numpy.full((9, 9), numpy.nan)
        self.board_wins = numpy.full((3, 3), numpy.nan)
        self.winner = numpy.nan
        self.active_board = None
        self.current_player = 1

    def which_board(self, boardid):
        row, col = boardid
        return self.boards[row * 3:(row + 1) * 3, col * 3:(col + 1) * 3]

    def update_board_wins(self, boardid):
        small_board = self.which_board(boardid)
        result = check_winner(small_board)
        if not numpy.isnan(result):
            self.board_wins[boardid] = result

    def big_winner(self):
        self.winner = check_winner(self.board_wins)

    def make_move(self, boardid, cellid):
        if not numpy.isnan(self.winner):
            return False
        if self.active_board and boardid != self.active_board:
            return False
        small_board = self.which_board(boardid)
        cellrow, cellcol = cellid
        if not numpy.isnan(small_board[cellrow, cellcol]):
            return False
        small_board[cellrow, cellcol] = self.current_player
        self.update_board_wins(boardid)

        nextboard_id = cellid
        next_small_board = self.which_board(nextboard_id)
        if not numpy.isnan(check_winner(next_small_board)) or numpy.all(~numpy.isnan(next_small_board)):
            self.active_board = None
        else:
            self.active_board = nextboard_id
        self.current_player = -self.current_player
        self.big_winner()

        return True

    def get_possible_moves(self):
        if not numpy.isnan(self.winner):
            return []
        moves = []
        if self.active_board is not None:
            boards_to_check = [self.active_board]
        else:
            boards_to_check = [(v1, v2) for v1 in range(3) for v2 in range(3)]
        for boardid in boards_to_check:
            if not numpy.isnan(self.board_wins[boardid]):
                continue
            small_board = self.which_board(boardid)
            empty = list(zip(*numpy.where(numpy.isnan(small_board))))
            for cell in empty:
                moves.append((boardid, tuple(cell)))
        return moves

    def game_over(self):
        if not numpy.isnan(self.winner):
            return True
        if len(self.get_possible_moves()) == 0:
            self.winner = 0
            return True
        return False

    def game_result(self):
        return self.winner

    def make_move_sim(self, action):
        boardid, cellid = action
        self.make_move(boardid, cellid)
        return self


class UltimateBoardGUI:
    def __init__(self, root, state, cell_size=50, margin=20):
        self.root = root
        self.state = state
        self.cell_size = cell_size
        self.margin = margin
        canvas_size = cell_size * 9 + margin * 2
        self.canvas = tk.Canvas(root, width=canvas_size, height=canvas_size)
        self.canvas.pack()

    def draw_board(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill="white")
        for row in range(9):
            for col in range(9):
                x1 = self.margin + col * self.cell_size
                y1 = self.margin + row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                board_row = row // 3
                board_col = col // 3
                if self.state.active_board is not None:
                    if (board_row, board_col) == self.state.active_board:
                        fill_color = "lightyellow"
                    else:
                        fill_color = "white"
                else:
                    fill_color = "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="gray")
                value = self.state.boards[row, col]
                if not numpy.isnan(value):
                    if value == 1:
                        text = "X"
                        color = "blue"
                    elif value == -1:
                        text = "O"
                        color = "red"
                    else:
                        text = ""
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=text, fill=color,
                                            font=("Arial", int(self.cell_size/2)))
        for i in range(4):
            x = self.margin + i * 3 * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.margin + 9 * self.cell_size, width=3)
            y = self.margin + i * 3 * self.cell_size
            self.canvas.create_line(self.margin, y, self.margin + 9 * self.cell_size, y, width=3)
        for board_row in range(3):
            for board_col in range(3):
                if not numpy.isnan(self.state.board_wins[board_row, board_col]):
                    center_x = self.margin + (board_col * 3 + 1.5) * self.cell_size
                    center_y = self.margin + (board_row * 3 + 1.5) * self.cell_size
                    if self.state.board_wins[board_row, board_col] == 1:
                        self.canvas.create_text(center_x, center_y, text="X", fill="blue",
                                                font=("Arial", int(self.cell_size*2)))
                    elif self.state.board_wins[board_row, board_col] == -1:
                        self.canvas.create_text(center_x, center_y, text="O", fill="red",
                                                font=("Arial", int(self.cell_size*2)))
        self.root.update()

    def get_human_move(self, state):
        self.selected_move = None

        def on_click(event):
            if event.x < self.margin or event.y < self.margin:
                return
            col = (event.x - self.margin) // self.cell_size
            row = (event.y - self.margin) // self.cell_size
            if col >= 9 or row >= 9:
                return
            board_id = (row // 3, col // 3)
            cell_id = (row % 3, col % 3)
            move = (board_id, cell_id)
            if move in state.get_possible_moves():
                self.selected_move = move
                self.canvas.unbind("<Button-1>")

        self.canvas.bind("<Button-1>", on_click)
        while self.selected_move is None:
            self.root.update()
            time.sleep(0.05)
        return self.selected_move


def select_move(state, strategy, state_dictionary, simulation_count, gui=None):
    if strategy == "human":
        return gui.get_human_move(state)
    elif strategy == "mcts_random":
        root_node = MCTSNode(copy.deepcopy(state), rollout=rollout_simulation_random, state_dictionary=state_dictionary)
        best_node = root_node.best_action(simulation_count)
        move= best_node.parent_action
        if move is None:
            move = random.choice(state.get_possible_moves())
        return move
    elif strategy == "mcts_calculated":
        root_node = MCTSNode(copy.deepcopy(state), rollout=rollout_simulation_ai, state_dictionary=state_dictionary)
        best_node = root_node.best_action(simulation_count)
        move = best_node.parent_action
        if move is None:
            move = random.choice(state.get_possible_moves())
        return move
    elif strategy == "random":
        return random.choice(state.get_possible_moves())
    else:
        raise ValueError("Unknown strategy")


def simulate_game(simulations_for_X=1000, simulations_for_O=1000, gui=None, player_strategies = None, state_dictionary =None):
    state = UltimateBoard()
    if gui is not None:
        gui.state = state
        gui.draw_board()
        time.sleep(0.5)
    move_counter = 1
    move_times = []
    if player_strategies is None:
        player_strategies = {1: "mcts_calculated", -1: "mcts_random"}
    if state_dictionary is None:
        state_dictionary = generate_state_dict()

    while not state.game_over():
        possible_moves = state.get_possible_moves()
        if not possible_moves:
            break

        current_player = state.current_player
        simulation_count = simulations_for_X if current_player == 1 else simulations_for_O

        start_time = time.perf_counter()
        strategy = player_strategies[current_player]
        move = select_move(state, strategy, state_dictionary, simulation_count, gui)
        duration = time.perf_counter() - start_time
        move_times.append(duration)

        print(f"Move {move_counter} (MCTS - Player {current_player}): Board ({move[0][0]}, {move[0][1]}), Cell ({move[1][0]}, {move[1][1]}) (Duration: {duration:.4f} s)")
        if move is None:
            print("MCTS did not find a move. Choosing a random move.")
            move = random.choice(possible_moves)
        state.make_move(*move)
        if gui is not None:
            gui.draw_board()
            time.sleep(0.5)
        move_counter += 1

    if state.winner == 0:
        result = "draw"
    else:
        result = state.winner
    avg_move_time = sum(move_times) / len(move_times) if move_times else 0
    print(f"Average move duration in this game: {avg_move_time:.4f} s")
    return result, move_times


if __name__ == "__main__":
    NUM_GAMES = 100
    results = {"1": 0, "-1": 0, "draw": 0}
    total_move_time = 0.0
    total_moves = 0

    root = tk.Tk()
    root.title("Ultimate Tic Tac Toe Simulation")

    info_label = tk.Label(root, text="Starting simulation...", font=("Arial", 14))
    info_label.pack(pady=5)

    initial_state = UltimateBoard()
    gui = UltimateBoardGUI(root, initial_state)

    state_dictionary = generate_state_dict()
    player_strategies = {1: "mcts_calculated", -1: "mcts_random"}

    for game in range(1, NUM_GAMES + 1):
        info_label.config(text=f"Game {game} of {NUM_GAMES}\nSimulating...")
        root.update()

        winner, move_times = simulate_game(simulations_for_X=1000, simulations_for_O=1000, gui=gui, player_strategies=player_strategies, state_dictionary=state_dictionary)
        if winner == "draw":
            results["draw"] += 1
        else:
            results[str(winner)] += 1
        total_move_time += sum(move_times)
        total_moves += len(move_times)
        avg_move_time = total_move_time / total_moves if total_moves > 0 else 0
        avg_moves_per_game = total_moves / game

        info_label.config(text=(f"Game {game} of {NUM_GAMES}\n"
                                f"Winner: {winner}\n"
                                f"Average move duration: {avg_move_time:.4f} s\n"
                                f"Average moves per game: {avg_moves_per_game:.2f}"))
        print ((f"Game {game} of {NUM_GAMES}\n"
                                f"Winner: {winner}\n"
                                f"Average move duration: {avg_move_time:.4f} s\n"
                                f"Player X wins: {results['1']}\n"
                                f"Player O wins: {results['-1']}\n"
                                f"Average moves per game: {avg_moves_per_game:.2f}"))
        root.update()
        time.sleep(1)

    overall_avg_move_time = total_move_time / total_moves if total_moves > 0 else 0
    overall_avg_moves = total_moves / NUM_GAMES if NUM_GAMES > 0 else 0
    summary = (f"Simulation finished after {NUM_GAMES} games.\n"
               f"Player X wins: {results['1']}\n"
               f"Player O wins: {results['-1']}\n"
               f"Draws: {results['draw']}\n"
               f"Average move duration: {overall_avg_move_time:.4f} s\n"
               f"Average moves per game: {overall_avg_moves:.2f}")
    print(summary)
    info_label.config(text=summary)
    root.mainloop()

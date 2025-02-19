import random
import copy
import numpy
import time
from collections import defaultdict
from multiprocessing import Pool
import tkinter as tk

positions = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Zeilen
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Spalten
    [0, 4, 8], [2, 4, 6]  # Diagonalen
]

positions_array = numpy.array(positions)

weights = numpy.array([2, 1.25, 2, 1.25, 2.75, 1.25, 2, 1.25, 2]) # Gewichtung der Felder


def evaluate_line(line): #Bewertung der Gewinnlinien
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


def board2hash(board): #Konvertierung von board(matrix) zu hash
    board_fixed = numpy.where(numpy.isnan(board), 0, board)
    return numpy.dot(board_fixed.reshape(-1) + 1, 3 ** numpy.arange(9))


def hash2board(hash_value): #Konvertierung von hash zu board(matrix)
    board = numpy.zeros(9, dtype=numpy.int8)
    for i in range(9):
        board[i] = hash_value % 3
        hash_value //= 3
    return board.reshape(3, 3) - 1


def check_win(matrix): #Sucht nach Gewinn in der 3x3 Matrix für evaluate_state
    for i in range(3):
    #linien
        row_sum = numpy.sum(matrix[i, :])
        col_sum = numpy.sum(matrix[:, i])
        if abs(row_sum) == 3:
            return int(numpy.sign(row_sum))
        if abs(col_sum) == 3:
            return int(numpy.sign(col_sum))
    #diagonalen
    diag1 = numpy.sum(numpy.diag(matrix))
    diag2 = numpy.sum(numpy.diag(numpy.fliplr(matrix)))
    if abs(diag1) == 3:
        return int(numpy.sign(diag1))
    if abs(diag2) == 3:
        return int(numpy.sign(diag2))
    return 0


def evaluate_state(hash_value): #Bewertung des Zustands
    matrix = hash2board(hash_value)
    winner = check_win(matrix)
    if winner == 1: #Bei Gewinner 100 Punkte für Gewinner
        return 100
    elif winner == -1:
        return -100
    total_value = 0
    for i in range(3): #sonst nach evaluate_line bewertet
        total_value += evaluate_line(matrix[i, :]) + evaluate_line(matrix[:, i])
    total_value += evaluate_line(matrix.diagonal()) + evaluate_line(numpy.fliplr(matrix).diagonal())
    return total_value


def generate_state_dict(): #generiert und bewertet alle möglichen 3x3 Zustände in einem Dictionary
    state_dict = {}
    for i in range(3 ** 9):
        hash_value = i
        if hash_value not in state_dict:
            total_value = evaluate_state(hash_value)
            state_dict[hash_value] = total_value
    return state_dict


def fieldvalues(fieldliste): #Bewertung der Felder für Rollout
    fieldliste = numpy.asarray(fieldliste)
    weighted_fieldliste = fieldliste*weights

    line_values = numpy.sum(weighted_fieldliste[positions_array], axis=1)

    score = (numpy.mean(line_values) + numpy.max(line_values) - numpy.min(line_values)) / 2

    return score

def rollout_simulation_random_wrapper(state, _): #Lösung für verschiedene Mengen an Argumenten in verschiedenen Rollout-Funktionen
    return rollout_simulation_random(state)
def rollout_simulation_random(state): #Rollout mit zufälligen Zügen
    current_state = copy.deepcopy(state)
    while not current_state.game_over():
        possible_moves = current_state.get_possible_moves()
        if not possible_moves:
            break
        action = random.choice(possible_moves)
        current_state = current_state.make_move_sim(action)
    return current_state.game_result()


def rollout_simulation_ai(state, state_dictionary):
    POW3 = [3 ** i for i in range(9)] #Die Liste wird einmalig vorab berechnet, um schneller die Potenzen von 3 abrufen zu können
    current_state = copy.deepcopy(state)

    b2h = board2hash #neue Variablen für Übersichtlichkeit
    s_d = state_dictionary

    while not current_state.game_over():
        moves = current_state.get_possible_moves()
        if not moves:
            break

        action1 = random.choice(moves) #2 Zufällige unterschiedliche Zügen
        moves.remove(action1)
        action2 = random.choice(moves) if moves else action1

        fieldliste = [s_d[b2h(current_state.which_board((i, j)))] for i in range(3) for j in range(3)] #Vorberechnung von Fieldliste, damit statische Boards nicht mehrmals berechnet werden.

        fieldliste1 = fieldliste.copy()
        fieldliste1[action1[0][0] * 3 + action1[0][1]] = s_d[                                    #Endbewertung des ersten zufälligen Zuges
            b2h(current_state.which_board(action1[0])) + POW3[action1[1][0] * 3 + action1[1][1]]]
        single_value1 = fieldvalues(fieldliste1)

        fieldliste2 = fieldliste.copy()
        fieldliste2[action2[0][0] * 3 + action2[0][1]] = s_d[                                    #Endbewertung des zweiten zufälligen Zuges
            b2h(current_state.which_board(action2[0])) + POW3[action2[1][0] * 3 + action2[1][1]]]
        single_value2 = fieldvalues(fieldliste2)

        chosen_action = action1 if single_value1 >= single_value2 else action2 #Wählt den besseren Zug

        current_state = current_state.make_move_sim(chosen_action)

    return current_state.game_result()

class MCTSNode:
    def __init__(self, state, rollout = rollout_simulation_random, state_dictionary=None, parent=None, parent_action=None): #Initialisierung
        self.state = state #Spielstand
        self.parent = parent #vorheriger Spielstand
        self.parent_action = parent_action #Aktion, die zum aktuellen Spielstand geführt hat
        self.children = [] #ausgeführte Aktionen
        self.visits = 0 #Anzahl der Besuche
        self.results = defaultdict(int) #Ergebnisse der Simulationen
        self.untried_actions = self.get_possible_moves() #mögliche Aktionen(noch nicht getestet)
        self.rollout = rollout #rollout Funktion (abhängig von Spieler)
        self.state_dictionary = state_dictionary #Dictionary mit Bewertungen der möglichen Zustände

    def get_possible_moves(self): #Alle möglichen Aktionen
        return self.state.get_possible_moves()

    def game_result(self): #Ergebnis des Spiels
        return self.state.game_result()

    def game_over(self): #Spiel vorbei?
        return self.state.game_over()

    def expand(self): #Erweitern des Baums
        action = self.untried_actions.pop() #Nächste Aktion, entfernt Aktion aus untried_actions
        statecopy = copy.deepcopy(self.state) #Kopie des aktuellen Zustands, um Änderungen zu vermeiden
        next_state = statecopy.make_move_sim(action) #Neuer Zustand nach Aktion
        child_node = MCTSNode(next_state, rollout=self.rollout,  parent=self, parent_action=action) #Neuer Knoten auf diesem Zustand
        self.children.append(child_node) #Hinzufügen des Knotens zu den Kindern des aktuellen Knotens
        return child_node

    def backpropagate(self, result): #Rückgabe der Ergebnisse
        self.visits += 1
        self.results[result] += 1 #Ergebnis hinzufügen
        if self.parent:
            self.parent.backpropagate(result)

    def fully_expanded(self): #Alle Aktionen getestet?
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param=1): #Beste Aktion, nutzt die MCTS-Bewertung
        player = self.state.current_player
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float("inf") #probiert als erstes noch nicht versuchte Aktionen
            else:
                if player == 1:
                    value = (child.results[1] - child.results[-1]) / child.visits #verschiedene Bewertungsfunktionen für Spieler 1 und -1
                else:
                    value = (child.results[-1] - child.results[1]) / child.visits
                weight = value + exploration_param * numpy.sqrt((2 * numpy.log(self.visits)) / child.visits) #UCT-Formel
            choices_weights.append(weight)
        best_index = numpy.argmax(choices_weights) #Bestes Kind (höchster Wert)
        return self.children[best_index]

    def tree_policy(self): #Entscheidung über die Aktion (Expand oder best_child) in best_action
        current_node = self
        while not current_node.game_over():
            if not current_node.fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number=1000, batch_size=10): #Beste Aktion nach Anzahl der Simulationen, nutzt Multiprocessing
        sims_remaining = simulations_number
        with Pool() as pool:
            while sims_remaining > 0: #Solange noch Simulationen übrig sind
                nodes = []
                states_for_rollout = []
                for _ in range(min(batch_size, sims_remaining)): #Wählt die Simulationen aus
                    node = self.tree_policy()
                    nodes.append(node)
                    states_for_rollout.append((node.state, self.state_dictionary))
                if self.rollout == rollout_simulation_random: #Führt Simulationen je nach Rollout-Funktion aus
                    results = pool.starmap(rollout_simulation_random_wrapper, states_for_rollout)
                else:
                    results = pool.starmap(self.rollout, states_for_rollout)
                for node, reward in zip(nodes, results): #Gibt die Bewertung zurück
                    node.backpropagate(reward)
                sims_remaining -= batch_size #Aktualisiert die fehlenden Simulationen
        best = max(self.children, key=lambda c: c.visits) #Wählt das am häufigsten besuchte Kind
        return best

wincombs = numpy.matrix([[1, 1, 1, 0, 0, 0, 0, 0, 0], #Alle Gewinnkombinationen
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1, 0, 1, 0, 0]], dtype=int)

def check_winner(board): #Check_Winner mittels Matrizenmultiplikation
    nozeroboard = numpy.where(numpy.isnan(board), 0, board)
    wins = wincombs * nozeroboard.reshape(9, 1)
    if (wins == 3).any():
        return 1
    if (wins == -3).any():
        return -1
    if not numpy.any(numpy.isnan(board)):
        return 0
    return numpy.nan

class UltimateBoard: #Spielzustandsobjekt
    def __init__(self):
        self.boards = numpy.full((9, 9), numpy.nan) #Gesamt-Board
        self.board_wins = numpy.full((3, 3), numpy.nan) #Gewinner-Board
        self.winner = numpy.nan
        self.active_board = None
        self.current_player = 1

    def which_board(self, boardid): #Ermittelt mit Koordinaten das 3x3 Board
        row, col = boardid
        return self.boards[row * 3:(row + 1) * 3, col * 3:(col + 1) * 3]

    def update_board_wins(self, boardid): #update des Gewinner-Boards nach Gewinn
        small_board = self.which_board(boardid)
        result = check_winner(small_board)
        if not numpy.isnan(result):
            self.board_wins[boardid] = result

    def big_winner(self): #Check für Gesamtgewinner
        self.winner = check_winner(self.board_wins)

    def make_move(self, boardid, cellid): #Zug machen, mithilfe von Koordinaten
        if not numpy.isnan(self.winner): #Board-Gewinner ist nicht leer
            return False
        if self.active_board and boardid != self.active_board: #richtiges Board?
            return False
        small_board = self.which_board(boardid) #3x3 Board
        cellrow, cellcol = cellid
        if not numpy.isnan(small_board[cellrow, cellcol]): #Zelle leer?
            return False
        small_board[cellrow, cellcol] = self.current_player #Macht Zug
        self.update_board_wins(boardid) #Update des Gewinner-Boards

        nextboard_id = cellid #Setzt das nächste Board, wo gespielt werden muss
        next_small_board = self.which_board(nextboard_id)
        if not numpy.isnan(check_winner(next_small_board)) or numpy.all(~numpy.isnan(next_small_board)): #Wurde das nächste Board schon gewonnen oder ist voll?
            self.active_board = None
        else:
            self.active_board = nextboard_id
        self.current_player = -self.current_player
        self.big_winner() #Check für Gesamtgewinner

        return True

    def get_possible_moves(self): #Alle möglichen Züge
        if not numpy.isnan(self.winner): #Spiel vorbei?
            return []
        moves = []
        if self.active_board is not None: #Begrenzt Suche auf aktives Board
            boards_to_check = [self.active_board]
        else:
            boards_to_check = [(v1, v2) for v1 in range(3) for v2 in range(3)] #Suche auf alle Boards
        for boardid in boards_to_check:
            if not numpy.isnan(self.board_wins[boardid]): #Board schon gewonnen?
                continue
            small_board = self.which_board(boardid)
            empty = list(zip(*numpy.where(numpy.isnan(small_board)))) #Leere Zellen
            for cell in empty:
                moves.append((boardid, tuple(cell)))
        return moves

    def game_over(self): #Spiel vorbei?
        if not numpy.isnan(self.winner):
            return True
        if len(self.get_possible_moves()) == 0:
            self.winner = 0
            return True
        return False

    def game_result(self): #Ergebnis des Spiels
        return self.winner

    def make_move_sim(self, action): #Zug machen, mithilfe von Aktion
        boardid, cellid = action
        self.make_move(boardid, cellid)
        return self

#Tkinter GUI
class UltimateBoardGUI:
    def __init__(self, root, state, cell_size=50, margin=20): #Initialisierung
        self.root = root #Fenster
        self.state = state #Spielzustand
        self.cell_size = cell_size #Größe der Zellen
        self.margin = margin #Rand
        canvas_size = cell_size * 9 + margin * 2 #Größe des Canvas
        self.canvas = tk.Canvas(root, width=canvas_size, height=canvas_size) #GUI
        self.canvas.pack() #Anzeigen

    def draw_board(self):
        self.canvas.delete("all")
        # Draw background
        self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill="white")
        # Draw each cell and its content
        for row in range(9):
            for col in range(9):
                x1 = self.margin + col * self.cell_size
                y1 = self.margin + row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                board_row = row // 3
                board_col = col // 3
                # Highlight the active small board if set
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
        # Draw thicker lines to separate small boards
        for i in range(4):
            x = self.margin + i * 3 * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.margin + 9 * self.cell_size, width=3)
            y = self.margin + i * 3 * self.cell_size
            self.canvas.create_line(self.margin, y, self.margin + 9 * self.cell_size, y, width=3)
        # Display winners for small boards
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
        self.root.update()  # Erzwingt die Aktualisierung des Fensters
    def get_human_move(self, state): #Menschlicher Zug
        self.selected_move = None #Ausgewählter Zug auf None setzen

        def on_click(event):#Klick-Event
            if event.x < self.margin or event.y < self.margin:
                return # Klick außerhalb des Spielfelds
            col = (event.x - self.margin) // self.cell_size #Klick-Koordinaten
            row = (event.y - self.margin) // self.cell_size
            if col >= 9 or row >= 9: #Klick außerhalb des Spielfelds
                return
            board_id = (row // 3, col // 3) #Koordinaten des Boards
            cell_id = (row % 3, col % 3) #Koordinaten der Zelle
            move = (board_id, cell_id) #Zug
            if move in state.get_possible_moves(): #Zug möglich?
                self.selected_move = move
                self.canvas.unbind("<Button-1>") # Klick-Event entfernen

        self.canvas.bind("<Button-1>", on_click) #Klick-Event hinzufügen
        while self.selected_move is None: #Solange kein Zug ausgewählt wurde immer nach Klicks suchen
            self.root.update()
            time.sleep(0.05)
        return self.selected_move


def select_move(state, strategy, state_dictionary, simulation_count, gui=None): #Wählt den Zug
    if strategy == "human": #Menschlicher Zug
        return gui.get_human_move(state)
    elif strategy == "mcts_random": #MCTS mit zufälligen Zügen
        root_node = MCTSNode(copy.deepcopy(state), rollout=rollout_simulation_random, state_dictionary=state_dictionary)
        best_node = root_node.best_action(simulation_count)
        move= best_node.parent_action
        if move is None:
            move = random.choice(state.get_possible_moves())
        return move
    elif strategy == "mcts_calculated": #MCTS mit berechneten Zügen
        root_node = MCTSNode(copy.deepcopy(state), rollout=rollout_simulation_ai, state_dictionary=state_dictionary)
        best_node = root_node.best_action(simulation_count)
        move = best_node.parent_action
        if move is None:
            move = random.choice(state.get_possible_moves())
        return move
    elif strategy == "random": #Zufälliger Zug
        return random.choice(state.get_possible_moves())
    else: #Falls keine Strategie bekannt
        raise ValueError("Unknown strategy")


def simulate_game(simulations_for_X=1000, simulations_for_O=1000, gui=None, player_strategies = None, state_dictionary =None): #Simulation eines Spiels, nutzt MCTS mit Simulationsanzahl
    state = UltimateBoard() #erstellt Spielzustandsobjekt
    if gui is not None: #erstellt GUI
        gui.state = state
        gui.draw_board()
        time.sleep(0.5)
    move_counter = 1 #Zuganzahl
    move_times = [] #Zeiten für Züge
    if player_strategies is None: #Falls keine Strategien angegeben sind, nutzt MCTS mit zufälligen Zügen
        player_strategies = {1: "mcts_calculated", -1: "mcts_random"}
    if state_dictionary is None: #Falls keine State-Dictionary angegeben ist, wird es generiert
        state_dictionary = generate_state_dict()

    while not state.game_over(): #Spiel läuft noch?
        possible_moves = state.get_possible_moves() #mögliche Züge
        if not possible_moves:
            break

        current_player = state.current_player #aktueller Spieler
        simulation_count = simulations_for_X if current_player == 1 else simulations_for_O #Wie viele Simulationen jetzt

        start_time = time.perf_counter() #Startzeit
        strategy = player_strategies[current_player] #Strategie des Spielers
        move = select_move(state, strategy, state_dictionary, simulation_count, gui) #Wählt den Zug
        duration = time.perf_counter() - start_time #Dauer des Zuges
        move_times.append(duration) #Zeit zu Liste hinzufügen

        print(f"Move {move_counter} (MCTS - Player {current_player}): Board ({move[0][0]}, {move[0][1]}), Cell ({move[1][0]}, {move[1][1]}) (Duration: {duration:.4f} s)") #Debug-Print in Konsole
        if move is None: #Falls keine Aktion gefunden wurde, sollte nicht passieren
            print("MCTS did not find a move. Choosing a random move.")
            move = random.choice(possible_moves)
        state.make_move(*move) #Zug machen
        if gui is not None: #GUI aktualisieren
            gui.draw_board()
            time.sleep(0.5)
        move_counter += 1 #Zuganzahl erhöhen

    if state.winner == 0: #Ergebnis des Spiels
        result = "draw"
    else:
        result = state.winner
    avg_move_time = sum(move_times) / len(move_times) if move_times else 0 #Durchschnittliche Zugdauer
    print(f"Average move duration in this game: {avg_move_time:.4f} s")
    return result, move_times

# ________________________________________________________________________#
#                                                                         #
#           Das Hauptprogramm, die GUI und die Statistiken                #
# ________________________________________________________________________#

if __name__ == "__main__":
    NUM_GAMES = 100  # Anzahl aller Spiele die Simuliert werden sollen
    results = {"1": 0, "-1": 0, "draw": 0} # Ergebnisse werden auf Null gesetzt
    total_move_time = 0.0 # Die Zeit (gemessene) pro Zug wird auf Null gesetzt
    total_moves = 0 # Die gemessene Anzahl an Zügen wird auf Null gesetzt

    root = tk.Tk() # Das Hauptfenster
    root.title("Ultimate Tic Tac Toe Simulation") # Der Fenstertitel

    info_label = tk.Label(root, text="Starting simulation...", font=("Arial", 14)) # Die Informationsanzeige
    info_label.pack(pady=5) # Das Label wird mit einem Abstand von 5 Pixeln nach oben positioniert

    initial_state = UltimateBoard() # Erzeugt den Anfangszustand der GUI
    gui = UltimateBoardGUI(root, initial_state) # Beginnt mit der Graphischen Initialisierung des Feldes (sorgt für Aktualisierungen)

    state_dictionary = generate_state_dict() # Die State Dictionary wird erstellt, die für die Bewertung von spielzuständen notwendig ist
                                             # (die Hashcodes werden mit bewertungen verknüpft, die man dadurch schneller aufrufen kann)
    player_strategies = {1: "mcts_calculated", -1: "mcts_random"} # Die Strategien der Spieler werden festgelegt, möglich ist: "mcts_calculated", "mcts_random", "random" und "human"

    for game in range(1, NUM_GAMES + 1): # Begrenzt die mögliche Anzahl von Spielen
        info_label.config(text=f"Game {game} of {NUM_GAMES}\nSimulating...") # Aktualisiert den Text des Labels
        root.update() # Aktualisiert das Fenster, um die Veränderung des Labels anzuzeigen

        winner, move_times = simulate_game(simulations_for_X=1000, simulations_for_O=1000, gui=gui, player_strategies=player_strategies, state_dictionary=state_dictionary) # Die Anzahl an Simulationen wird Festgelegt (die pro Zug gemacht werden)
        if winner == "draw": # Die Ergebnissstatistiken werden gezählt (draw)
            results["draw"] += 1
        else:
            results[str(winner)] += 1 # Die Ergebnissstatistiken werden gezählt (win/loose)
        total_move_time += sum(move_times) # Die Statistik der Gesamtzeit des Spiels
        total_moves += len(move_times) # Die Statistik der gesamten Züge des Spiels
        avg_move_time = total_move_time / total_moves if total_moves > 0 else 0 # Die Statistik der durchschnittlichen Zugzeit des Spiels
        avg_moves_per_game = total_moves / game # Die durchschnittliche Zuganzahl der Spiele

        # Die (oben berechneten) Statistiken werden jedes Spiel ausgegeben im Informationstext und in der Konsole
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
        root.update()  # Die GUI wird aktualisiert
        time.sleep(1)  # Es wird kurz angehalten, damit sich der Zuschauer/Spieler die Statistiken anschauen kann

    overall_avg_move_time = total_move_time / total_moves if total_moves > 0 else 0 # Die durchschnitliche Zeit pro Zug übber alle Simulationen wird berechnet
    overall_avg_moves = total_moves / NUM_GAMES if NUM_GAMES > 0 else 0 # Die durchschnittliche anzahl an Zügen übber alle Simulationen wird berechnet
    summary = (f"Simulation finished after {NUM_GAMES} games.\n" # Das Gesamtergebniss aller Spiele wird angezeigt
               f"Player X wins: {results['1']}\n"
               f"Player O wins: {results['-1']}\n"
               f"Draws: {results['draw']}\n"
               f"Average move duration: {overall_avg_move_time:.4f} s\n"
               f"Average moves per game: {overall_avg_moves:.2f}")
    print(summary) # Das Gesamtergebniss wird in der Console ausgegeben
    info_label.config(text=summary)
    root.mainloop()

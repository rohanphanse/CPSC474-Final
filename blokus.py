import random
import time

# Utility functions
def rotate_piece(piece):
    return [(-dy, dx) for (dx, dy) in piece]

def reflect_piece(piece):
    return [(-dx, dy) for (dx, dy) in piece]

def normalize_piece(piece):
    min_x = min(x for x, _ in piece)
    min_y = min(y for _, y in piece)
    normalized = sorted((x - min_x, y - min_y) for x, y in piece)
    return tuple(normalized)

def get_orientations(piece):
    orientations = []
    seen = set()
    cur = piece
    for i in range(4):
        for refl in [False, True]:
            temp = reflect_piece(cur) if refl else cur
            normalized = normalize_piece(temp)
            if normalized not in seen:
                seen.add(normalized)
                orientations.append((temp, i * 90, refl))
        cur = rotate_piece(cur)
    return orientations

# Data
pieces = {
    "plus": [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0)],   
    "toilet": [(0, 0), (0, 1), (1, 0), (1, -1), (2, 0)],     
    "z-shape-5": [(0, 0), (0, 1), (1, 0), (2, 0), (2, -1)],     
    "staircase": [(0, 0), (1, 0), (1, -1), (2, -1), (2, -2)],    
    "l-shape-5": [(0, 0), (0, -1), (0, -2), (1, -2), (2, -2)], 
    "t-shape-5": [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],      
    "twig": [(0, 0), (0, -1), (1, -1), (0, -2), (0, -3)],   
    "c-shape": [(0, 0), (1, 0), (1, -1), (1, -2), (0, -2)],  
    "hand": [(0, 0), (1, 0), (1, 1), (0, -1), (1, -1)],   
    "lightning": [(0, 0), (1, 0), (1, -1), (2, -1), (3, -1)],    
    "long-l-shape": [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)],    
    "line-5": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],      
    "z-shape-4": [(0, 0), (1, 0), (1, -1), (2, -1)],  
    "square-4": [(0, 0), (1, 0), (0, -1), (1, -1)],        
    "t-shape-4": [(0, 0), (1, 0), (1, -1), (2, 0)],          
    "l-shape-4": [(0, 0), (0, -1), (1, -1), (2, -1)],  
    "line-4": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "l-shape-3": [(0, 0), (0, -1), (1, -1)],            
    "line-3": [(0, 0), (1, 0), (2, 0)],      
    "line-2": [(0, 0), (1, 0)],
    "square-1": [(0, 0)],                              
}
num_pieces = len(pieces)
orientations = {piece_name: get_orientations(piece) for piece_name, piece in pieces.items()}
N = 14 # Board size
players = ["\033[34m□\033[0m", "\033[31m□\033[0m"]
players_filled = ["\033[34m■\033[0m", "\033[31m■\033[0m"]
moves = [[], []]

# State
class BlokusState:
    def __init__(self, board = None, hands = None, P = 0, passing = None):
        self.board = board if board else [["." for _ in range(N)] for _ in range(N)]
        self.hands = hands if hands else [list(pieces.keys()), list(pieces.keys())]
        self.P = P # Current player
        self.passing = passing if passing else [False, False]
    
    def display_board(self, action = None, player = 0):
        if action and action != "Pass":
            piece = self.get_piece(action[0], action[3], action[4])
            for r in range(N):
                for x in range(N):
                    y = N - 1 - r
                    found = False
                    for dx, dy in piece:
                        new_x = action[1] + dx
                        new_y = action[2] + dy
                        if new_x == x and new_y == y:
                            found = True
                            break
                    if found:
                        print(players_filled[player], end = " ")
                    else:
                        print(self.board[r][x], end = " ")
                print()  
        else:
            for row in self.board:
                print(" ".join(row))

    def get_piece(self, piece_name, rot = 0, refl = False):
        piece = pieces[piece_name]
        for _ in range(rot // 90):
            piece = rotate_piece(piece)
        if refl:
            piece = reflect_piece(piece)
        return piece

    def place_piece(self, piece_name, x, y, rot = 0, refl = False):
        piece = self.get_piece(piece_name, rot, refl)
        y = N - 1 - y
        for dx, dy in piece:
            self.board[y - dy][x + dx] = players[self.P]
        self.hands[self.P].remove(piece_name)

    def next_turn(self):
        self.P = 1 - self.P

    def is_valid_move(self, piece, x, y):
        y = N - 1 - y
        # Out of board
        if (x < 0 or x >= N) or (y < 0 or y >= N):
            return False
        for dx, dy in piece:
            new_x = x + dx
            new_y = y - dy
            if (new_x < 0 or new_x >= N) or (new_y < 0 or new_y >= N):
                return False
            # Spot already occupied
            if self.board[new_y][new_x] != ".":
                return False
        # First move
        if len(self.hands[self.P]) == num_pieces:
            for dx, dy in piece:
                new_x = x + dx
                new_y = y - dy
                if (self.P == 0 and new_x == 4 and new_y == 9) or (self.P == 1 and new_x == 9 and new_y == 4):
                    return True
            return False
        # Must touch one of the player's existing pieces by a corner but not an edge
        valid = False
        for dx, dy in piece:
            # Check corners
            for corner_x, corner_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_x = x + dx + corner_x
                new_y = y - dy + corner_y
                if (new_x < 0 or new_x >= N) or (new_y < 0 or new_y >= N):
                    continue
                if self.board[new_y][new_x] == players[self.P]:
                    valid = True
            # Check edges
            for edge_x, edge_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x = x + dx + edge_x
                new_y = y - dy + edge_y
                if (new_x < 0 or new_x >= N) or (new_y < 0 or new_y >= N):
                    continue
                if self.board[new_y][new_x] == players[self.P]:
                    return False
        return valid
                    
    def get_actions(self):
        if (self.passing[self.P]):
            return "Pass"
        valid_moves = []
        for piece_name in self.hands[self.P]:
            for orientation in orientations[piece_name]:
                for x in range(N):
                    for y in range(N):
                        if self.is_valid_move(orientation[0], x, y):
                            valid_moves.append((piece_name, x, y, orientation[1], orientation[2]))
        if len(valid_moves) == 0:
            return "Pass"
        return valid_moves
    
    def get_score(self, player):
        score = 0
        for piece_name in self.hands[player]:
            score += len(pieces[piece_name])
        return score
    
    def is_terminal(self):
        return (self.passing[0] and self.passing[1]) or (not self.hands[0] and not self.hands[1])
    
    def actor(self):
        return self.P

    def successor(self, action):
        new_state = self.copy()
        if action == "Pass":
            turn = self.P
            new_state.next_turn()
            if not new_state.passing[turn]:
                new_state.passing[turn] = True
            return new_state
        piece_name, x, y, rot, refl = action
        new_state.place_piece(piece_name, x, y, rot, refl)
        new_state.next_turn()
        return new_state

    def payoff(self):
        if self.is_terminal():
            p1_score = self.get_score(0)
            p2_score = self.get_score(1)
            return round((p2_score - p1_score) / (p1_score + p2_score + 1e-6), 5)
        return None
    
    def copy(self):
        new_board = [row.copy() for row in self.board]
        new_hands = [hand.copy() for hand in self.hands]
        new_passing = self.passing.copy()
        return BlokusState(new_board, new_hands, self.P, new_passing)
    
if __name__ == "__main__":
    state = BlokusState() # Initial state
    # Random agent vs. random agent
    while not state.is_terminal():
        actions = state.get_actions()
        random_action = actions if actions == "Pass" else random.choice(actions)
        moves[state.P].append(random_action)
        state = state.successor(random_action)
        print("\033[2J\033[H", end="")
        state.display_board()
        print(f"Player #{2 - state.P} chose: {random_action}")
        time.sleep(1)
    # print("Player #1 moves:", moves[0])
    # print("Player #2 moves:", moves[1])
    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))
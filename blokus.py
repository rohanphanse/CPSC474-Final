import random
import time

class Blokus:
    def __init__(self, N = 14):
        self.N = N # Board size
        self.board = [["." for _ in range(N)] for _ in range(N)]
        self.players = ["\033[34m□\033[0m", "\033[31m□\033[0m"]
        self.P = 0 # Current player
        self.pieces = {
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
        self.hands = [list(self.pieces.keys()), list(self.pieces.keys())]
        self.moves = [[], []]
        self.passing = [False, False]
        self.orientations = {}
        for piece_name, piece in self.pieces.items():
            self.orientations[piece_name] = self.get_orientations(piece)
    
    def display_board(self):
        for row in self.board:
            print(" ".join(row))

    def place_piece(self, piece_name, x, y, rot = 0, refl = False):
        piece = self.pieces[piece_name]
        for _ in range(rot // 90):
            piece = self.rotate_piece(piece)
        if refl:
            piece = self.reflect_piece(piece)
        y = self.N - 1 - y
        for dx, dy in piece:
            self.board[y - dy][x + dx] = self.players[self.P]
        self.hands[self.P].remove(piece_name)
        self.moves[self.P].append((piece_name, x, y, rot, refl))

    def switch_turn(self):
        self.P = 1 - self.P

    def is_valid_move(self, piece, x, y):
        y = self.N - 1 - y
        # Out of board
        if (x < 0 or x >= self.N) or (y < 0 or y >= self.N):
            return False
        for dx, dy in piece:
            new_x = x + dx
            new_y = y - dy
            if (new_x < 0 or new_x >= self.N) or (new_y < 0 or new_y >= self.N):
                return False
            # Spot already occupied
            if self.board[new_y][new_x] != ".":
                return False
        # First move
        if len(self.moves[self.P]) == 0:
            for dx, dy in piece:
                new_x = x + dx
                new_y = y - dy
                if self.P == 0 and new_x == 4 and new_y == 9:
                    return True
                if self.P == 1 and new_x == 9 and new_y == 4:
                    return True
            return False
        # Must touch one of the player's existing pieces by a corner but not an edge
        valid = False
        for dx, dy in piece:
            # Check corners
            for corner_x, corner_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_x = x + dx + corner_x
                new_y = y - dy + corner_y
                if (new_x < 0 or new_x >= self.N) or (new_y < 0 or new_y >= self.N):
                    continue
                if self.board[new_y][new_x] == self.players[self.P]:
                    valid = True
            # Check edges
            for edge_x, edge_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x = x + dx + edge_x
                new_y = y - dy + edge_y
                if (new_x < 0 or new_x >= self.N) or (new_y < 0 or new_y >= self.N):
                    continue
                if self.board[new_y][new_x] == self.players[self.P]:
                    return False
        return valid
                    
    def get_valid_moves(self):
        if (self.passing[self.P]):
            self.moves[self.P].append("Pass")
            return "Pass"
        valid_moves = []
        for piece_name in self.hands[self.P]:
            for orientation in self.orientations[piece_name]:
                for x in range(self.N):
                    for y in range(self.N):
                        if self.is_valid_move(orientation[0], x, y):
                            valid_moves.append((piece_name, x, y, orientation[1], orientation[2]))
        if len(valid_moves) == 0:
            self.passing[self.P] = True
            self.moves[self.P].append("Pass")
            return "Pass"
        return valid_moves
    
    def rotate_piece(self, piece):
        return [(-dy, dx) for (dx, dy) in piece]
    
    def reflect_piece(self, piece):
        return [(-dx, dy) for (dx, dy) in piece]
    
    def normalize_piece(self, piece):
        min_x = min(x for x, _ in piece)
        min_y = min(y for _, y in piece)
        normalized = sorted((x - min_x, y - min_y) for x, y in piece)
        return tuple(normalized)
    
    def get_orientations(self, piece):
        orientations = []
        seen = set()
        cur = piece
        for i in range(4):
            for refl in [False, True]:
                temp = self.reflect_piece(cur) if refl else cur
                normalized = self.normalize_piece(temp)
                if not normalized in seen:
                    seen.add(normalized)
                    orientations.append((temp, i * 90, refl))
            cur = self.rotate_piece(cur)
        return orientations
    
    def get_score(self, player):
        score = 0
        for piece_name in self.hands[player]:
            score += len(self.pieces[piece_name])
        return score
    
    def is_over(self):
        if len(game.moves[0]) > 0 and len(game.moves[1]) > 0 and game.moves[0][-1] == "Pass" and game.moves[1][-1] == "Pass":
            return True
        return False

    
if __name__ == "__main__":
    game = Blokus()
    while not game.is_over():
        valid_moves = game.get_valid_moves()
        if valid_moves == "Pass":
            game.switch_turn()
            continue
        piece_name, x, y, rot, refl = random.choice(valid_moves)
        game.place_piece(piece_name, x, y, rot, refl)
        game.switch_turn()
        print("\033[2J\033[H", end="")
        game.display_board()
        time.sleep(1)
    print("Player #1 moves:", game.moves[0])
    print("Player #2 moves:", game.moves[1])
    print("Player #1 remaining pieces:", game.hands[0])
    print("Player #2 remaining pieces:", game.hands[1])
    print("Player #1 score:", game.get_score(0))
    print("Player #2 score:", game.get_score(1))
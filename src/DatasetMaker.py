import chess
import chess.pgn
import numpy as np
import os
import h5py

# FONTIONS FOR DATASET 

# Fonction pour convertir un board en tenseur 8x8x17
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 17), dtype=np.float32)  

    piece_to_channel = {
        chess.PAWN:   {True: 0, False: 6}, #True = blanc, False=noir
        chess.KNIGHT: {True: 1, False: 7},
        chess.BISHOP: {True: 2, False: 8},
        chess.ROOK:   {True: 3, False: 9},
        chess.QUEEN:  {True: 4, False: 10},
        chess.KING:   {True: 5, False: 11},
    }

    # Placement des pièces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            channel = piece_to_channel[piece.piece_type][piece.color] 
            #piece.color renvoit true si blanc, false si noir
            tensor[row, col, channel] = 1

    # Tour de jeu (couleur)
    tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0

    # Droits de roque 
    tensor[:,:,13] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    tensor[:,:,14] =  1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    tensor[:,:,15] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0 
    tensor[:,:,16] =  1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    
    return tensor


def result_to_label(result):
    if result == "1-0":
        return 1 # victoire des blancs
    elif result == "0-1":
        return -1 # victoire des noirs
    else:
        return 0 # nulle

def move_to_tensor(move):
    """ Encode un coup en one-hot 8x8x73 """
    tensor = np.zeros((8, 8, 73), dtype=np.float32)

    start = move.from_square
    end = move.to_square

    row_start, col_start = 7 - (start // 8), start % 8

    move_index = get_move_index(start, end, move) # le coup parmi les 73 possibles
    
    if move_index is not None:
        tensor[row_start, col_start, move_index] = 1  
    else: raise ValueError(f"Erreur sur le move index")

    return tensor.flatten()


def get_move_index(start, end, move):
    """ Assigne un index unique à chaque type de mouvement """
    delta_row = (end // 8) - (start // 8)
    delta_col = (end % 8) - (start % 8)
    
    # (Tous les mouvements possibles de la reine pour les pièces)
    # Mouvement de [1...7] cases dans les 8 directions [N,NE,E,SE,S,SW,W,NW] = 0-55
    direction_map = {
        (1, 0): 0, (2, 0): 1, (3, 0): 2, (4, 0): 3, (5, 0): 4, (6, 0): 5, (7, 0): 6,  # ↑
        (-1, 0): 7, (-2, 0): 8, (-3, 0): 9, (-4, 0): 10, (-5, 0): 11, (-6, 0): 12, (-7, 0): 13,  # ↓
        (0, 1): 14, (0, 2): 15, (0, 3): 16, (0, 4): 17, (0, 5): 18, (0, 6): 19, (0, 7): 20,  # →
        (0, -1): 21, (0, -2): 22, (0, -3): 23, (0, -4): 24, (0, -5): 25, (0, -6): 26, (0, -7): 27,  # ←
        (1, 1): 28, (2, 2): 29, (3, 3): 30, (4, 4): 31, (5, 5): 32, (6, 6): 33, (7, 7): 34,  # ↗
        (-1, -1): 35, (-2, -2): 36, (-3, -3): 37, (-4, -4): 38, (-5, -5): 39, (-6, -6): 40, (-7, -7): 41,  # ↙
        (-1, 1): 42, (-2, 2): 43, (-3, 3): 44, (-4, 4): 45, (-5, 5): 46, (-6, 6): 47, (-7, 7): 48,  # ↖
        (1, -1): 49, (2, -2): 50, (3, -3): 51, (4, -4): 52, (5, -5): 53, (6, -6): 54, (7, -7): 55,  # ↘
    }

    # Les 8 mouvements du cavalier = 56-63
    knight_moves = {
        (2, 1): 56, # N N E
        (2, -1): 57, # N N W
        (-2, 1): 58, # S S E
        (-2, -1): 59, # S S W
        (1, 2): 60, # E E N
        (1, -2): 61, # W W N
        (-1, 2): 62, # E E S
        (-1, -2): 63, # W W S
    }

    # Promotions (64-72)
    if move.promotion:
        promo_map = {
            # Blancs
            (1, 1): {chess.KNIGHT: 64, chess.BISHOP: 67, chess.ROOK: 70, chess.QUEEN: 28},  # Promotion en haut-droite
            (1, -1): {chess.KNIGHT: 65, chess.BISHOP: 68, chess.ROOK: 71, chess.QUEEN: 49}, # Promotion en haut-gauche
            (1, 0): {chess.KNIGHT: 66, chess.BISHOP: 69, chess.ROOK: 72, chess.QUEEN: 0},  # Promotion en haut
            
            # Noirs
            (-1, 1): {chess.KNIGHT: 64, chess.BISHOP: 67, chess.ROOK: 70, chess.QUEEN:  42},  # Promotion en bas-droite
            (-1, -1): {chess.KNIGHT: 65, chess.BISHOP: 68, chess.ROOK: 71, chess.QUEEN: 35}, # Promotion en bas-gauche
            (-1, 0): {chess.KNIGHT: 66, chess.BISHOP: 69, chess.ROOK: 72, chess.QUEEN: 7},  # Promotion en bas
        }
        return promo_map[(delta_row, delta_col)][move.promotion]

    if (delta_row, delta_col) in knight_moves:
        return knight_moves[(delta_row, delta_col)]

    if (delta_row, delta_col) in direction_map:
        return direction_map[(delta_row, delta_col)]

    


# FONTIONS FOR PIPELINE

def parse_pgn_lazy(pgn_path):
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # Fin du fichier
            board = game.board()
            result = game.headers.get("Result", "1/2-1/2")
            result = result_to_label(result)

            for move in game.mainline_moves():
                tensor = board_to_tensor(board)
                move_tensor = move_to_tensor(move)

                yield tensor, move_tensor, result  # 💡 Ici, on ne stocke rien en RAM !

                board.push(move)  # Appliquer le coup


'''
def save_dataset_h5_lazy(pgn_path, filename, batch_size=65536):
    with h5py.File(f"{filename}.h5", "w", libver="latest", swmr=True, track_order=True) as f:
        f.create_dataset("tensors", shape=(0, 8, 8, 17), maxshape=(None, 8, 8, 17), dtype=np.float32, compression="gzip")
        f.create_dataset("moves", shape=(0, 4672), maxshape=(None, 4672), dtype=np.float32, compression="gzip")
        f.create_dataset("results", shape=(0,), maxshape=(None,), dtype=np.float32, compression="gzip")

        tensors, moves, results = [], [], []
        
        for i, (tensor, move_tensor, result) in enumerate(parse_pgn_lazy(pgn_path)):
            tensors.append(tensor)
            moves.append(move_tensor)
            results.append(result)

            # Sauvegarde par batch
            if (i + 1) % batch_size == 0:
                tensors_np = np.array(tensors, dtype=np.float32)
                moves_np = np.array(moves, dtype=np.float32)
                results_np = np.array(results, dtype=np.float32)

                # Agrandir les datasets
                f["tensors"].resize((f["tensors"].shape[0] + tensors_np.shape[0]), axis=0)
                f["moves"].resize((f["moves"].shape[0] + moves_np.shape[0]), axis=0)
                f["results"].resize((f["results"].shape[0] + results_np.shape[0]), axis=0)

                # Ajouter les nouvelles données
                f["tensors"][-tensors_np.shape[0]:] = tensors_np
                f["moves"][-moves_np.shape[0]:] = moves_np
                f["results"][-results_np.shape[0]:] = results_np

                # Reset buffers
                tensors, moves, results = [], [], []


                print(f"Ajouté {i + 1} positions au fichier {filename}.h5 ✅")

    print(f"Fichier {filename}.h5 complété ! 🎉")



def parse_pgn_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pgn"):
            file_path = os.path.join(folder_path, filename)
            print(f"📂 Traitement de {file_path}...")
            save_dataset_h5_lazy(file_path, f"../../ChessData/data/train/{filename}")
'''


def save_dataset_h5_lazy(pgn_path, output_folder, batch_size=65536, max_size=2**23, file_index=1, position_count=0, f=None):
    """
    Enregistre les positions d'échecs dans des fichiers HDF5, en découpant les fichiers
    lorsqu'on dépasse max_size positions.
    """
    def create_new_file():
        nonlocal file_index, position_count
        file_index += 1
        position_count = 0
        f = h5py.File(os.path.join(output_folder, f"dataset_{file_index:03d}.h5"), "w", libver="latest", swmr=True, track_order=True)
        
        # Création des datasets
        f.create_dataset("tensors", shape=(0, 8, 8, 17), maxshape=(None, 8, 8, 17), dtype=np.float32, compression="gzip")
        f.create_dataset("moves", shape=(0, 4672), maxshape=(None, 4672), dtype=np.float32, compression="gzip")
        f.create_dataset("results", shape=(0,), maxshape=(None,), dtype=np.float32, compression="gzip")

        print(f"🆕 Création du fichier {f.filename} avec datasets {list(f.keys())} ✅")
        return f


    if f is None:  # Créer un fichier si aucun n'est ouvert
        f = create_new_file()
        
    tensors, moves, results = [], [], []

    for tensor, move_tensor, result in parse_pgn_lazy(pgn_path):
        tensors.append(tensor)
        moves.append(move_tensor)
        results.append(result)
        position_count += 1
        
        # Sauvegarde et création d'un nouveau fichier si on atteint max_size
        if position_count >= max_size:
            if tensors:
                tensors_np = np.array(tensors, dtype=np.float32)
                moves_np = np.array(moves, dtype=np.float32)
                results_np = np.array(results, dtype=np.float32)

                f["tensors"].resize((f["tensors"].shape[0] + tensors_np.shape[0]), axis=0)
                f["moves"].resize((f["moves"].shape[0] + moves_np.shape[0]), axis=0)
                f["results"].resize((f["results"].shape[0] + results_np.shape[0]), axis=0)

                f["tensors"][-tensors_np.shape[0]:] = tensors_np
                f["moves"][-moves_np.shape[0]:] = moves_np
                f["results"][-results_np.shape[0]:] = results_np

                tensors, moves, results = [], [], []

            f.close()
            f = create_new_file()
            position_count = 0  # On remet à zéro car nouveau fichier
            
        # Sauvegarde partielle si on atteint batch_size
        if len(tensors) >= batch_size:
            tensors_np = np.array(tensors, dtype=np.float32)
            moves_np = np.array(moves, dtype=np.float32)
            results_np = np.array(results, dtype=np.float32)

            f["tensors"].resize((f["tensors"].shape[0] + tensors_np.shape[0]), axis=0)
            f["moves"].resize((f["moves"].shape[0] + moves_np.shape[0]), axis=0)
            f["results"].resize((f["results"].shape[0] + results_np.shape[0]), axis=0)

            f["tensors"][-tensors_np.shape[0]:] = tensors_np
            f["moves"][-moves_np.shape[0]:] = moves_np
            f["results"][-results_np.shape[0]:] = results_np

            tensors, moves, results = [], [], []
            print(f"Ajouté {position_count} positions au fichier dataset_{file_index:03d}.h5 ✅")
    
    if tensors:
        tensors_np = np.array(tensors, dtype=np.float32)
        moves_np = np.array(moves, dtype=np.float32)
        results_np = np.array(results, dtype=np.float32)

        f["tensors"].resize((f["tensors"].shape[0] + tensors_np.shape[0]), axis=0)
        f["moves"].resize((f["moves"].shape[0] + moves_np.shape[0]), axis=0)
        f["results"].resize((f["results"].shape[0] + results_np.shape[0]), axis=0)

        f["tensors"][-tensors_np.shape[0]:] = tensors_np
        f["moves"][-moves_np.shape[0]:] = moves_np
        f["results"][-results_np.shape[0]:] = results_np

        print(f"Fichier dataset_{file_index:03d}.h5 complété avec {position_count} positions ! 🎉")

    return file_index, position_count, f  # On retourne aussi le fichier ouvert

def parse_pgn_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_index, position_count = 1, 0  # Initialisation des compteurs
    f = None  # Fichier HDF5 en cours

    for filename in os.listdir(folder_path):
        if filename.endswith(".pgn"):
            file_path = os.path.join(folder_path, filename)
            print(f"📂 Traitement de {file_path}...")
            file_index, position_count, f = save_dataset_h5_lazy(file_path, output_folder, file_index=file_index, position_count=position_count, f=f)

    if f:  # Ne pas oublier de fermer le dernier fichier ouvert
        f.close()
# Utilisation

#dataset = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2019-10.pgn") # pgn de 117mo
#dataset = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2016-02.pgn") # pgn de 10,4 mo
#dataset = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2015-09.pgn") # pgn de 1,4 mo
#dataset = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2014-12.pgn") # pgn de 450ko
#datasetTrain = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2015-05.pgn") # pgn de 548ko
#datasetTest = parse_pgn("../../../ChessData/Lichess Elite Database/lichess_elite_2014-08.pgn") # pgn de 117ko
#dataset = parse_pgn("../../../ChessData/data/test.pgn")
#dataset = parse_pgn_folder("../../../ChessData/Lichess Elite Database/")

# pgn de 313mo
#save_dataset_h5_lazy("../../ChessData/Lichess Elite Database/lichess_elite_2020-05.pgn", "../../ChessData/data/train/datasetTrain")
 # pgn de 60mo
#save_dataset_h5_lazy("../../ChessData/Lichess Elite Database/lichess_elite_2019-07.pgn","../../ChessData/data/train")
parse_pgn_folder("../../ChessData/Lichess Elite Database/", "../../ChessData/data/train")
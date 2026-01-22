import chess.pgn
from pathlib import Path

def diagnose_pgn(pgn_path, num_games=5):
    """Diagnose PGN file format"""
    print(f"Analyzing: {pgn_path}\n")
    print("="*60)
    
    # Show first 30 lines of raw file
    print("First 30 lines of file:")
    print("-"*60)
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            print(f"{i+1:3d}: {line.rstrip()}")
    
    print("\n" + "="*60)
    print(f"\nParsing first {num_games} games with chess.pgn:")
    print("-"*60)
    
    # Parse games properly
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i in range(num_games):
            game = chess.pgn.read_game(f)
            
            if game is None:
                print(f"\nGame {i+1}: Could not parse (end of file or error)")
                break
            
            print(f"\nGame {i+1}:")
            print(f"  Event: {game.headers.get('Event', 'N/A')}")
            print(f"  White: {game.headers.get('White', 'N/A')}")
            print(f"  Black: {game.headers.get('Black', 'N/A')}")
            print(f"  WhiteElo: {game.headers.get('WhiteElo', 'N/A')}")
            print(f"  BlackElo: {game.headers.get('BlackElo', 'N/A')}")
            print(f"  Result: {game.headers.get('Result', 'N/A')}")
            
            # Count moves
            move_count = sum(1 for _ in game.mainline_moves())
            print(f"  Moves: {move_count}")
    
    print("\n" + "="*60)
    print("\nDiagnosis complete!")

if __name__ == "__main__":
    pgn_path = Path(__file__).parent.parent / "data" / "games.pgn"
    
    if not pgn_path.exists():
        print(f"Error: {pgn_path} not found!")
    else:
        diagnose_pgn(pgn_path)
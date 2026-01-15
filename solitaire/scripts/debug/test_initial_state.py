import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import SolitaireEnv
import numpy as np

env = SolitaireEnv()
obs, info = env.reset()
mask = env.action_masks()

print('='*60)
print('INITIAL STATE ANALYSIS')
print('='*60)
print(f'Available actions: {np.sum(mask)} / 87')
print(f'Surrender available: {mask[86]}')
print(f'Draw stock available: {mask[0]}')
print(f'Stock size: {len(env.stock)}')
print(f'Waste size: {len(env.waste)}')

# Symuluj 4 kroki z surrender
print('\n' + '='*60)
print('SIMULATING 4 STEPS WITH DIFFERENT ACTIONS')
print('='*60)

for step in range(1, 5):
    # Sprawdź które akcje są dostępne
    valid_actions = np.where(mask)[0]
    print(f'\nStep {step}:')
    print(f'  Valid actions: {len(valid_actions)}')
    print(f'  First 10 valid: {valid_actions[:10]}')
    print(f'  Surrender (86) valid: {mask[86]}')
    
    # Wykonaj akcję 0 (draw stock) jeśli dostępna
    if mask[0]:
        obs, reward, term, trunc, info = env.step(0)
        print(f'  Action: 0 (Draw Stock), Reward: {reward:.2f}, Done: {term or trunc}')
    else:
        # Weź pierwszą dostępną akcję (nie surrender)
        action = valid_actions[0] if valid_actions[0] != 86 else (valid_actions[1] if len(valid_actions) > 1 else 86)
        obs, reward, term, trunc, info = env.step(action)
        print(f'  Action: {action}, Reward: {reward:.2f}, Done: {term or trunc}')
    
    if term or trunc:
        reason = info.get('reason', 'unknown')
        print(f'  Episode ended! Reason: {reason}')
        print(f'  Score: {info["score"]:.2f}')
        print(f'  Moves: {info["moves"]}')
        break
    
    mask = env.action_masks()

# Test surrender
print('\n' + '='*60)
print('TEST SURRENDER ACTION')
print('='*60)
env.reset()
obs, reward, term, trunc, info = env.step(86)
print(f'Reward: {reward:.2f}')
print(f'Terminated: {term}')
print(f'Score: {info["score"]:.2f}')
print(f'Moves: {info["moves"]}')

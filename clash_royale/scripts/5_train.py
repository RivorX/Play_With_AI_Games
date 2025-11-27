import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import sys

# Dodaj katalog nadrzędny do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model import ClashRoyaleAgent
import numpy as np

class ClashDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path) # Lista krotek (obraz, akcja)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, action = self.data[idx]
        # Konwersja obrazu do tensora (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        card_idx, x, y = action
        target_card = torch.tensor(card_idx, dtype=torch.long)
        target_pos = torch.tensor([x, y], dtype=torch.float)
        
        return img, target_card, target_pos

def train():
    config_path = "clash_royale/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ClashRoyaleAgent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    
    # Loss functions
    criterion_card = nn.CrossEntropyLoss()
    criterion_pos = nn.MSELoss()
    
    # Sprawdź czy są dane
    data_file = os.path.join(config['paths']['processed_data'], "dataset.pt")
    if not os.path.exists(data_file):
        print("Brak przetworzonych danych. Uruchom najpierw data_processor.py")
        return

    dataset = ClashDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=True)
    
    model.train()
    for epoch in range(config['model']['epochs']):
        total_loss = 0
        for imgs, target_cards, target_pos in dataloader:
            imgs = imgs.to(device)
            target_cards = target_cards.to(device)
            target_pos = target_pos.to(device)
            
            # Symulacja eliksiru dla treningu (bo dataset jeszcze go nie ma)
            # W przyszłości trzeba dodać eliksir do datasetu w data_processor.py
            elixir = torch.randint(0, 11, (imgs.size(0), 1)).float().to(device)
            
            optimizer.zero_grad()
            
            pred_card_logits, pred_pos, _ = model(imgs, elixir)
            
            loss_card = criterion_card(pred_card_logits, target_cards)
            loss_pos = criterion_pos(pred_pos, target_pos)
            
            loss = loss_card + loss_pos
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
    # Zapisz model
    torch.save(model.state_dict(), config['paths']['model_save'])
    print("Model zapisany.")

if __name__ == "__main__":
    train()

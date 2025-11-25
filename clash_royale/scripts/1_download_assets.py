import json
import os
import requests
import yaml

def download_assets():
    url = "https://royaleapi.github.io/cr-api-data/json/cards.json"
    print(f"Pobieranie danych kart z {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        cards_data = response.json()
    except Exception as e:
        print(f"Błąd pobierania danych: {e}")
        return

    assets_dir = "clash_royale/assets/cards"
    os.makedirs(assets_dir, exist_ok=True)
    
    cards_config = {}
    
    print(f"Znaleziono {len(cards_data)} kart. Rozpoczynam pobieranie ikon...")
    
    for card in cards_data:
        name = card.get('key', card.get('name').lower().replace(' ', '-'))
        elixir = card.get('elixir', 0)
        
        # Zapisz koszt eliksiru do konfigu
        cards_config[name] = elixir
        
        # Konstruowanie URL ikony
        # Używamy repozytorium RoyaleAPI/cr-api-assets
        icon_url = f"https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/{name}.png"
            
        icon_path = os.path.join(assets_dir, f"{name}.png")
        
        if not os.path.exists(icon_path):
            try:
                img_data = requests.get(icon_url).content
                if len(img_data) > 0:
                    with open(icon_path, 'wb') as f:
                        f.write(img_data)
                    # print(f"Pobrano: {name}")
                else:
                    print(f"Pusty plik dla {name}")
            except Exception as e:
                print(f"Błąd pobierania ikony dla {name}: {e}")
        else:
            pass # Już istnieje
            
    print("Pobieranie zakończone.")
    
    # Aktualizacja pliku config.yaml
    config_path = "clash_royale/config/config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        
    config['cards'] = cards_config
    
    # Zapisz zaktualizowany config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Zaktualizowano {config_path} o koszty {len(cards_config)} kart.")

if __name__ == "__main__":
    download_assets()

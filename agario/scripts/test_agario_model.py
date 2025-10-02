import os
import yaml
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2
from stable_baselines3 import PPO
import torch

# Wczytaj konfigurację

# Wczytaj konfigurację względnie do lokalizacji skryptu
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ścieżka do modelu względna do lokalizacji skryptu
agario_dir = os.path.abspath(os.path.join(script_dir, '..'))
MODEL_PATH = os.path.join(agario_dir, config['paths']['model_path'])

# Custom env podobny do train, ale z real browser
class RealAgarIoEnv:
    def __init__(self, driver):
        self.driver = driver
        self.frame_history = config['environment']['frame_history']
        self.history = np.zeros((self.frame_history, 3, *config['environment']['screen_size']))
    
    def get_obs(self):
        screenshot = self.driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        crop = img[100:h-100, 100:w-100]
        crop = cv2.resize(crop, config['environment']['screen_size'])
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = crop
        return self.history[np.newaxis, ...]  # Batch dim
    
    def act(self, action):
        dx, dy, split_eject = action
        # Mysz move (uproszczone: pyautogui)
        import pyautogui
        pyautogui.moveRel(dx * 100, dy * 100)  # Skala
        if split_eject > 0.5:
            pyautogui.press('w')  # Split
        elif split_eject > 0:
            pyautogui.press(' ')  # Eject

def test_model():
    model = PPO.load(MODEL_PATH)
    model.policy.eval()
    
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://agar.io")
    time.sleep(5)
    
    env = RealAgarIoEnv(driver)
    
    start_time = time.time()
    scores = []
    while time.time() - start_time < 600:  # 10 min
        obs = env.get_obs()
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        env.act(action)
        time.sleep(0.1)  # FPS
    
    driver.quit()
    print("Test zakończony. Średni score:", np.mean(scores) if scores else 0)

if __name__ == "__main__":
    test_model()
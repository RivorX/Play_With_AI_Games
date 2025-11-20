from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv

class TrainProgressCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        # Create file with header if not exists
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length', 'win_rate'])

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                reward = info['episode']['r']
                length = info['episode']['l']
                # ✅ FIX: Sprawdzaj czy gra się skończyła z flagą terminal=True
                # W Pasjansie zwycięstwo ustawiamy flagę is_success w info
                is_success = info.get('is_success', False)
                
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_successes.append(is_success)
                
        # Log aggregated stats co jakiś czas (nie każdy epizod)
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_ep_length = np.mean(self.episode_lengths)
            win_rate = np.mean(self.episode_successes)
            
            # Log razem z info
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, mean_reward, mean_ep_length, win_rate])
                
        return True

class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, verbose=1, plot_script_path=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.plot_script_path = plot_script_path

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = self._evaluate()
            if self.verbose > 0:
                print(f"Eval at step {self.num_timesteps}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print("New best mean reward!")
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            
            # Zawsze nadpisuj aktualny model (solitaire_ppo_model.zip)
            self.model.save(os.path.join(self.best_model_save_path, "solitaire_ppo_model"))

            # Trigger plotting
            if self.plot_script_path:
                try:
                    import subprocess
                    subprocess.Popen(['python', self.plot_script_path])
                except Exception as e:
                    print(f"Failed to run plot script: {e}")
                
        return True

    def _evaluate(self):
        # Custom evaluation loop that handles Action Masks
        total_rewards = []
        
        for _ in range(5):  # n_eval_episodes
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # Get action masks from the environment
                # Note: eval_env is a VecEnv, so we need to access the method properly
                # sb3_contrib ActionMasker adds 'action_masks' method
                action_masks = self.eval_env.env_method("action_masks")
                # action_masks is a list of masks (one for each env in VecEnv)
                # Since we use n_envs=1 for eval, we take the first one
                action_mask = action_masks[0]
                
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0] # VecEnv returns list of rewards
                
            total_rewards.append(episode_reward)
            
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        return mean_reward, std_reward

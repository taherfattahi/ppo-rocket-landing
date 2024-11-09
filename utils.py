import numpy as np
import cv2
import pandas as pd
import json
import matplotlib as plt
import torch 
import os
import json
from datetime import datetime


################ Training Management ####################
def get_training_config():
    """Get training configuration from user"""
    config = {}
    
    print("\n====== Training Configuration ======")
    
    # Task and rocket selection
    while True:
        task = input("\nSelect task (hover/landing) [default: landing]: ").lower()
        if task in ['', 'hover', 'landing']:
            config['task'] = 'landing' if task == '' else task
            break
        print("Invalid choice. Please select 'hover' or 'landing'")
    
    while True:
        rocket = input("\nSelect rocket type (falcon/starship) [default: starship]: ").lower()
        if rocket in ['', 'falcon', 'starship']:
            config['rocket_type'] = 'starship' if rocket == '' else rocket
            break
        print("Invalid choice. Please select 'falcon' or 'starship'")
    
    # Visualization preferences
    config['render'] = input("\nEnable environment rendering? (y/n) [default: n]: ").lower() == 'y'
    config['plot_realtime'] = input("Enable real-time plotting? (y/n) [default: y]: ").lower() != 'n'
    config['save_plots'] = input("Save training plots? (y/n) [default: y]: ").lower() != 'n'
    
    # Training parameters
    config['max_episodes'] = int(input("\nEnter maximum episodes [default: 1000]: ") or 1000)
    config['save_freq'] = int(input("Save checkpoint frequency (episodes) [default: 100]: ") or 100)
    
    return config

################ Checkpoint Management ####################
def find_checkpoints(directory, env_name):
    """Find all available checkpoints"""
    checkpoints = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.startswith(f"PPO_{env_name}") and file.endswith(".pth"):
                checkpoints.append(file)
    return sorted(checkpoints)

def load_checkpoint(directory, env_name):
    """Handle checkpoint loading with user interaction"""
    checkpoints = find_checkpoints(directory, env_name)
    
    if not checkpoints:
        print("\nNo existing checkpoints found. Starting fresh training.")
        return None, None
    
    print("\n====== Available Checkpoints ======")
    for i, ckpt in enumerate(checkpoints):
        print(f"{i+1}. {ckpt}")
    
    while True:
        choice = input("\nSelect checkpoint number to load (or press Enter to start fresh): ")
        if choice == "":
            return None, None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return os.path.join(directory, checkpoints[idx]), checkpoints[idx]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

################ Logging Management ####################
def setup_logging(log_dir, env_name, run_num):
    """Setup logging with continuation support"""
    log_path = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')
    
    if os.path.exists(log_path):
        print(f"\nFound existing log file: {log_path}")
        choice = input("Continue logging to this file? (y/n) [default: n]: ").lower()
        if choice == 'y':
            return open(log_path, 'a'), True
    
    return open(log_path, 'w+'), False

################ Plot Management ####################
def setup_plotting(config):
    """Setup plotting based on configuration"""
    if not config['plot_realtime'] and not config['save_plots']:
        return None, None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    
    if config['plot_realtime']:
        plt.ion()
        plt.show(block=False)
    
    return fig, ax

def update_plots(fig, ax, episode_rewards, window_size, config, save_dir=None):
    """Update and optionally save plots"""
    if fig is None or ax is None:
        return
    
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_std = np.array([np.std(episode_rewards[i-window_size+1:i+1]) 
                             for i in range(window_size-1, len(episode_rewards))])
        episodes = np.arange(window_size-1, len(episode_rewards))
        
        ax.clear()
        ax.plot(episodes, moving_avg, label='Moving Average')
        ax.fill_between(episodes, moving_avg-moving_std, moving_avg+moving_std, 
                       alpha=0.2, label='Standard Deviation')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        
        if config['plot_realtime']:
            plt.draw()
            plt.pause(0.01)
        
        if config['save_plots'] and save_dir:
            plt.savefig(os.path.join(save_dir, f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            
################ Some helper functions... ####################

def moving_avg(x, N=500):

    if len(x) <= N:
        return []

    x_pad_left = x[0:N]
    x_pad_right = x[-N:]
    x_pad = x_pad_left[::-1] + x + x_pad_right[::-1]
    y = np.convolve(x_pad, np.ones(N) / N, mode='same')
    return y[N:-N]


def load_bg_img(path_to_img, w, h):
    bg_img = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img, (w, h))
    return bg_img


def create_circle_poly(center, radius, N=50):
    pts = []
    for i in range(N):
        x = center[0] + radius*np.cos(i/N*2*np.pi)
        y = center[1] + radius*np.sin(i/N*2*np.pi)
        pts.append([x, y])
    return pts


def create_ellipse_poly(center, rx, ry, N=50):
    pts = create_circle_poly(center, radius=1.0, N=N)
    for pt in pts:
        pt[0] = pt[0] * rx
        pt[1] = pt[1] * ry
    return pts


def create_rectangle_poly(center, w, h):
    x0, y0 = center
    pts = [[x0-w/2, y0+h/2], [x0+w/2, y0+h/2], [x0+w/2, y0-h/2], [x0-w/2, y0-h/2]]
    return pts


################ Let's do some math... ####################

def scale_matrix(sx=1.0, sy=1.0, sz=1.0):

    ScaleMatrix = np.eye(4)
    ScaleMatrix[0, 0] = sx  # scale on x
    ScaleMatrix[1, 1] = sy  # scale on y
    ScaleMatrix[2, 2] = sz  # scale on z

    return ScaleMatrix

def rotation_matrix(rx=0., ry=0., rz=0.):

    # input should be radians (e.g., 0, pi/2, pi)

    Rx = np.eye(4)
    Rx[1, 1] = np.cos(rx)
    Rx[1, 2] = -np.sin(rx)
    Rx[2, 1] = np.sin(rx)
    Rx[2, 2] = np.cos(rx)

    Ry = np.eye(4)
    Ry[0, 0] = np.cos(ry)
    Ry[0, 2] = np.sin(ry)
    Ry[2, 0] = -np.sin(ry)
    Ry[2, 2] = np.cos(ry)

    Rz = np.eye(4)
    Rz[0, 0] = np.cos(rz)
    Rz[0, 1] = -np.sin(rz)
    Rz[1, 0] = np.sin(rz)
    Rz[1, 1] = np.cos(rz)

    # RZ * RY * RX
    RotationMatrix = np.asmatrix(Rz) * np.asmatrix(Ry) * np.asmatrix(Rx)

    return np.array(RotationMatrix)


def translation_matrix(tx=0., ty=0., tz=0.):

    TranslationMatrix = np.eye(4)
    TranslationMatrix[0, -1] = tx
    TranslationMatrix[1, -1] = ty
    TranslationMatrix[2, -1] = tz

    return TranslationMatrix


def create_pose_matrix(tx=0., ty=0., tz=0.,
                       rx=0., ry=0., rz=0.,
                       sx=1.0, sy=1.0, sz=1.0,
                       base_correction=np.eye(4)):

    # Scale matrix
    ScaleMatrix = scale_matrix(sx, sy, sz)

    # Rotation matrix
    RotationMatrix = rotation_matrix(rx, ry, rz)

    # Translation matrix
    TranslationMatrix = translation_matrix(tx, ty, tz)

    # TranslationMatrix * RotationMatrix * ScaleMatrix
    PoseMatrix = np.asmatrix(TranslationMatrix) \
                 * np.asmatrix(RotationMatrix) \
                 * np.asmatrix(ScaleMatrix) \
                 * np.asmatrix(base_correction)

    return np.array(PoseMatrix)

# Add these imports at the top of utils.py
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

################ Training Management ####################

def get_training_config():
    """Get all training configurations from user"""
    config = {}
    
    print("\n====== Training Configuration ======")
    
    # Task selection
    while True:
        task = input("\nSelect task (hover/landing) [default: landing]: ").lower()
        if task in ['', 'hover', 'landing']:
            config['task'] = 'landing' if task == '' else task
            break
        print("Invalid choice. Please select 'hover' or 'landing'")
    
    # Rocket type selection
    while True:
        rocket = input("Select rocket type (falcon/starship) [default: starship]: ").lower()
        if rocket in ['', 'falcon', 'starship']:
            config['rocket_type'] = 'starship' if rocket == '' else rocket
            break
        print("Invalid choice. Please select 'falcon' or 'starship'")
    
    # Visualization preferences
    config['render'] = input("Enable environment rendering? (y/n) [default: n]: ").lower() == 'y'
    config['plot_realtime'] = input("Enable real-time plotting? (y/n) [default: y]: ").lower() != 'n'
    config['save_plots'] = input("Save training plots? (y/n) [default: y]: ").lower() != 'n'
    
    # Training parameters
    try:
        config['max_episodes'] = int(input("Enter maximum episodes [default: 1000]: ") or 1000)
        config['save_freq'] = int(input("Save checkpoint frequency (episodes) [default: 100]: ") or 100)
    except ValueError:
        print("Invalid input for episodes. Using defaults.")
        config['max_episodes'] = 1000
        config['save_freq'] = 100
    
    return config

def setup_directories(base_dir="PPO_preTrained", env_name="RocketLanding"):
    """Create necessary directories"""
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create environment directory
    env_dir = os.path.join(base_dir, env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
        
    # Create logs directory
    log_dir = os.path.join("PPO_logs", env_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    return env_dir, log_dir

def get_latest_checkpoint(directory, env_name):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(directory):
        return None, 0, 0
    
    files = [f for f in os.listdir(directory) if f.startswith(f"PPO_{env_name}")]
    if not files:
        return None, 0, 0
    
    runs = []
    for f in files:
        try:
            parts = f.split('_')
            seed, run = int(parts[-2]), int(parts[-1].split('.')[0])
            runs.append((seed, run, f))
        except:
            continue
    
    if not runs:
        return None, 0, 0
        
    latest = max(runs, key=lambda x: x[1])
    return os.path.join(directory, latest[2]), latest[0], latest[1]

def setup_training_state(directory, log_dir, env_name, ppo_agent):
    """Setup training state and handle checkpoint loading."""
    latest_checkpoint, checkpoint_seed, checkpoint_run = get_latest_checkpoint(directory, env_name)
    
    if latest_checkpoint is not None:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        response = input("Continue from previous checkpoint? (y/n) [default: n]: ").lower()
        
        if response == 'y':
            # Load model weights
            ppo_agent.load(latest_checkpoint)
            
            # Load training logs
            log_path = os.path.join(log_dir, f'PPO_{env_name}_log_{checkpoint_run}.csv')
            if os.path.exists(log_path):
                data = np.genfromtxt(log_path, delimiter=',', skip_header=1)
                i_episode = int(data[-1, 0])
                time_step = int(data[-1, 1])
                rewards = data[:, 2].tolist()
            else:
                i_episode, time_step, rewards = 0, 0, []
            
            # Setup logging
            log_f = open(log_path, 'a')
            
            print(f"Resuming training from episode {i_episode}, timestep {time_step}")
            return i_episode, time_step, rewards, checkpoint_run, log_f, latest_checkpoint
    
    # Start fresh training
    run_num = len(next(os.walk(log_dir))[2])
    log_path = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')
    log_f = open(log_path, 'w+')
    log_f.write('episode,timestep,reward\n')
    checkpoint_path = os.path.join(directory, f"PPO_{env_name}_0_{run_num}.pth")
    
    return 0, 0, [], run_num, log_f, checkpoint_path

def setup_plotting(config):
    """Setup plotting based on configuration"""
    if not config['plot_realtime'] and not config['save_plots']:
        return None, None
    
    plt.close('all')  # Close any existing plots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    
    if config['plot_realtime']:
        plt.ion()
        plt.show(block=False)
    
    return fig, ax

def update_plots(fig, ax, episode_rewards, window_size, config, save_dir=None):
    """Update and optionally save plots"""
    if fig is None or ax is None:
        return
        
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_std = np.array([np.std(episode_rewards[i-window_size+1:i+1]) 
                             for i in range(window_size-1, len(episode_rewards))])
        episodes = np.arange(window_size-1, len(episode_rewards))
        
        ax.clear()
        ax.plot(episodes, moving_avg, label='Moving Average')
        ax.fill_between(episodes, moving_avg-moving_std, moving_avg+moving_std, 
                       alpha=0.2, label='Standard Deviation')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        
        if config['plot_realtime']:
            plt.draw()
            plt.pause(0.01)
        
        if config['save_plots'] and save_dir:
            plt.savefig(os.path.join(save_dir, 'training_progress.png'))

################ Configuration Management ####################
def save_training_config(config, directory):
    """Save training configuration for reproducibility"""
    config_path = os.path.join(directory, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_training_config(directory):
    """Load previous training configuration"""
    config_path = os.path.join(directory, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def get_best_model_path(log_dir, env_name):
    """Find the best performing model based on logs"""
    best_reward = float('-inf')
    best_model = None
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    for log_file in log_files:
        data = pd.read_csv(os.path.join(log_dir, log_file))
        avg_reward = data['reward'].mean()
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = log_file.replace('log', 'model').replace('.csv', '.pth')
    
    return best_model if best_model else None

################ Performance Monitoring ####################
def track_training_stats():
    """Track various training statistics"""
    return {
        'best_reward': float('-inf'),
        'best_episode': 0,
        'running_avg': [],
        'episode_lengths': [],
        'success_rate': [],
        'crash_rate': []
    }

def update_training_stats(stats, reward, episode_length, success, crash):
    """Update training statistics"""
    stats['running_avg'].append(reward)
    stats['episode_lengths'].append(episode_length)
    stats['success_rate'].append(1 if success else 0)
    stats['crash_rate'].append(1 if crash else 0)
    
    if reward > stats['best_reward']:
        stats['best_reward'] = reward
        stats['best_episode'] = len(stats['running_avg'])
    
    return stats

################ Plot Management ####################
def setup_training_plots(plot_config):
    """Setup multiple plots for training visualization"""
    if not plot_config['enabled']:
        return None
        
    figs = {}
    figs['reward'] = plt.figure(figsize=(10, 5))
    figs['success_rate'] = plt.figure(figsize=(10, 5))
    figs['episode_length'] = plt.figure(figsize=(10, 5))
    
    return figs

def update_training_plots(figs, stats, save_dir=None):
    """Update all training plots"""
    if not figs:
        return
        
    # Update reward plot
    plt.figure(figs['reward'].number)
    plt.clf()
    plt.plot(stats['running_avg'])
    plt.title('Training Rewards')
    
    # Update success rate plot
    plt.figure(figs['success_rate'].number)
    plt.clf()
    window = 100
    success_rate = np.convolve(stats['success_rate'], 
                              np.ones(window)/window, 
                              mode='valid')
    plt.plot(success_rate)
    plt.title('Success Rate')
    
    if save_dir:
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f'{name}.png'))
            
################ Error Handling and Logging ####################
def setup_logger(log_dir, env_name):
    """Setup logging configuration"""
    import logging
    
    logger = logging.getLogger('rocket_training')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f'{env_name}_training.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

################ Training Resume Management ####################
def save_training_state(directory, episode, timestep, stats, model):
    """Save complete training state"""
    state = {
        'episode': episode,
        'timestep': timestep,
        'stats': stats,
        'model_state': model.state_dict()
    }
    torch.save(state, os.path.join(directory, 'training_state.pth'))

def load_training_state(directory):
    """Load complete training state"""
    state_path = os.path.join(directory, 'training_state.pth')
    if os.path.exists(state_path):
        return torch.load(state_path)
    return None

def load_existing_training(directory, log_dir, env_name, ppo_agent, checkpoint_path, checkpoint_seed, checkpoint_run):
    """Load existing training state"""
    try:
        # Load model weights
        ppo_agent.load(checkpoint_path)
        
        # Load training logs
        log_path = os.path.join(log_dir, f'PPO_{env_name}_log_{checkpoint_run}.csv')
        if os.path.exists(log_path):
            data = np.genfromtxt(log_path, delimiter=',', skip_header=1)
            episode = int(data[-1, 0])
            timestep = int(data[-1, 1])
            rewards = data[:, 2].tolist()
        else:
            episode, timestep, rewards = 0, 0, []
            
        # Setup logging
        log_f = open(log_path, 'a')
        
        return episode, timestep, rewards, checkpoint_run, log_f, checkpoint_path
    except Exception as e:
        print(f"Error loading existing training: {e}")
        return setup_new_training(directory, log_dir, env_name)

def setup_new_training(directory, log_dir, env_name):
    """Setup new training session"""
    # Get new run number
    run_num = len(next(os.walk(log_dir))[2])
    
    # Create new log file
    log_path = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')
    log_f = open(log_path, 'w+')
    log_f.write('episode,timestep,reward\n')
    
    # Create new checkpoint path
    checkpoint_path = os.path.join(directory, f"PPO_{env_name}_0_{run_num}.pth")
    
    return 0, 0, [], run_num, log_f, checkpoint_path

def setup_directories(base_dir="PPO_preTrained", env_name="RocketLanding"):
    """Create necessary directories"""
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create environment directory
    env_dir = os.path.join(base_dir, env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
        
    # Create logs directory
    log_dir = os.path.join("PPO_logs", env_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    return env_dir, log_dir
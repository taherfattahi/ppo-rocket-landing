import os
from datetime import datetime
import utils
import numpy as np

from PPO import PPO  # Assuming PPO is your policy class
from rocket import Rocket  # Import your Rocket environment class

import matplotlib.pyplot as plt

def get_latest_checkpoint(directory, env_name):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(directory):
        return None, 0, 0
    
    files = [f for f in os.listdir(directory) if f.startswith(f"PPO_{env_name}")]
    if not files:
        return None, 0, 0
    
    # Extract run numbers and find the latest
    runs = []
    for f in files:
        try:
            # Format: PPO_RocketLanding_0_13.pth
            parts = f.split('_')
            seed, run = int(parts[-2]), int(parts[-1].split('.')[0])
            runs.append((seed, run, f))
        except:
            continue
    
    if not runs:
        return None, 0, 0
        
    # Get the latest run
    latest = max(runs, key=lambda x: x[1])
    return os.path.join(directory, latest[2]), latest[0], latest[1]

def load_training_state(log_dir, env_name, run_num):
    """Load the previous training state from logs."""
    log_file = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')
    if not os.path.exists(log_file):
        return 0, 0, [], 0
    
    try:
        data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
        if len(data) == 0:
            return 0, 0, [], 0
        
        last_episode = int(data[-1, 0])
        last_timestep = int(data[-1, 1])
        rewards = data[:, 2].tolist()
        return last_episode, last_timestep, rewards, run_num
    except:
        return 0, 0, [], 0
    
    
################################### Training ###################################
def train():
    print("============================================================================================")

    # Get training configuration first
    config = utils.get_training_config()

    ####### initialize environment hyperparameters ######
    env_name = "RocketLanding"
    max_ep_len = 1000
    max_training_timesteps = int(6e6)
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    # Initialize the Rocket environment
    env = Rocket(max_steps=max_ep_len, task=config['task'], 
                rocket_type=config['rocket_type'])

    # Set state and action dimensions
    state_dim = env.state_dims
    action_dim = env.action_dims

    ################ PPO hyperparameters ################
    has_continuous_action_space = False
    update_timestep = max_ep_len * 4
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    # Setup directories
    directory, log_dir = utils.setup_directories()

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, 
                    K_epochs, eps_clip, has_continuous_action_space)

    # Setup training state (includes checkpoint loading if available)
    i_episode, time_step, episode_rewards, run_num, log_f, checkpoint_path = utils.setup_training_state(
        directory, log_dir, env_name, ppo_agent
    )

    # Track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # Initialize logging variables
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    window_size = 10

    # Setup plotting if enabled
    if config['plot_realtime'] or config['save_plots']:
        fig, ax = utils.setup_plotting(config)
    else:
        fig, ax = None, None

    # Training loop
    try:
        while time_step <= max_training_timesteps:
            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len + 1):
                # Select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)

                # Save reward and terminal state
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if config['render'] and i_episode % 50 == 0:
                    env.render()

                # Update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # Log to file
                if time_step % log_freq == 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_f.write('{},{},{}\n'.format(i_episode, time_step, round(log_avg_reward, 4)))
                    log_running_reward, log_running_episodes = 0, 0

                # Print average reward
                if time_step % print_freq == 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        i_episode, time_step, round(print_avg_reward, 2)))
                    print_running_reward, print_running_episodes = 0, 0

                # Save model weights
                if time_step % save_model_freq == 0:
                    ppo_agent.save(checkpoint_path)
                    print("Model saved at timestep: ", time_step)

                if done:
                    break

            # Update rewards and episodes
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1
            episode_rewards.append(current_ep_reward)

            # Update plot if enabled
            if fig is not None and len(episode_rewards) >= window_size:
                utils.update_plots(fig, ax, episode_rewards, window_size, config)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
    finally:
        if time_step > 0:
            ppo_agent.save(checkpoint_path)
            print("Final model saved at: ", checkpoint_path)
        if fig is not None:
            plt.close('all')
        log_f.close()
        print("Finished training at : ", datetime.now().replace(microsecond=0))

if __name__ == '__main__':
    train()
    
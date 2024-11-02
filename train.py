import os
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO  # Assuming PPO is your policy class
from rocket import Rocket  # Import your Rocket environment class

import matplotlib.pyplot as plt

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "RocketLanding"
    task = 'landing'  # 'hover' or 'landing'

    render = True

    has_continuous_action_space = False  # Discrete action space for Rocket

    max_ep_len = 1000                   # Max timesteps in one episode
    max_training_timesteps = int(6e6)   # Break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # Print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # Log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # Save model frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4    # Update policy every n timesteps
    K_epochs = 80                       # Update policy for K epochs in one PPO update
    eps_clip = 0.2                      # Clip parameter for PPO
    gamma = 0.99                        # Discount factor
    lr_actor = 0.0003                   # Learning rate for actor network
    lr_critic = 0.001                   # Learning rate for critic network
    random_seed = 0                     # Set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # Initialize the Rocket environment
    env = Rocket(max_steps=max_ep_len, task=task, rocket_type='starship')  # Adjust as needed for the hover task

    # Set state and action dimensions
    state_dim = env.state_dims
    action_dim = env.action_dims

    ###################### logging ######################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    # Initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space)

    # Track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # Initialize logging variables
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    episode_rewards = []
    window_size = 10  # Window size for moving average and standard deviation

    # Initialize the plot for real-time updating
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    plt.show(block=False)
    window_size = 10  # Window size for moving average and standard deviation

    # Training loop
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

            if render and i_episode % 50 == 0:
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
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, round(print_avg_reward, 2)))
                print_running_reward, print_running_episodes = 0, 0

            # Save model weights
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                print("Model saved at timestep: ", time_step)

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1
        
        episode_rewards.append(current_ep_reward)

        # Update the plot
        if len(episode_rewards) >= window_size:
            # Calculate moving average and standard deviation
            moving_avg = np.convolve(
                episode_rewards, np.ones(window_size)/window_size, mode='valid'
            )
            moving_std = np.array([
                np.std(episode_rewards[i-window_size+1:i+1]) 
                for i in range(window_size-1, len(episode_rewards))
            ])
            episodes = np.arange(window_size-1, len(episode_rewards))

            # Clear the axis and redraw
            ax.clear()
            ax.plot(episodes, moving_avg, label='Moving Average Reward')

            # Shade the area between (mean - std) and (mean + std)
            lower_bound = moving_avg - moving_std
            upper_bound = moving_avg + moving_std
            ax.fill_between(episodes, lower_bound, upper_bound, color='blue', alpha=0.2, label='Standard Deviation')

            # Set labels and title
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Progress with Variability Shading')
            ax.legend()
            plt.draw()
            plt.pause(0.01)
        else:
            # For initial episodes where we don't have enough data for moving average
            ax.clear()
            ax.plot(range(len(episode_rewards)), episode_rewards, label='Episode Reward')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Progress')
            ax.legend()
            plt.draw()
            plt.pause(0.01)

    log_f.close()
    print("Finished training at : ", datetime.now().replace(microsecond=0))

if __name__ == '__main__':
    train()
    
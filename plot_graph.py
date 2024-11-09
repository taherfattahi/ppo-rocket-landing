import os
import pandas as pd
import matplotlib.pyplot as plt

def save_graph():
    print("============================================================================================")
    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
    env_name = 'RocketLanding'

    fig_num = 0     #### change this to prevent overwriting figures in same env_name folder
    plot_avg = True    # plot average of all runs; else plot all runs separately
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

    # Setup directories
    figures_dir = os.path.join("PPO_figs", env_name)
    os.makedirs(figures_dir, exist_ok=True)
    fig_save_path = os.path.join(figures_dir, f'PPO_{env_name}_fig_{fig_num}.png')
    log_dir = os.path.join("PPO_logs", env_name)

    # Get log files
    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)
    all_runs = []

    # Load and process data
    for run_num in range(num_runs):
        log_f_name = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')
        print("Loading data from:", log_f_name)
        
        try:
            # Read CSV with specific column names
            data = pd.read_csv(log_f_name, names=['episode', 'timestep', 'reward'])
            print("Data shape:", data.shape)
            all_runs.append(data)
            print("-" * 90)
            
        except Exception as e:
            print(f"Error loading {log_f_name}: {str(e)}")
            continue

    if not all_runs:
        print("No valid data files found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if plot_avg:
        # Average all runs
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        # Smooth out rewards
        data_avg['reward_smooth'] = data_avg['reward'].rolling(
            window=window_len_smooth, 
            win_type='triang', 
            min_periods=min_window_len_smooth
        ).mean()
        
        data_avg['reward_var'] = data_avg['reward'].rolling(
            window=window_len_var, 
            win_type='triang', 
            min_periods=min_window_len_var
        ).mean()

        # Plot
        data_avg.plot(kind='line', x='timestep', y='reward_smooth',
                     ax=ax, color=colors[0],
                     linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep', y='reward_var',
                     ax=ax, color=colors[0],
                     linewidth=linewidth_var, alpha=alpha_var)

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], [f"reward_avg_{len(all_runs)}_runs"], loc=2)

    else:
        for i, run in enumerate(all_runs):
            run[f'reward_smooth_{i}'] = run['reward'].rolling(
                window=window_len_smooth,
                win_type='triang',
                min_periods=min_window_len_smooth
            ).mean()
            
            run[f'reward_var_{i}'] = run['reward'].rolling(
                window=window_len_var,
                win_type='triang',
                min_periods=min_window_len_var
            ).mean()

            run.plot(kind='line', x='timestep', y=f'reward_smooth_{i}',
                    ax=ax, color=colors[i % len(colors)],
                    linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep', y=f'reward_var_{i}',
                    ax=ax, color=colors[i % len(colors)],
                    linewidth=linewidth_var, alpha=alpha_var)

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [handles[i] for i in range(0, len(handles), 2)]
        new_labels = [labels[i] for i in range(0, len(labels), 2)]
        ax.legend(new_handles, new_labels, loc=2)

    # Finalize plot
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)
    plt.title(env_name, fontsize=14)

    # Save and show
    print("============================================================================================")
    plt.savefig(fig_save_path)
    print("Figure saved at:", fig_save_path)
    print("============================================================================================")
    plt.show()

if __name__ == '__main__':
    save_graph()
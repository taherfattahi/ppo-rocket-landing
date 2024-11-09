import os
import time
import numpy as np
import cv2
import torch
from PPO import PPO
from rocket import Rocket

def create_confetti_particles(num_particles=30):  # Reduced particles
    """Create confetti particles"""
    particles = []
    for _ in range(num_particles):
        particles.append({
            'x': np.random.randint(0, 800),
            'y': np.random.randint(-50, 0),
            'color': (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ),
            'size': np.random.randint(5, 15),
            'speed': np.random.randint(5, 15),
            'angle': np.random.uniform(-np.pi/4, np.pi/4)
        })
    return particles

def celebrate_landing(window_name="Perfect Landing!", duration=1.0):  # Reduced duration
    """Show celebration animation"""
    width, height = 800, 600
    particles = create_confetti_particles()
    
    start_time = time.time()
    while time.time() - start_time < duration:
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Update and draw particles in one pass
        for p in particles:
            p['y'] += p['speed']
            p['x'] += np.sin(p['angle']) * 2
            p['speed'] += 0.5
            p['angle'] += np.random.uniform(-0.1, 0.1)
            
            cv2.circle(frame, 
                      (int(p['x']), int(p['y'])), 
                      p['size'], 
                      p['color'], 
                      -1)
        
        # Add celebration text
        text = "Perfect Landing!"
        cv2.putText(frame, text, 
                   (width//4, height//2), 
                   cv2.FONT_HERSHEY_DUPLEX, 
                   2.0, (0, 0, 0), 2)
        
        cv2.imshow(window_name, frame)
        if cv2.waitKey(16) & 0xFF == 27:  # ~60 FPS
            break
    
    cv2.destroyWindow(window_name)

def get_test_config():
    """Get test-specific configuration"""
    config = {}
    
    print("\n====== Test Configuration ======")
    
    # Task selection
    task = input("\nSelect task (hover/landing) [default: landing]: ").lower()
    config['task'] = 'landing' if task in ['', 'landing'] else 'hover'
    
    # Rocket type selection
    rocket = input("Select rocket type (falcon/starship) [default: starship]: ").lower()
    config['rocket_type'] = 'starship' if rocket in ['', 'starship'] else 'falcon'
    
    # Rendering preference
    config['render'] = input("Enable rendering? (y/n) [default: y]: ").lower() != 'n'
    config['frame_delay'] = int(input("Frame delay in milliseconds [default: 16]: ") or 16)
    
    return config

def test():
    print("============================================================================================")

    # Get test configuration
    config = get_test_config()
    
    ################## Hyperparameters ##################
    env_name = "RocketLanding"
    max_ep_len = 1000
    total_test_episodes = 10
    
    # PPO hyperparameters
    has_continuous_action_space = False
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    
    # Initialize environment
    env = Rocket(max_steps=max_ep_len, 
                task=config['task'], 
                rocket_type=config['rocket_type'])
    
    # Set dimensions
    state_dim = env.state_dims
    action_dim = env.action_dims
    
    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, 
                    K_epochs, eps_clip, has_continuous_action_space)
    
    # Load pretrained model
    checkpoint_path = os.path.join("PPO_preTrained", env_name, "PPO_RocketLanding_0_0.pth")
    if not os.path.exists(checkpoint_path):
        print(f"\nError: No checkpoint found at {checkpoint_path}")
        print("Please ensure you have trained the model first.")
        return
        
    print(f"\nLoading model from: {checkpoint_path}")
    try:
        ppo_agent.load(checkpoint_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nStarting testing...")
    test_running_reward = 0
    successful_landings = 0
    
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        
        for t in range(1, max_ep_len + 1):
            # Select action
            with torch.no_grad():  # Faster inference
                action = ppo_agent.select_action(state)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            
            # Render if enabled
            if config['render'] and t % 2 == 0:  # Skip frames for speed
                env.render(window_name="Rocket Test", 
                         wait_time=config['frame_delay'])
            
            if done:
                # Check landing conditions
                x_pos, y_pos = next_state[0], next_state[1]
                vx, vy = next_state[2], next_state[3]
                theta = next_state[4]
                
                # Stricter landing conditions
                if (reward > 500 and  # High reward
                    abs(x_pos) < 20.0 and  # Close to center
                    abs(vx) < 5.0 and abs(vy) < 5.0 and  # Low velocity
                    abs(theta) < 0.1):  # Nearly vertical
                    
                    successful_landings += 1
                    print(f"\nPerfect landing! Reward: {reward:.2f}")
                    celebrate_landing()
                break
                
            state = next_state
        
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print(f'Episode: {ep} \t\t Reward: {round(ep_reward, 2)}')
    
    env.close()
    cv2.destroyAllWindows()
    
    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    success_rate = (successful_landings / total_test_episodes) * 100
    print(f"Average test reward: {round(avg_test_reward, 2)}")
    print(f"Successful landings: {successful_landings}/{total_test_episodes} ({success_rate:.1f}%)")
    print("============================================================================================")

if __name__ == '__main__':
    test()
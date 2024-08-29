import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

from maze_env import MazeEnv

# Hyperparameters
learning_rate = 0.2
discount_factor = 0.95
epsilon = 0.9
epsilon_decay=0.995
episodes = 10000

# Initialize the maze environment
env = MazeEnv()
state_size = 13  # position (2) + goal (2) + grid (3x3)
action_size = 4

# Initialize the Q-table
q_table = np.zeros((env.size, env.size, action_size))

# Function to preprocess the state
def preprocess_state(state):
    position, goal, grid = state
    flat_grid = np.array(grid).flatten()
    return np.concatenate([position, goal, flat_grid])

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(action_size)
    else:
        return np.argmax(q_table[state[0], state[1]])

# Variables to store the reward and steps
rewards = []
steps = []

# Variable to store the first path to the green goal
first_path = None

# Training the agent
for episode in range(episodes):
    state = preprocess_state(env.reset())
    done = False
    total_reward = 0
    path = []

    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)

        # Update Q-table using Q-learning update rule
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])

        new_value =  old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state[0], state[1], action] = new_value

        state = next_state
        total_reward += reward
        path.append(state)
        epsilon = max(0.1, epsilon * epsilon_decay)
        # Render the environment for visualization
        if episode%100==0:
            env.render()
        # Check if the agent reaches the goal
        if reward == 200 and first_path is None:
            first_path = path
    
    rewards.append(total_reward)
    steps.append(episode + 1)

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Save the Q-table
np.save("q_table.npy", q_table)
print("Q-table saved to q_table.npy.")

# Save the Q-table to a text file
np.savetxt("q_table.txt", q_table.reshape(-1, action_size), fmt='%f')
print("Q-table saved to q_table.txt.")

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(steps, rewards, label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Q-learning: Episodes vs. Total Reward')
plt.legend()
plt.grid(True)
plt.savefig('training_rewards.png')
plt.show()





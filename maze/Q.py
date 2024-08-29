import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# Maze Environment
class Maze:
    def __init__(self, size=40, obstacles=[], cliffs=[]):
        self.size = size
        self.obstacles = obstacles
        self.cliffs = cliffs
        self.max_steps = 500
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size))
        self.start = (0, 0)
        self.end = self.random_end_position()
        self.steps = 0
        self.state = self.start
        self.done = False
        self.path = [self.start]  # Initialize path
        self.visited = np.zeros((self.size, self.size))  # Track visited states
        self.visited_count = np.zeros((self.size, self.size))  # Track visit counts
        self.visited[self.start] = 1  # Mark the start as visited
        self.add_obstacles()
        self.add_cliffs()
        return self.state

    def random_end_position(self):
        return (random.randint(30, 39), random.randint(30, 39))

    def add_obstacles(self):
        for x, y in self.obstacles:
            self.maze[x, y] = 1  # Gray

    def add_cliffs(self):
        for x, y in self.cliffs:
            self.maze[x, y] = -1  # Red

    def get_state(self):
        state = np.copy(self.maze)
        state[self.state] = 3  # Current position
        state[self.end] = 2  # End position
        return state

    def step(self, action):
        x, y = self.state
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1

        if x < 0 or x >= 40 or y < 0 or y >= 40 or self.maze[x, y] == 1:
            next_state = self.state  # 碰壁
        else:
            next_state = (x, y)

        self.state = next_state
        self.steps += 1

        if next_state == self.end:
            reward = 200
            self.done = True
        elif x < 0 or x >= 40 or y < 0 or y >= 40 or self.maze[x, y] == 1:
            reward = -5
            self.done = False
        elif self.maze[next_state[0], next_state[1]] == -1:
            reward = -100
            self.done = True
        elif self.steps >= self.max_steps:
            reward = -10
            self.done = True
        else:
            reward = -1
            self.done = False

        if self.steps % 20 == 0:
            self.end = self.random_end_position()

        return next_state, reward, self.done


    def render_path(self, path):
        for (x, y) in path:
            self.maze[x, y] = 4  
        cmap = mcolors.ListedColormap(['red', 'white', 'gray', 'orange', 'yellow', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        state = self.get_state()
        plt.imshow(state, cmap=cmap, norm=norm)
        plt.show()


# Q-Learning Agent with improved strategies
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma, epsilon=1.0, epsilon_decay=0.8):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train_with_replay(self, env, episodes, replay_size=32, test_interval=1):
        rewards = []
        test_rewards = []
        replay_memory = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                replay_memory.append((state, action, reward, next_state, done))

                if len(replay_memory) > replay_size:
                    batch = random.sample(replay_memory, replay_size)
                    for s, a, r, ns, d in batch:
                        self.update(s, a, r, ns)

                state = next_state
                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)
            if (episode + 1) % test_interval == 0:
                test_reward = self.test(env)
                test_rewards.append(test_reward)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        return rewards, test_rewards
    def test(self, env):
        state = env.reset()
        total_reward = 0
        path = [env.start]
        while True:
            action = np.argmax(self.q_table[state])
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            path.append(env.state)
            if done:
                break
        env.render_path(path)
        return total_reward

def analyze_performance(rewards, window=50):
    mean_rewards = [np.mean(rewards[i - window:i]) for i in range(window, len(rewards))]
    plt.plot(mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Performance Analysis')
    plt.show()

def analyze_test_performance(test_rewards, test_interval):
    plt.plot([i * test_interval for i in range(1, len(test_rewards) + 1)], test_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Test Reward')
    plt.title('Test Performance Analysis')
    plt.show()

if __name__ == "__main__":
    obstacles = [(1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (10, 4), (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10), (17, 11), (18, 12), (19, 13), (20, 14), (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (28, 4), (29, 4), (30, 4), (31, 4), (32, 4), (33, 4), (34, 4), (35, 4), (36, 4), (25, 22), (25, 23), (25, 24), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37), (25, 38), (25, 39)]  # Define your obstacles here
    cliffs = [(2, 0), (2, 1), (22, 2), (22, 3), (22, 4), (22, 5), (22, 6), (22, 7), (22, 8), (24, 15), (24, 16), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21), (36, 26), (37, 26), (38, 26), (39, 26)]  # Define your cliffs here
    env = Maze(obstacles=obstacles, cliffs=cliffs)
    state_size = env.size * env.size
    action_size = 4
   
    agent = QLearningAgent(state_size, action_size, lr=0.2, gamma=0.99, epsilon=1.0, epsilon_decay=0.995) # Adjusted learning parameters
    episodes = 10000
    test_interval = 10000
    rewards, test_rewards = agent.train_with_replay(env, episodes, test_interval=test_interval)
    analyze_performance(rewards)
    analyze_test_performance(test_rewards, test_interval)

    
    agent.test(env)
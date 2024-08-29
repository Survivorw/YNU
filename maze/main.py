import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
from collections import defaultdict
import queue

class Environment:
    def __init__(self):
        self.size = 40
        self.maze = self._create_maze()
        self.start = (0, 0)
        self.end_area = [(i, j) for i in range(30, 40) for j in range(30, 40)]
        self.state = self.start
        self.steps = 0
        self.max_steps = 500
        self.end = self._get_random_end()
        self.done = False

        self.qstate=[]
        self.goal_area = [(x, y) for x in range(30, 40) for y in range(30, 40)]
        self.obstacles = (
            [(1, j) for j in range(6, 25)]
            + [(25, j) for j in range(22, 40)]
            + [(i, 4) for i in range(23, 36)]
            + [
                (10, 3),
                (11, 4),
                (12, 5),
                (13, 6),
                (14, 7),
                (15, 8),
                (16, 9),
                (17, 10),
                (18, 11),
                (19, 12),
                (20, 13),
            ]
        )
        self.cliffs = (
            [(2, j) for j in range(0, 2)]
            + [(22, j) for j in range(2, 8)]
            + [(24, j) for j in range(15, 22)]
            + [(i, 26) for i in range(36, 40)]
        )
        pygame.init()
        self.cell_size = 17
        self.screen = pygame.display.set_mode(
            (self.size * self.cell_size, self.size * self.cell_size)
        )
        pygame.display.set_caption("Maze Environment")
        self.clock = pygame.time.Clock()

    def _create_maze(self):
        maze = np.zeros((self.size, self.size), dtype=int)
        # 设置障碍物区域为-1
        maze[1, 6:26] = -1
        obstacle_area_2 = [
            (10, 3),
            (11, 4),
            (12, 5),
            (13, 6),
            (14, 7),
            (15, 8),
            (16, 9),
            (17, 10),
            (18, 11),
            (19, 12),
            (20, 13),
        ]
        for i, j in obstacle_area_2:
            maze[i, j] = -1
        maze[23:37, 4] = -1
        maze[25, 22:40] = -1
        maze[2, 0:2] = -1
        maze[22, 2:9] = -1
        maze[24, 15:22] = -1
        maze[36:40, 26] = -1
        return maze

    def _get_random_end(self):
        return random.choice(self.end_area)

    def reset(self):
        self.state = self.start
        self.steps = 0
        self.end = self._get_random_end()
        self.done = False
        return self.state

    def step(self, action):
        i, j = self.state
        if action == 0:  # 上
            i -= 1
        elif action == 1:  # 下
            i += 1
        elif action == 2:  # 左
            j -= 1
        elif action == 3:  # 右
            j += 1

        if i < 0 or i >= 40 or j < 0 or j >= 40:
            next_state = self.state  # 碰壁
        else:
            next_state = (i, j)
            
        self.f=0
        self.state = next_state
        self.steps += 1
        old_distance = np.linalg.norm(np.array(self.state) - np.array(self.end))
        new_distance = np.linalg.norm(np.array(next_state) - np.array(self.end))

        if next_state == self.end:
            self.f=1
            reward = 1000
            self.done = True

        elif i < 0 or i >= 40 or j < 0 or j >= 40:
            reward = -10
            self.done = False

        elif self.maze[next_state[0], next_state[1]] == -1:
            reward = -1000
            self.done = True

        elif self.steps >= self.max_steps:
            reward = -1
            self.done = True
        
        
        elif self.state in self.goal_area:
            if new_distance <= old_distance:
                reward = 10
            
            else:
                reward = -10
            if len(self.qstate) < 20:
                self.qstate.append(self.state)
            elif self.state in self.qstate:
                    reward = -10
                    self.qstate.pop(0)
                    self.qstate.append(self.state)
            else:
                self.qstate.pop(0)
                self.qstate.append(self.state)
        else:    
            reward = -1
            self.done = False
        
       
        # elif self.state in self.goal_area:
        #     if new_distance <= old_distance:
        #         reward = new_distance*1
        #     else:
        #         reward = -new_distance*1
        # elif new_distance > old_distance:
        #     if self.state not in self.goal_area:
        #         None
        #     else :reward = -1
            
        

        if self.steps % 20 == 0:
            self.end = self._get_random_end()

        return next_state, reward, self.done,self.f

    def render(self, coordinates=None):
        self.screen.fill((255, 255, 255))

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if (x, y) in self.obstacles:
                    pygame.draw.rect(self.screen, (128, 128, 128), rect)
                elif (x, y) in self.cliffs:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                elif (x, y) in self.goal_area:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        start_rect = pygame.Rect(
            self.start[1] * self.cell_size,
            self.start[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        goal_rect = pygame.Rect(
            self.end[1] * self.cell_size,
            self.end[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, (255, 255, 0), start_rect)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)

        agent_rect = pygame.Rect(
            self.state[1] * self.cell_size,
            self.state[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.ellipse(self.screen, (0, 0, 255), agent_rect)

        pygame.display.flip()
        self.clock.tick(60)


class Agent:
    def __init__(self, environment, epsilon, alpha, gamma, epsilon_decay,alpha_decay):
        self.environment = environment
        self.q_table = np.zeros((environment.size, environment.size, 4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.alpha_decay=alpha_decay

    def train(self, EPISODES):
        reward_list = []
        f_list=[]
        for episode in range(EPISODES):
            state = self.environment.reset()
            done = False
            eps_reward = 0
            
            while not done:
                if np.random.random() < self.epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(self.q_table[state[0], state[1]])
                next_state, reward, done ,f= self.environment.step(action)
                # 更新Q表
                Q = self.q_table[state[0], state[1], action]
                maxQ = np.max(self.q_table[next_state[0], next_state[1]])
                self.q_table[state[0], state[1], action] = Q + self.alpha * (
                    reward + self.gamma * maxQ - Q
                )
                # 统计奖励
                eps_reward += reward
                state = next_state
                
                # if episode % 1000 == 0:
                #     env.render()
            self.alpha = max (0.3,self.alpha * self.alpha_decay)
            self.epsilon = max(0.005, self.epsilon * self.epsilon_decay)
            f_list.append(f)
            reward_list.append(eps_reward)
            print(f"Episode:{episode + 1}/{EPISODES},Reward: {eps_reward}")
        return reward_list,f_list


EPISODES = 1000
alpha = 0.6
gamma = 0.99
alpha_decay=0.995
epsilon = 0.005

epsilon_decay = 0

env = Environment()

agent = Agent(env, epsilon, alpha, gamma, epsilon_decay,alpha_decay)

reward_list ,f_list= agent.train(EPISODES)

plt.plot(reward_list, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("QLearningAgent")
plt.plot(
    [
        (
            np.mean(reward_list[i - 50 : i + 1])
            if i >= 50
            else np.mean(reward_list[: i + 1])
        )
        for i in range(len(reward_list))
    ],
    "b-",
    label="avg_reward",
)
plt.legend()
plt.show(block=True)

plt.plot(f_list, label="F")
plt.xlabel("Episode")
plt.ylabel("F")

plt.legend()
plt.show(block=True)

states = list()
state = env.reset()
eps_reward = 0
while not env.done:
    action = np.argmax(agent.q_table[state[0], state[1]])
    next_state, reward, done,f = env.step(action)
    states.append(next_state)
    state = next_state

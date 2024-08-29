import gym
import numpy as np
import pygame

class MazeEnv(gym.Env):
    def __init__(self):
        self.size = 40
        self.state = np.zeros((self.size, self.size))
        self.start = (0, 0)
        self.goal_area = [(x, y) for x in range(30, 40) for y in range(30, 40)]
        self.obstacles = [(1, j) for j in range(6, 25)] + \
                         [(25, j) for j in range(22, 40)] + \
                         [(i, 4) for i in range(23, 36)] + \
                         [(10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13)]
        self.cliffs = [(2, j) for j in range(0, 2)] + \
                      [(22, j) for j in range(2, 8)] + \
                      [(24, j) for j in range(15, 22)] + \
                      [(i, 26) for i in range(36, 40)]
        self.reset()
        pygame.init()
        self.cell_size = 17
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("Maze Environment")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.position = self.start
        self.goal = self.goal_area[np.random.randint(0, len(self.goal_area))]
        self.steps_taken = 0
        self.total_reward = 0
        self.visited_states = set()
        return self.get_state()

    def step(self, action):
        new_position = self.position
        if action == 0:  # up
            new_position = (self.position[0], self.position[1] - 1)
        elif action == 1:  # down
            new_position = (self.position[0], self.position[1] + 1)
        elif action == 2:  # left
            new_position = (self.position[0] - 1, self.position[1])
        elif action == 3:  # right
            new_position = (self.position[0] + 1, self.position[1])

        reward = 0
        done = False
        old_distance = np.linalg.norm(np.array(self.position) - np.array(self.goal))
        
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size and new_position not in self.obstacles:
            if new_position in self.cliffs:
                reward = -200
                done = True
                self.position = self.start
                
            elif new_position == self.goal:
                reward = 400
                done = True
                self.position = self.start
                
            elif new_position == self.goal_area:
                reward = 100
                done = False
            else:
                reward = -1
                new_distance = np.linalg.norm(np.array(new_position) - np.array(self.goal))
                
                if new_distance < old_distance:
                    reward += 2
                else:
                    reward -= 1
                    
                if new_position not in self.visited_states:
                    reward += 1  # 访问新状态奖励
                    self.visited_states.add(new_position)
                
                self.position = new_position
                self.steps_taken += 1
                self.total_reward += reward

                if self.steps_taken % 20 == 0:
                    self.goal = self.goal_area[np.random.randint(0, len(self.goal_area))]

                if self.steps_taken >= 500:
                    reward -= 1
                    self.position = self.start
                    self.steps_taken = 0
                    done = True
                

        else:
            reward = -1
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self):
        x, y = self.position
        state = []
        for i in range(x - 1, x + 2):
            row = []
            for j in range(y - 1, y + 2):
                if 0 <= i < self.size and 0 <= j < self.size:
                    if (i, j) in self.obstacles:
                        row.append(-1)
                    elif (i, j) in self.cliffs:
                        row.append(-500)
                    elif (i, j) in self.goal_area:
                        row.append(100)
                    else:
                        row.append(0)
                else:
                    row.append(-1)
            state.append(row)
        return (self.position, self.goal, state)

    def render(self):
        self.screen.fill((255, 255, 255))

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if (x, y) in self.obstacles:
                    pygame.draw.rect(self.screen, (128, 128, 128), rect)
                elif (x, y) in self.cliffs:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                elif (x, y) in self.goal_area:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        start_rect = pygame.Rect(self.start[1] * self.cell_size, self.start[0] * self.cell_size, self.cell_size, self.cell_size)
        goal_rect = pygame.Rect(self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 255, 0), start_rect)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)

        agent_rect = pygame.Rect(self.position[1] * self.cell_size, self.position[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.ellipse(self.screen, (0, 0, 255), agent_rect)

        pygame.display.flip()
        self.clock.tick(60)  # 控制帧率

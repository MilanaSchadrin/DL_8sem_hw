import os
import sys
import time
import random
import numpy as np
from collections import deque
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import skimage
from skimage import color, transform, exposure

sys.path.append('game/')
import game.wrapped_flappy_bird as game
sys.path.pop()
wandb.init(project="flappy-bird-dqn") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('params.yaml') as f:
    params = yaml.safe_load(f)

ACTIONS = 2  # 0 = Do Nothing, 1 = Flap
BATCH_SIZE = params['training']['batch_size']
LR = params['training']['lr']
GAMMA = params['training']['gamma']
EPSILON = params['training']['epsilon']
EPSILON_DECAY = params['training']['epsilon_decay']
FINAL_EPSILON = params['training']['final_epsilon']
OBSERVE = 1000
EXPLORE = 1000000  
TARGET_REPLACE_ITER = 1000
MEMORY_SIZE = params['training']['memory_size']
SAVE_PATH = 'flappy_model.pth'

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(2304 , 512)
        self.fc2 = nn.Linear(512 , 256)
        self.fc3 = nn.Linear(256 , 128)
        self.out = nn.Linear(128, ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DQN:
    def __init__(self):
        self.eval_net = NET().to(device)
        self.target_net = NET().to(device)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.best_score = -float('inf')

    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state).to(device)
        if np.random.rand() > epsilon:
            with torch.no_grad():
                q_values = self.eval_net(state)
            action = torch.argmax(q_values).item()
        else:
            action = 0 if random.random() < 0.5 else 1
        action_vec = np.zeros(ACTIONS)
        action_vec[action] = 1
        return action_vec

    def store_transition(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s_, d = zip(*batch)

        s = torch.FloatTensor(np.concatenate(s)).to(device)
        s_ = torch.FloatTensor(np.concatenate(s_)).to(device)
        a = torch.LongTensor(np.array(a).argmax(axis=1)).unsqueeze(1).to(device)
        r = torch.FloatTensor(r).unsqueeze(1).to(device)
        d = torch.BoolTensor(d).unsqueeze(1).to(device)

        q_eval = self.eval_net(s).gather(1, a)
        with torch.no_grad():
            next_actions = self.eval_net(s_).argmax(1).unsqueeze(1)  # shape: [B, 1]
            q_next = self.target_net(s_).gather(1, next_actions)
        q_target = r + GAMMA * q_next * (~d)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        wandb.log({"loss": loss.item(), "learn_step": self.learn_step_counter})

    def save_model(self):
         torch.save({
        'eval': self.eval_net.state_dict(),
        'target': self.target_net.state_dict(),
    }, SAVE_PATH)

    def load_model(self):
        if os.path.exists(SAVE_PATH):
            checkpoint = torch.load(SAVE_PATH, map_location=device)
            self.eval_net.load_state_dict(checkpoint['eval'])
            self.target_net.load_state_dict(checkpoint['target'])
        else:
            print("No model found.")


def preprocess(frame):
    frame = color.rgb2gray(frame)
    frame = transform.resize(frame, (80, 80))
    frame = frame/255.0
    return frame


def get_initial_state(game_state):
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, _, terminal= game_state.frame_step(do_nothing)
    x_t = preprocess(x_t)
    s_t = np.stack([x_t] * 4, axis=0)
    return s_t.reshape(1, 4, 80, 80), terminal

def train(episode):
    agent = DQN()
    game_state = game.GameState()
    epsilon = EPSILON
    step = 0
    agent.load_model()
    rewards = [] 

    for episode in range(episode):
        s_t, _ = get_initial_state(game_state)
        done = False
        episode_reward = 0

        while not done:
            a_t = agent.choose_action(s_t, epsilon)
            x_t1_colored, reward, terminal= game_state.frame_step(a_t)
            r_t=reward
            x_t1 = preprocess(x_t1_colored).reshape(1, 1, 80, 80)
            s_t1 = np.append(s_t[:, 1:, :, :], x_t1, axis=1)

            agent.store_transition(s_t, a_t, r_t, s_t1, terminal)

            if step > OBSERVE:
                agent.learn()

            s_t = s_t1
            done = terminal
            episode_reward += r_t
            step += 1

            if epsilon > FINAL_EPSILON and step < EXPLORE:
                epsilon *= EPSILON_DECAY

            if step % 100 == 0:
                if episode_reward > agent.best_score:
                    agent.save_model()
                    agent.best_score=episode_reward
                    print(f"Saved model at step {step}")

        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "epsilon": epsilon,
            "step": step
        })
        rewards.append(episode_reward)
        if len(rewards) >= 20:
            avg_reward = np.mean(rewards[-20:])
            if avg_reward > agent.best_score:
                agent.best_score = avg_reward
                agent.save_model()
                print(f"Saved model at step {step} with avg reward {avg_reward:.2f}")
        print(f"Episode {episode} | Total reward: {episode_reward} | Epsilon: {epsilon:.5f}")

    agent.save_model()
    wandb.save(SAVE_PATH)
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid()
    plt.savefig("training_rewards.png")
    plt.show()


def test():
    agent = DQN()
    game_state = game.GameState()
    agent.load_model()

    s_t, _ = get_initial_state(game_state)
    done = False
    total_reward = 0

    while True:
        a_t = agent.choose_action(s_t, epsilon=0.01)
        x_t1_colored, _, terminal = game_state.frame_step(a_t)
        r_t = 1 if not terminal else -10
        x_t1 = preprocess(x_t1_colored).reshape(1, 1, 80, 80)
        s_t1 = np.append(s_t[:, 1:, :, :], x_t1, axis=1)
        s_t = s_t1
        total_reward += r_t

        if terminal:
            total_reward=int(total_reward//10)
            print(f"Game Over | Score: {total_reward}")
            break



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True, help="train or test")
    args = parser.parse_args()

    if args.mode == 'train':
        train(50000)
    else:
        test()

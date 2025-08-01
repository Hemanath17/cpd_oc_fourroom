import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import io
from gym import spaces

class FourRoomsEnv:
    def __init__(self, fail_prob=0.333, reward_goal=1.0, reward_default=0.0,
                 max_steps=500, seed=0):
        self.fail_prob = fail_prob
        self.reward_goal = reward_goal
        self.reward_default = reward_default
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.seed_val = seed  # used for gif filename
        self.action_space = spaces.Discrete(4) # for cpd
        self.episode_counter = 0 # for cpd


        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([
            [1 if c == 'w' else 0 for c in line]
            for line in layout.splitlines()
        ])
        self.height, self.width = self.occupancy.shape
        self.free_positions = [(r, c) for r in range(self.height)
                               for c in range(self.width)
                               if self.occupancy[r, c] == 0]

        self.state_to_pos = {i: pos for i, pos in enumerate(self.free_positions)}
        self.pos_to_state = {pos: i for i, pos in self.state_to_pos.items()}

        self.nS = len(self.free_positions)
        self.nA = 4  # up, down, left, right

        self.episode_count = 0
        self.goal_switch_episode = 1000
        self.east_doorway_rc = (1, 7)
        self._post_switch_goal_state = self._to_state(4, 10)  # Fixed post-switch goal
        self._goal_switch_printed = False

        self.record_gif_for_episodes = {1000, 1500}
        self.trajectory = []

    def _to_state(self, r, c):
        return self.pos_to_state[(r, c)]

    def _from_state(self, s):
        return self.state_to_pos[s]

    def reset(self):
        self.episode_count += 1

        if self.episode_count in self.record_gif_for_episodes:
            self.trajectory = []

        # Save previous trajectory
        if (self.episode_count - 1) in self.record_gif_for_episodes and self.trajectory:
            self._save_gif(self.trajectory, self.goal_state, self.episode_count - 1)

        if self.episode_count <= self.goal_switch_episode:
            r, c = self.east_doorway_rc
            self.goal_state = self._to_state(r, c)
        else:
            self.goal_state = self._post_switch_goal_state
            if not self._goal_switch_printed:
                print(f"[ENV] Post-switch goal fixed at (4, 10) (state={self.goal_state})")
                self._goal_switch_printed = True

        valid_states = [self._to_state(r, c) for (r, c) in self.free_positions
                        if self._to_state(r, c) != self.goal_state]
        self.state = self.rng.choice(valid_states)
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        r, c = self._from_state(self.state)
        r2, c2 = self._move_deterministic(r, c, action)

        if self.occupancy[r2, c2] == 0:
            if self.rng.rand() > self.fail_prob:
                self.state = self._to_state(r2, c2)

        reward = self.reward_goal if self.state == self.goal_state else self.reward_default
        done = (self.state == self.goal_state) or (self.steps >= self.max_steps)

        if self.episode_count in self.record_gif_for_episodes:
            self.trajectory.append(self.state)

        return self.state, reward, done, {}

    def _move_deterministic(self, r, c, a):
        if a == 0:   # up
            r2, c2 = max(0, r - 1), c
        elif a == 1: # down
            r2, c2 = min(self.height - 1, r + 1), c
        elif a == 2: # left
            r2, c2 = r, max(0, c - 1)
        elif a == 3: # right
            r2, c2 = r, min(self.width - 1, c + 1)
        return r2, c2

    def _save_gif(self, states, goal_state, episode, out_dir="results/gifs"):
        os.makedirs(out_dir, exist_ok=True)
        frames = [self._draw_frame(s, goal_state) for s in states]
        path = os.path.join(out_dir, f"seed_{self.seed_val}_ep{episode}.gif")
        imageio.mimsave(path, frames, fps=15)
        print(f"[GIF] Saved {path}")

    def _draw_frame(self, agent_state, goal_state):
        r_agent, c_agent = self._from_state(agent_state)
        r_goal, c_goal = self._from_state(goal_state)

        img = np.ones((self.height, self.width, 3), dtype=np.float32)
        walls = self.occupancy == 1
        img[walls] = 0.0

        img[r_goal, c_goal] = np.array([0.0, 0.0, 1.0])  # blue goal
        img[r_agent, c_agent] = np.array([1.0, 0.0, 0.0])  # red agent

        scale = 32
        img_large = np.kron(img, np.ones((scale, scale, 1)))

        fig = plt.figure(figsize=(4, 4), dpi=100)
        plt.imshow(img_large, interpolation='nearest')
        plt.axis('off')
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buff.seek(0)
        frame = imageio.imread(buff)
        buff.close()
        return frame

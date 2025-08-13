# experiments/run_fourroom.py
import os
import numpy as np
import sys
from tqdm import tqdm

from core.utils import load_config, ensure_dirs, set_global_seeds
from core.logger import CSVLogger
from envs.fourrooms import FourRoomsEnv
from agents.option_critic import TabularOptionCritic

def run_experiment(config_path, record_gif=False):
    cfg = load_config(config_path)

    # IO paths
    raw_dir = cfg["io"]["raw_dir"]
    plots_dir = cfg["io"]["plots_dir"]
    ensure_dirs(raw_dir, plots_dir, cfg["io"]["aggregates_dir"])

    # Env/Agent params
    n_options = cfg["agent"]["num_options"]
    n_actions = cfg["agent"]["num_actions"]
    gamma = cfg["env"]["gamma"]
    temp = cfg["agent"]["temperature"]
    alpha_c = cfg["agent"]["alpha_critic"]
    alpha_th = cfg["agent"]["alpha_theta"]
    alpha_b = cfg["agent"]["alpha_beta"]
    epsilon_option = cfg["agent"]["epsilon_option"]

    episodes = cfg["train"]["episodes"]
    seeds = cfg["train"]["seeds"]
    log_every = cfg["train"]["log_every"]
    fail_prob = cfg["env"]["fail_prob"]
    max_steps_per_episode = cfg["env"]["max_steps_per_episode"]
    reward_goal = cfg["env"]["reward_goal"]
    reward_default = cfg["env"]["reward_default"]

    for seed in tqdm(seeds, desc=f"OC-{n_options}", dynamic_ncols=True, leave=True):
        print(f"[INFO] Running seed={seed} with {n_options} options")
        sys.stdout.flush()
        set_global_seeds(seed)
        env = FourRoomsEnv(
            fail_prob=fail_prob,
            reward_goal=reward_goal,
            reward_default=reward_default,
            max_steps=max_steps_per_episode,
            seed=seed,
        )
        n_states = env.nS

        agent = TabularOptionCritic(
            n_states=n_states,
            n_actions=n_actions,
            n_options=n_options,
            gamma=gamma,
            alpha_critic=alpha_c,
            alpha_theta=alpha_th,
            alpha_beta=alpha_b,
            temperature=temp,
            epsilon_option=epsilon_option,
            seed=seed
        )

        out_csv = os.path.join(raw_dir, f"oc_{n_options}_seed_{seed}.csv")
        logger = CSVLogger(out_csv, columns=["episode", "steps", "goal_reached", "seed", "n_options"])

        for ep in range(1, episodes + 1):
            # switch goal at ep = goal_switch_episode + 1 (as per figure 2 at exactly 1000)
            s = env.reset()
            # after s = env.reset()
            if env.episode_count > env.goal_switch_episode and env.episode_count <= env.goal_switch_episode + 200:
                agent.temperature = 0.05   # more exploration right after switch
            else:
                agent.temperature = 0.001  # paper value

            agent.start_episode(s)

            steps = 0
            goal_reached = 0
            done = False
            while not done and steps < max_steps_per_episode:
                steps += 1
                a = agent.select_action(s, agent.current_option)
                s_next, r, done, _ = env.step(a)
                if r > 0.0:
                    goal_reached = 1
                agent.step(s, a, r, s_next, done)
                s = s_next

            # 👇 Add this for visible sanity print every 10 episodes
            if ep % 10 == 0 or ep == 1:
                print(f"  Seed {seed} | Episode {ep} | Steps: {steps} | Goal reached: {goal_reached}", flush=True)

            if ep % log_every == 0:
                logger.log_row(episode=ep,
                               steps=steps,
                               goal_reached=goal_reached,
                               seed=seed,
                               n_options=n_options)

        logger.flush()
        print(f"[DONE] Seed {seed} completed: saved {out_csv}")
        sys.stdout.flush()


def run_both_from_cli(record_gif=False):
    run_experiment("configs/fourrooms_oc_4.yaml", record_gif=record_gif)
    run_experiment("configs/fourrooms_oc_8.yaml", record_gif=record_gif)


# tests/test_env_goal_shift.py

from envs.fourrooms import FourRoomsEnv

def test_goal_shift():
    env = FourRoomsEnv(seed=42)

    # Episode 1–1000: goal should be in east doorway (1, 6)
    for ep in [1, 100, 500, 1000]:
        s = env.reset(goal_in_east_doorway=True)
        goal_r, goal_c = env._from_state(env.goal_state)
        assert (goal_r, goal_c) == (1, 6), f"Episode {ep}: Goal should be at (1,6), got ({goal_r},{goal_c})"
    print("[PASS] Goal is correctly placed at east doorway for episodes ≤ 1000")

    # Episode 1001+: goal should move to lower-right room (r > 6 and c > 6)
    for ep in [1001, 1200, 1500]:
        s = env.reset(goal_in_east_doorway=False)
        goal_r, goal_c = env._from_state(env.goal_state)
        assert (goal_r > 6 and goal_c > 6), \
            f"Episode {ep}: Goal should be in lower-right room, got ({goal_r},{goal_c})"
    print("[PASS] Goal is correctly moved to lower-right room for episodes > 1000")


if __name__ == "__main__":
    test_goal_shift()

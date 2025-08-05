# viz/trajectory_gif.py
import os
import io
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def _draw_frame(env, agent_state, goal_state, agent_color='red', goal_color='blue'):
    """Return an RGB numpy array image of the current env with agent+goal drawn."""
    r_agent, c_agent = env._from_state(agent_state)
    r_goal, c_goal = env._from_state(goal_state)

    # Create a base image: walls=black, free=white
    img = np.ones((env.height, env.width, 3), dtype=np.float32)
    walls = env.occupancy == 1
    img[walls] = 0.0  # black walls

    # Draw goal (blue)
    img[r_goal, c_goal] = np.array([0.0, 0.0, 1.0])
    # Draw agent (red)
    img[r_agent, c_agent] = np.array([1.0, 0.0, 0.0])

    # Upscale for nicer visuals
    scale = 32
    img_large = np.kron(img, np.ones((scale, scale, 1)))

    # Convert to a bytes buffer via matplotlib (so we get axes-free image)
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

def save_trajectory_gif(env, states, goal_state, out_path, fps=15):
    """Make sure out_path dir exists and write an animated gif of the trajectory."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = []
    for s in states:
        frames.append(_draw_frame(env, s, goal_state))
    imageio.mimsave(out_path, frames, fps=fps)

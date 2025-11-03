#!/usr/bin/env python3
"""
Show a simple MyoLeg model from MyoSuite using MuJoCo viewer.
No reinforcement learning — just visualization.

Make sure you have:
  pip install myosuite mujoco gymnasium
"""
from myosuite.utils import gym  # this registers all MyoSuite environments
import time
import os
from IPython.display import HTML
from base64 import b64encode
import numpy as np
import skvideo.io

def show_video(video_path, video_width = 400):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")


# Pick a leg environment (available ones include 'myoLeg-v0', 'myoLegWalk-v0', etc.)
env_id = "myoLegWalk-v0"  # or "myoLegWalk-v0" if available

# Create the environment
env = gym.make(env_id)  # use "human" to open the viewer

# Reset to start state
obs, info = env.reset(seed=42)
frames = []

# Just run a few random steps to see movement
for _ in range(1000):
    frames.append(env.sim.renderer.render_offscreen(
                        width=400,
                        height=400,
                        camera_id=0))
        
    action = env.action_space.sample()  # random muscle activations
    env.step(action)
    
env.close()
del env

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/walk_trial.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

# show in the notebook
show_video('videos/walk_trial.mp4')
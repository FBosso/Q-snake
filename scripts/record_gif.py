"""Train an agent headless and record a few games along the way into a GIF.

Usage: uv run python scripts/record_gif.py
"""

import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import torch
from PIL import Image

from snake_rl import game as game_module
from snake_rl.agent import Agent
from snake_rl.game import SnakeGameAI

RECORD_GAMES = [1, 50, 80, 100]
MAX_FRAMES_PER_GAME = 300
FRAME_MS = 40
SCALE = 0.5
OUT = sys.argv[1] if len(sys.argv) > 1 else "assets/training.gif"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# tiny network: extra threads only add overhead
torch.set_num_threads(1)

# no need to wait between frames when nobody is watching
game_module.SPEED = 0

label_font = pygame.font.Font(None, 48)


def grab(display, n_game):
    surface = display.copy()
    label = label_font.render(f"Game {n_game}", True, (255, 255, 255))
    surface.blit(label, (surface.get_width() - label.get_width() - 10, 8))
    img = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
    w, h = img.size
    return img.resize((int(w * SCALE), int(h * SCALE)), Image.NEAREST)


def main():
    agent = Agent()
    game = SnakeGameAI()
    frames = []
    recording = []

    while agent.n_games < RECORD_GAMES[-1]:
        n_game = agent.n_games + 1
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if n_game in RECORD_GAMES and not done and len(recording) < MAX_FRAMES_PER_GAME:
            recording.append(grab(game.display, n_game))

        if done:
            if recording:
                # hold the last frame so the end of each game is readable
                frames += recording + [recording[-1]] * 15
                print(f"recorded game {n_game}: score {score}, {len(recording)} frames")
                recording = []
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"saved {OUT} ({len(frames)} frames, {os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

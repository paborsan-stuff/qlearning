"""
Educational Use License

Copyright (c) 2025 UP Student IA Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, and distribute the Software solely for educational and
non-commercial purposes, subject to the following conditions:

[Include the full text as above or a reference to the LICENSE file]

This license is intended for educational purposes only.
"""

import argparse
import mujoco
from RoboticArmEnv import RoboticArmEnv

from robot_descriptions.loaders.mujoco import load_robot_description
# Cargar configuración de Q-learning

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for training a Deep Q-Network model for robotic simulations."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create parser for the "train" command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose output"
    )
    train_parser.add_argument(
        "--log", action="store_true", help="Enable logging during training"
    )
    train_parser.add_argument(
        "--steps", type=int, default=100, help="Number of training steps"
    )
    train_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes"
    )
    args = parser.parse_args()
    
    # Create parser for the "inference" command
    inference_parser = subparsers.add_parser("inference", help="Create an inference with the model")

    if args.command == "train":
        mdl = load_robot_description("so_arm100_mj_description")
        env = RoboticArmEnv(mdl, debug_mode=args.debug, steps=args.steps, max_ep=args.episodes)
        env.train(episodes=1)
    elif args.command == "inference":
        mdl = load_robot_description("so_arm100_mj_description")
        env = RoboticArmEnv(mdl, debug_mode=args.debug, steps=args.steps, max_ep=args.episodes)
        env.inference(episodes=1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
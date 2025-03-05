import argparse
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
    
    args = parser.parse_args()
    
    if args.command == "train":
        mdl = load_robot_description("so_arm100_mj_description")
        env = RoboticArmEnv(mdl, debug_mode=args.debug)
        env.train(episodes=1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
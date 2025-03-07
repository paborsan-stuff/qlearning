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
    train_parser.add_argument(
        "--log", action="store_true", help="Enable logging during training"
    )
    train_parser.add_argument(
        "--memory_size", type=int, default=10000, help="Set the memory size for the training"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=64, help="Set the batch size for the training"
    )
    train_parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Set the epsilon for the training"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        mdl = load_robot_description("so_arm100_mj_description")
        env = RoboticArmEnv(mdl, debug_mode=args.debug,memory_size=args.memory_size, batch_size=args.batch_size, epsilon=args.epsilon)
        # Passing the memory_size and batch_size to the train method
        env.train(episodes=1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

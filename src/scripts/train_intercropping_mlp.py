import argparse
from pathlib import Path
from typing import Optional

from src.dataset.intercropping.intercropping_mlp import IntercroppingMLP
from src.dataset.intercropping.readers.dataset_loader import DatasetLoader
from src.trainings.intercropping_mlp_trainer import IntercroppingMLPTrainer
from src.utils.configs.config_reader import ConfigReader
from src.utils.configs.ini_config_reader import INIConfigReader

"""
Intercropping MLP Training and Evaluation Script

This script provides command-line utilities for training and evaluating the Intercropping MLP model.
It supports model training with various configurations, data augmentation, checkpointing, and evaluation.

Usage:
    Training:
        python script.py train --config CONFIG_PATH --dataset DATASET_PATH [options]

    Evaluation:
        python script.py evaluate --config CONFIG_PATH --dataset DATASET_PATH [options]

Examples:
    Train a model:
        >>> python script.py train --config configs/default.ini --dataset data/intercrop.csv --plot

    Train with custom settings:
        >>> python script.py train --config configs/custom.ini --dataset data/intercrop.csv --log-level 2 --no-augment

    Resume training from checkpoint:
        >>> python script.py train --config configs/default.ini --dataset data/intercrop.csv --checkpoint checkpoints/model_50.pth

    Evaluate a trained model:
        >>> python script.py evaluate --config configs/default.ini --dataset data/intercrop.csv --log-level 3
"""


def _get_path(directory: str, file: str):
    """
    Resolves and validates file paths relative to specified directory.

    :param directory: Base directory name ('configs' or 'data')
    :param file: File name or path
    :return: Full validated path to the file
    :raises FileNotFoundError: If the file doesn't exist
    """
    path = Path(file)
    if len(path.parts) == 1:  # Only filename provided: add "configs" before the path
        path = Path(__file__).parent.parent.parent / Path('data') / Path(directory) / path
    config_path = str(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    return config_path


def _load_model(config: ConfigReader):
    """
    Creates and configures an IntercroppingMLP model instance based on configuration.

    :param config: Configuration reader containing model parameters
    :return: Configured IntercroppingMLP model instance
    :raises TypeError: If hidden_layers_sizes configuration is invalid
    """
    act_function = config.get_param('model.activation_function', v_type=str)
    hidden_layers_sizes = config.get_param('model.hidden_layers_sizes', v_type=tuple)
    try:
        hidden_layers_sizes = tuple([int(hl) for hl in hidden_layers_sizes])
    except TypeError:
        raise TypeError("hidden_layers_sizes in the config file must be a tuple of int")

    return IntercroppingMLP(
        hidden_layers_sizes=hidden_layers_sizes,
        act_function=act_function
    )


def train_intercropping_mlp(config_path: str, dataset_path: str, evaluate: bool = True,
                            augment: bool = True, plot: bool = False, log_level: int = 1,
                            checkpoint_file: Optional[str] = None) -> None:
    """
    Train the Intercropping MLP model.

    Args:
        config_path: Path to the configuration file
        dataset_path: Path to the dataset
        evaluate: Whether to evaluate after training
        augment: Whether to use data augmentation
        plot: Whether to generate plots
        log_level: Logging verbosity level
        checkpoint_file: Path to checkpoint file to resume training from
    """

    dataset_loader = DatasetLoader(dataset_path)
    config_reader = INIConfigReader(config_path)
    model = _load_model(config_reader)
    trainer = IntercroppingMLPTrainer(
        model=model,
        dataset_loader=dataset_loader,
        config_reader=config_reader
    )

    trainer.train(
        log_level=log_level,
        plot=plot,
        augment=augment,
        start_from_checkpoint=checkpoint_file
    )

    if evaluate:
        evaluate_intercropping_mlp(config_path, dataset_path, log_level=log_level)


def evaluate_intercropping_mlp(config_path: str, dataset_path: str, log_level: int = 1) -> None:
    """
    Evaluates a trained Intercropping MLP model.

    :param config_path: Path to the configuration file
    :param dataset_path: Path to the dataset file
    :param log_level: Logging verbosity (0: silent, 1: basic, 2: detailed, 3: debug)
    """
    dataset_loader = DatasetLoader(dataset_path)
    config_reader = INIConfigReader(config_path)
    model = _load_model(config_reader)
    trainer = IntercroppingMLPTrainer(
        model=model,
        dataset_loader=dataset_loader,
        config_reader=config_reader
    )

    trainer.evaluate(log_level=log_level)


def main():
    usage = """
    Main entry point handling command-line arguments for training and evaluation.

    Command-line Arguments:
        Train mode:
            --config: Path to configuration file
            --dataset: Path to dataset file
            --no-evaluate: Skip evaluation after training
            --no-augment: Disable data augmentation
            --plot: Enable training progress plots
            --log-level: Logging verbosity level (default: 1)
            --checkpoint: Path to checkpoint file for resuming training

        Evaluate mode:
            --config: Path to configuration file
            --dataset: Path to dataset file
            --log-level: Logging verbosity level (default: 1)
    """
    parser = argparse.ArgumentParser(description='Train or evaluate Intercropping MLP model',epilog=usage)
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, required=True,
                              help='Path to configuration file or name of the configuration file')
    train_parser.add_argument('--dataset', type=str, required=True,
                              help='Path to dataset')
    train_parser.add_argument('--no-evaluate', action='store_true',
                              help='Skip evaluation after training')
    train_parser.add_argument('--no-augment', action='store_true',
                              help='Disable data augmentation')
    train_parser.add_argument('--plot', action='store_true',
                              help='Generate plots during training')
    train_parser.add_argument('--log-level', type=int, default=1,
                              help='Logging verbosity level')
    train_parser.add_argument('--checkpoint', type=str,
                              help='Path to checkpoint file to resume training from')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration file')
    eval_parser.add_argument('--dataset', type=str, required=True,
                             help='Path to dataset')
    eval_parser.add_argument('--log-level', type=int, default=1,
                             help='Logging verbosity level')

    args = parser.parse_args()

    if args.command == 'train':
        train_intercropping_mlp(
            config_path=_get_path('configs', args.config),
            dataset_path=_get_path('datasets', args.dataset),
            evaluate=not args.no_evaluate,
            augment=not args.no_augment,
            plot=args.plot,
            log_level=args.log_level,
            checkpoint_file=args.checkpoint
        )
    elif args.command == 'evaluate':
        evaluate_intercropping_mlp(
            config_path=args.config,
            dataset_path=args.dataset,
            log_level=args.log_level
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

from pathlib import Path
from typing import Dict, Optional


def main(config: Optional[str]):
    """
    :param config: name of config file to use passed with main arguments. Default: None
    """
    pass


if __name__ == '__main__':
    """
    Usage example:
        python main.py --config "config.ini"
        # It will automatically retrieve the config inside the configs/ directory
    or:
        python main.py --config "other_configs/config.ini"
        # For configuration files outside the configs/directory
    or: 
        python main.py
        # No configs specified
    """
    import argparse

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run application with configuration file')
    parser.add_argument('--config', '-c',
                        type=str,
                        default=None,
                        help='Path to configuration file (default: config.ini)')
    args = parser.parse_args()
    config_path = None
    if args.config:
        path = Path(args.config)
        if len(path.parts) == 1:  # Only filename provided: add "configs" before the path
            path = Path('configs') / path
        config_path = str(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
    main(config_path)

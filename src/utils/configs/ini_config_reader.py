import configparser
from pathlib import Path
from typing import Union, Optional, Dict, Any

from src.utils.configs.config_reader import ConfigReader


class INIConfigReader(ConfigReader):
    """
    A class to read and manage configuration parameters from INI files\
    See 'ConfigReader' methods documentation for further reference.
    :param config_path: Path to the INI configuration file
    :ivar config_data: Dictionary containing the parsed configuration data
                      Structure: {section_name: {param_name: param_value}}
                      Example: {'database': {'host': 'localhost', 'port': '5432'}}
    :raises FileNotFoundError: If the configuration file doesn't exist
    :raises ValueError: If the file is not an INI file

    Example:
        # Create a config.ini file
        ################ config.ini ####################
        [database]
        host = localhost
        port = 5432
        username = admin ; Comments also this way!
        restricted = true

        [api]
        url = https://api.example.com
        timeout = 30
        ################################################

        # Use the ConfigReader
        config = ConfigReader("config.ini")

        # Get specific parameters
        host = config.get_param("database.host")  # Returns "localhost"
        port = config.get_param("database.port", default="5432")  # With default value

        # Get entire section
        api_settings = config.get_section("api")  # Returns dict with all api settings

        # Dictionary-style access
        db_config = config["database"]  # Returns entire database section

        # Use default value if no value is provided
        default_value = config.get_param("database.default", default=None)
    """

    def __init__(self, config_path: Union[str, Path], base_path: Path = Path('')):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        if self.config_path.suffix.lower() != '.ini':
            raise ValueError(f"File must be an INI file, got: {self.config_path.suffix}")
        super().__init__(self._load_config(), base_path)

    def _load_config(self) -> dict[str, dict[str, str]]:
        """
        Load the configuration from the INI file\
        :raises configparser.Error: If there's an error parsing the INI file
        """
        parser = configparser.ConfigParser(inline_comment_prefixes=(';',))
        parser.read(self.config_path)
        config_data = {}
        for section in parser.sections():
            config_data[section] = dict(parser[section])
        return config_data

    def get_param(self, param_path: str, default: Any = None, v_type: type = None, nullable=False,domain=None) -> Any:
        try:
            return super().get_param(param_path, default, v_type, nullable,domain)
        except (ValueError, TypeError) as e:
            raise type(e)(f'Error with configuration file "{self.config_path}": {str(e)}') from e

    def get_collection(self, param_path: str, default: Any = None, v_type: type = None, collection_type: type = tuple,
                       nullable: bool = False, num_elems: int = None,domain=None):
        try:
            return super().get_collection(param_path, default, v_type, collection_type, nullable, num_elems,domain)
        except (ValueError, TypeError) as e:
            raise type(e)(f'Error with configuration file "{self.config_path}": {str(e)}') from e

    def get_section(self, section: str) -> Optional[Dict[str, str]]:
        try:
            return super().get_section(section)
        except (ValueError, TypeError) as e:
            raise type(e)(f'Error with configuration file "{self.config_path}": {str(e)}') from e
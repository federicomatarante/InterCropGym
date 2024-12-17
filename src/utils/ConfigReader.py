import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigReader:
    """
    A class to read and manage configuration parameters from INI files\
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
        username = admin

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

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Dict[str, str]] = {}

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        if self.config_path.suffix.lower() != '.ini':
            raise ValueError(f"File must be an INI file, got: {self.config_path.suffix}")

        self._load_config()

    def _load_config(self) -> None:
        """
        Load the configuration from the INI file\
        :raises configparser.Error: If there's an error parsing the INI file
        """
        parser = configparser.ConfigParser()
        parser.read(self.config_path)

        self.config_data = {}
        for section in parser.sections():
            self.config_data[section] = dict(parser[section])

    def get_param(self, param_path: str, default: Any = None) -> Any:
        """
        Get a parameter value using dot notation path\
        :param param_path: Path to the parameter using dot notation (e.g., 'database.host')
        :param default: Default value to return if parameter is not found
        :return: The parameter value if found, otherwise the default value
        """
        try:
            section, param = param_path.split('.')
            return self.config_data[section][param]
        except (KeyError, ValueError):
            return default

    def get_section(self, section: str) -> Optional[Dict[str, str]]:
        """
        Get all parameters in a section\
        :param section: Name of the configuration section
        :return: Dictionary containing all parameters in the section, or None if not found
        """
        return self.config_data.get(section)

    def __getitem__(self, key: str) -> Dict[str, str]:
        """
        Allow dictionary-style access to configuration sections\
        :param key: Section name
        :return: Dictionary containing all parameters in the section
        :raises KeyError: If the section is not found
        """
        return self.config_data[key]
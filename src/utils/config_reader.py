from typing import Optional, Dict, Any


class ConfigReader:
    """
    A configuration reader class that provides convenient access to configuration data\
    The configuration dictionary must be organized in sections, where each section contains related parameters\
    Config dictionary example:
        {
           'section1': {
               'param1': value1,
               'param2': value2
           },
           'section2': {
               'param1': value1,
               'param2': value2
           }
        }

    Usage example:
        config_data = {
           'app_settings': {
               'debug_mode': True,
               'log_level': 'INFO'
           },
           'display': {
               'width': 800,
               'height': 600,
               'fullscreen': False
           }
        }
        config = ConfigReader(config_data)
        debug_enabled = config.get_param('app_settings.debug_mode')    # Returns True
        display_conf = config.get_section('display')                   # Returns {'width': 800, 'height': 600, 'fullscreen': False}
        app_conf = config['app_settings']                             # Returns {'debug_mode': True, 'log_level': 'INFO'}
    """

    def __init__(self, config_data: Dict):
        """
        Initialize the ConfigReader with configuration data\
        :param config_data: Dictionary containing the configuration data organized in sections
        """
        self.config_data = config_data

    def get_param(self, param_path: str, default: Any = None) -> Any:
        """
        Get a parameter value using dot notation path\
        :param param_path: Path to the parameter using dot notation (e.g., 'database.host' where 'database' is the section)
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

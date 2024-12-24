class ValueParser:
    """
    Static helper class for parsing various data types from string inputs.

    This class provides methods to safely parse different value types while handling
    invalid inputs and special cases.

    Examples:
        ValueParser.parse_float("12.5")
        12.5
        ValueParser.parse_boolean("Yes")
        True
        ValueParser.parse_soil_texture("Sandy Loam (USD)")
        "Sandy Loam"
    """

    @staticmethod
    def parse_float(value: str, default: float = -1.0) -> float:
        """
        Parse a string into a float value, handling special cases and invalid inputs.
        Special cases:
            - -12,6
            - 'NA', 'Unclear', 'Varying' ( transforms them into default )
            - 124.1 214 alpha ( removes trailer words or number separated by space)
            - other values ( transforms them into default value )
        :param value: String value to parse.
        :param default: Default value to return if parsing fails, defaults to -1.0
        :return: Parsed float value or default value if parsing fails

        Examples:
            ValueParser.parse_float("12.5")
            12.5
            ValueParser.parse_float("NA")
            -1.0
        """
        try:
            if value in ('NA', 'Unclear', 'Varying'):
                return default
            return float(value.split(" ")[0].replace('\ufeff', '').replace(',', '.'))
        except (ValueError, AttributeError):
            return default

    @staticmethod
    def parse_boolean(value: str, default=False) -> bool:
        """
        Parse a string into a boolean value.

        :param value: String value to parse ('Yes' or 'No' - case insensitive)
        :param default: default value to use in case given value is not allowed.
        :return: True if value is 'Yes', False if 'No'. Else 'Default'
        """
        if value.lower() == 'yes':
            return True
        if value.lower() == 'no':
            return False
        return default

    @staticmethod
    def parse_soil_texture(value: str) -> str:
        """
        Parse soil texture value, removing the trailing word if present.

        :param value: Soil texture description
        :return: Parsed soil texture or 'NA' if invalid
        """
        return " ".join(value.split(" ")[0:-1]) if value != "NA" else "NA"

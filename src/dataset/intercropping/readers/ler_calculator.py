
from typing import Tuple, Optional, Dict

from src.dataset.intercropping.readers.value_parser import ValueParser


class LerCalculator:
    """
    Helper class for calculating Land Equivalent Ratio (LER) values.
    :param record: Dictionary containing crop data
    The input dictionary must contain the following keys:
        - LER_crop1: str - Direct LER value for crop 1 or 'NA'
        - LER_crop2: str - Direct LER value for crop 2 or 'NA'
        - Crop_1_yield_intercropped: str - Yield of crop 1 in intercropping system
        - Crop_1_yield_sole: str - Yield of crop 1 in sole cropping
        - Crop_2_yield_intercropped: str - Yield of crop 2 in intercropping system
        - Crop_2_yield_sole: str - Yield of crop 2 in sole cropping
    This class handles the calculation of LER values for both crops in an intercropping system.

    Examples:
        record = {"LER_crop1": "1.2", "LER_crop2": "0.8", ...}
        calculator = LerCalculator(record)
        ler_values = calculator.calculate_ler_values()
        print(f"LER values: {ler_values}")

    Attributes:
        record (Dict): Dictionary containing crop yield and LER data
    """

    def __init__(self, record: Dict):
        self.record = record

    def calculate_ler_values(self) -> Optional[Tuple[float, float]]:
        """
        Calculate LER values for both crops.

        :return: Tuple of LER values for both crops, or None if calculation is not possible
        """
        ler1 = self.calculate_single_ler("1")
        ler2 = self.calculate_single_ler("2")

        if ler1 is not None and ler2 is not None:
            return ler1, ler2
        return None

    def calculate_single_ler(self, crop_number: str) -> Optional[float]:
        """
        Calculate LER value for a single crop.

        :param crop_number: Crop identifier ("1" or "2")
        :return: Calculated LER value or None if calculation is not possible
        """
        ler_value = self.record[f"LER_crop{crop_number}"]
        if ler_value != 'NA':
            return ValueParser.parse_float(ler_value)

        yield_inter = self.record[f"Crop_{crop_number}_yield_intercropped"]
        yield_sole = self.record[f"Crop_{crop_number}_yield_sole"]

        if yield_inter == 'NA' or yield_sole == 'NA':
            return None

        yield_sole_float = ValueParser.parse_float(yield_sole)
        if yield_sole_float == 0.0:
            return None

        yield_inter_float = ValueParser.parse_float(yield_inter)
        return yield_inter_float / yield_sole_float

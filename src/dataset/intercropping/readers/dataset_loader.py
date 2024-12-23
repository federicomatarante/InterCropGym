from pathlib import Path
from typing import List, Tuple

from src.dataset.intercropping.readers.ler_calculator import LerCalculator
from src.dataset.intercropping.readers.record_parser import RecordParser
from src.dataset.intercropping.utils.parameters import Crops, IntercroppingDescriptors, ExperimentalSite, \
    CropManagementPractices
from src.utils.csv_reader import CsvReader


class DatasetLoader:
    """
    Main class for loading and processing the intercropping dataset.
    :param file_path: Path to the dataset file
    The input CSV file must contain the following columns:
        Crop Information:
            - Crop_1_Common_Name: str - Common name of the first crop
            - Crop_2_Common_Name: str - Common name of the second crop

        Intercropping Information:
            - Intercropping_design: str - Design of the intercropping system
            - Intercropping_pattern: str - Pattern of the intercropping system

        Management Practices:
            - Greenhouse: str - 'Yes' or 'No'
            - Organic_ferti: str - 'Yes' or 'No'
            - Mineral_ferti: str - 'Yes' or 'No'
            - Nitrogen_rate_kg_ha: str - Nitrogen rate or 'NA'/'Unclear'/'Varying'
            - Pesticide: str - 'Yes' or 'No'
            - Irrigation: str - 'Yes' or 'No'

        Site Information:
            - Climate_zone: str - Climate zone classification
            - Soil_texture: str - Soil texture description
            - Soil_pH: str - Soil pH value or 'NA'

        LER and Yield Data:
            - LER_crop1: str - LER value for crop 1 or 'NA'
            - LER_crop2: str - LER value for crop 2 or 'NA'
            - Crop_1_yield_intercropped: str - Yield of crop 1 in intercropping
            - Crop_1_yield_sole: str - Yield of crop 1 in sole cropping
            - Crop_2_yield_intercropped: str - Yield of crop 2 in intercropping
            - Crop_2_yield_sole: str - Yield of crop 2 in sole cropping

    The CSV must use semicolon (;) as delimiter and UTF-8 with BOM encoding.
    This class coordinates the loading and parsing of the dataset, using helper classes
    to process individual components.

    Examples:
        # Basic usage with default file path
        loader = DatasetLoader()
        dataset = loader.load()

        # Usage with custom file path
        loader = DatasetLoader('custom/path/data.csv')
        dataset = loader.load()

        # Processing specific components
        for data, ler in dataset:
        ...     crops, intercropping, site, management = data
        ...     print(f"Crops: {crops}, LER: {ler}")

    Attributes:
        file_path (Path): Path to the dataset file
        dataset (List): List to store processed dataset entries
    """

    def __init__(self, file_path: str):

        self.file_path = Path(file_path)
        self.dataset = []

    def load(self) -> List[Tuple[
        Tuple[
            Crops,  # crop1
            Crops,  # crop2
            IntercroppingDescriptors,  # intercropping_info
            ExperimentalSite,  # site_info
            CropManagementPractices  # management_info
        ],
        Tuple[float, float]  # (ler1, ler2)
    ]]:
        """
        Load and process the dataset.

        :return: List of tuples containing processed data and LER values
        :raises FileNotFoundError: If the dataset file is not found
        :raises ValueError: If there are issues parsing the data
        """
        csv_reader = CsvReader(self.file_path)
        records = csv_reader.read()

        for record in records:
            try:
                parser = RecordParser(record)
                ler_calculator = LerCalculator(record)

                crops = parser.parse_crops()
                intercropping_info = parser.parse_intercropping()
                management_info = parser.parse_management()
                site_info = parser.parse_site_info()
                ler_values = ler_calculator.calculate_ler_values()

                if ler_values:
                    self.dataset.append(
                        ((crops[0], crops[1], intercropping_info, site_info, management_info),
                         ler_values)
                    )
            except ValueError as e:
                print(f"Skipping record due to error: {e}")

        return self.dataset

from typing import Tuple, Dict

from src.dataset.intercropping.utils.parameters import Crops, IntercroppingDescriptors, IntercroppingDesigns, \
    IntercroppingPatterns, CropManagementPractices, ExperimentalSite, SoilTextures, ClimateZones
from src.dataset.intercropping.readers.value_parser import ValueParser


class RecordParser:
    """
    Helper class for parsing different components of a data record.

    :param record: Dictionary containing the record data
    The input record must contain the following keys:
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
    This class handles the parsing of various record components including crops,
    intercropping information, management practices, and site information.

    Examples:
        record = {"Crop_1_Common_Name": "Maize", "Crop_2_Common_Name": "Beans", ...}
        parser = RecordParser(record)
        crops = parser.parse_crops()
        management = parser.parse_management()

    Attributes:
        record (Dict): Dictionary containing the record data to be parsed
    """

    def __init__(self, record: Dict):
        self.record = record

    def parse_crops(self) -> Tuple[Crops, Crops]:
        """
        Parse crop information from the record.

        :return: Tuple containing two Crops objects
        :raises ValueError: If crop names are invalid
        """
        return (
            Crops.from_value(self.record["Crop_1_Common_Name"]),
            Crops.from_value(self.record["Crop_2_Common_Name"])
        )

    def parse_intercropping(self) -> IntercroppingDescriptors:
        """
        Parse intercropping information from the record.

        :return: IntercroppingDescriptors object containing design and pattern
        :raises ValueError: If intercropping design or pattern is invalid
        """
        return IntercroppingDescriptors(
            design=IntercroppingDesigns.from_value(self.record["Intercropping_design"]),
            pattern=IntercroppingPatterns.from_value(self.record["Intercropping_pattern"])
        )

    def parse_management(self) -> CropManagementPractices:
        """
        Parse management practices from the record.

        :return: CropManagementPractices object containing management information
        """
        return CropManagementPractices(
            greenhouse=ValueParser.parse_boolean(self.record["Greenhouse"]),
            organic_fertilizer=ValueParser.parse_boolean(self.record["Organic_ferti"]),
            mineral_fertilizer=ValueParser.parse_boolean(self.record["Mineral_ferti"]),
            nitrogen_rate=ValueParser.parse_float(self.record["Nitrogen_rate_kg_ha"]),
            pesticide=ValueParser.parse_boolean(self.record["Pesticide"], default=True),
            irrigation=ValueParser.parse_boolean(self.record["Irrigation"], default=True),
        )

    def parse_site_info(self) -> ExperimentalSite:
        """
        Parse experimental site information from the record.

        :return: ExperimentalSite object containing site information
        :raises ValueError: If climate zone or soil texture is invalid
        """
        return ExperimentalSite(
            climate_zone=ClimateZones.from_value(self.record["Climate_zone"]),
            soil_texture=SoilTextures.from_value(
                ValueParser.parse_soil_texture(self.record["Soil_texture"])
            ),
            soil_ph=ValueParser.parse_float(self.record["Soil_pH"])
        )

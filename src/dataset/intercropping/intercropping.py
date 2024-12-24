from abc import ABC, abstractmethod
from typing import Tuple

from src.dataset.intercropping.utils.parameters import CropManagementPractices, ExperimentalSite, IntercroppingDescriptors, \
    Crops


class Intercropping(ABC):

    @abstractmethod
    def get_results(self, crop_1: Crops, crop_2: Crops, intercropping_description: IntercroppingDescriptors,
                    site_conditions: ExperimentalSite,
                    crop_management_practices: CropManagementPractices) -> Tuple[
        float, float]:
        """
        Calculate Land Equivalent Ratio (LER) for two intercropped species.
        LER is the relative land area under sole crops that is required to produce
        the yields achieved in intercropping.

        :param crop_1: Name of the first crop species. Must be registered in the dataset
        :param crop_2: Name of the second crop species. Must be registered in the dataset
        :param intercropping_description: Design descriptors for the intercropping system (pattern and design)
        :param site_conditions: Experimental site and climate conditions
        :param crop_management_practices: Management practices applied to the crops
        :return: Tuple containing LER values for crop_1 and crop_2 respectively.
                Each LER value represents the partial land equivalent ratio:
                - LER_1 = yield_intercrop_1 / yield_sole_crop_1
                - LER_2 = yield_intercrop_2 / yield_sole_crop_2
                Where yeld_intercrop is the yield while intercropped, and yield_sole is the yield while being cultivated
                 alone.
                Values > 1.0 indicate intercropping advantage for that crop.
        :raises ValueError: If crop_1_name or crop_2_name are not registered in the dataset
        """
        raise NotImplementedError()

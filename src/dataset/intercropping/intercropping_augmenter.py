from __future__ import annotations
import random
from typing import List, Tuple, TypeAlias

from src.dataset.intercropping.utils.parameters import (
    Crops, IntercroppingDescriptors, ExperimentalSite,
    CropManagementPractices, IntercroppingDesigns, IntercroppingPatterns,
    ClimateZones, SoilTextures
)


class IntercroppingAugmenter:
    """
    A class for augmenting intercropping datasets through various data transformation techniques.

    This class provides methods to create synthetic data points through operations like
    crop swapping and random masking of features. These augmentations can help in
    increasing dataset size and improving model robustness.

    The input dataset should contain records with the following structure:
    - Tuple of (data_info, ler_values) where:
        - data_info is a tuple of (crop1, crop2, intercropping_info, site_info, management_info)
        - ler_values is a tuple of (ler1, ler2) representing Land Equivalent Ratios

    Example usage:
        dataset = [...]  # Your original dataset
        augmenter = IntercroppingAugmenter(dataset)

        # Perform crop swapping on 30% of the data
        swapped_data = augmenter.random_crops_swap(0.3)

        # Perform random masking
        masked_data = augmenter.random_masking(0.3, 0.3)

        # Chain augmentations
        swapped = augmenter.random_crops_swap(0.3)
        masked = IntercroppingAugmenter(swapped).random_masking(0.5, 0.2)
    """

    def __init__(self, dataset: List[Tuple[
        Tuple[
            Crops,  # crop1
            Crops,  # crop2
            IntercroppingDescriptors,  # intercropping_info
            ExperimentalSite,  # site_info
            CropManagementPractices  # management_info
        ],
        Tuple[float, float]  # (ler1, ler2)
    ]]):
        """
        Initialize the augmenter with a dataset.

        :param dataset: List of data points to be augmented
        """
        self.dataset = dataset

    def random_crops_swap(self, sample_rate: float = 0.3) -> List[Tuple[
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
        Create augmented data by swapping crops and their corresponding LER values.

        This method randomly selects a subset of the data based on the sample_rate
        and swaps the positions of crop1 with crop2 and their corresponding LER values.
        This augmentation technique helps in ensuring the model learns that the order
        of crops should not affect the predictions.

        :param sample_rate: Fraction of dataset to augment (between 0 and 1)
        :raises ValueError: If sample_rate is not between 0 and 1

        Example:
            augmenter = IntercroppingAugmenter(dataset)
            augmented_data = augmenter.random_crops_swap(0.3)

            # Original record:
            # ((Maize, Beans, info, site, management), (0.5, 0.6))
            # Augmented record:
            # ((Beans, Maize, info, site, management), (0.6, 0.5))
        """
        if not 0 <= sample_rate <= 1:
            raise ValueError("Sample rate must stay between 0 and 1")

        sub_dataset = random.sample(self.dataset, int(len(self.dataset) * sample_rate))
        augmented_dataset = []

        for record in sub_dataset:
            (
                (crop1, crop2, intercropping_info, site_info, management_info),
                (ler1, ler2)
            ) = record

            augmented_record = (
                (crop2, crop1, intercropping_info, site_info, management_info),
                (ler2, ler1)
            )
            augmented_dataset.append(augmented_record)

        return augmented_dataset

    def random_masking(self, sample_rate: float = 0.3, masking_prob: float = 0.3) -> List[Tuple[
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
        Create augmented data by randomly masking (setting to NA/default) various features.

        This method first selects a subset of data based on sample_rate, then for each
        selected record, it randomly masks features based on masking_prob. Masked features
        are set to their respective NA or default values. This technique helps in making
        the model more robust to missing data.

        :param sample_rate: Fraction of dataset to augment (between 0 and 1)
        :param masking_prob: Probability of masking each feature (between 0 and 1)
        :raises ValueError: If sample_rate or masking_prob is not between 0 and 1

        Example:
            augmenter = IntercroppingAugmenter(dataset)
            masked_data = augmenter.random_masking(0.3, 0.3)

            # Original record:
            # ((crop1, crop2,
            #   IntercroppingDescriptors(design=ROW, pattern=STRIP),
            #   ExperimentalSite(climate_zone=TROPICAL, soil_texture=CLAY, soil_ph=6.5),
            #   management_info),
            #  (0.5, 0.6))

            # Possible masked record:
            # ((crop1, crop2,
            #   IntercroppingDescriptors(design=NA, pattern=STRIP),
            #   ExperimentalSite(climate_zone=TROPICAL, soil_texture=NA, soil_ph=-1.0),
            #   management_info),
            #  (0.5, 0.6))
        """
        if not 0 <= sample_rate <= 1:
            raise ValueError("sample_rate must stay between 0 and 1")
        if not 0 <= masking_prob <= 1:
            raise ValueError("masking_prob must stay between 0 and 1")

        sub_dataset = random.sample(self.dataset, int(len(self.dataset) * sample_rate))
        augmented_dataset = []

        for record in sub_dataset:
            (
                (crop1, crop2, intercropping_info, site_info, management_info),
                (ler1, ler2)
            ) = record

            augmented_record = (
                (crop2, crop1,
                 IntercroppingDescriptors(
                     design=intercropping_info.design if random.random() > masking_prob else IntercroppingDesigns.NA,
                     pattern=intercropping_info.pattern if random.random() > masking_prob else IntercroppingPatterns.NA,
                 ),
                 ExperimentalSite(
                     climate_zone=site_info.climate_zone if random.random() > masking_prob else ClimateZones.NA,
                     soil_texture=site_info.soil_texture if random.random() > masking_prob else SoilTextures.NA,
                     soil_ph=site_info.soil_ph if random.random() > masking_prob else -1.0
                 ),
                 CropManagementPractices(
                     nitrogen_rate=management_info.nitrogen_rate if random.random() > masking_prob else -1.0,
                     greenhouse=management_info.greenhouse,
                     organic_fertilizer=management_info.organic_fertilizer,
                     mineral_fertilizer=management_info.mineral_fertilizer,
                     pesticide=management_info.pesticide,
                     irrigation=management_info.irrigation,
                 )),
                (ler2, ler1)
            )
            augmented_dataset.append(augmented_record)

        return augmented_dataset

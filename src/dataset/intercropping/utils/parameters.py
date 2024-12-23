# For storing all field data in a structured way
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import fields
from enum import Enum
from typing import List, TypeVar, Type, Any

import torch


class BaseEnum(Enum):
    """
    Base Enum class that provides utility methods for all subclasses.
    Automatically adds from_value classmethod to get enum members from their values.

    Example:
        class Color(BaseEnum):
            RED = 1
            GREEN = 2
            BLUE = 3

        # Get enum member from value
        color = Color.from_value(1)  # Returns Color.RED
        print(color)  # Color.RED
        print(color.value)  # 1

        # Error handling is automatic
        Color.from_value(99)  # Raises ValueError: 'Color' has no value 99
    """

    @classmethod
    def from_value(cls: Type[Enum], value: Any):
        """
        Get enum member from its value.

        :param value: The value to get the enum member for
        :return: The enum member corresponding to the value
        :raises: ValueError: If no enum member has the specified value
        """
        try:
            return cls._value2member_map_[value]
        except KeyError:
            raise ValueError(f"'{cls.__name__}' has no value {value}")


    @classmethod
    def has_value(cls, value: Any) -> bool:
        """
        Check if the enum has a member with the specified value.

        Args:
            value: The value to check

        Returns:
            True if a member with the value exists, False otherwise
        """
        return value in cls._value2member_map_

    def __str__(self) -> str:
        """Return the enum member's name."""
        return self.name


class Crops(BaseEnum):
    BASIL = "Basil"
    MOTH_BEAN = "Moth bean"
    XANTHOSOMA = "Xanthosoma"
    MUNG_BEAN = "Mung bean"
    GREEN_CHIRETA = "Green chireta"
    VELVET_BEAN = "Velvet bean"
    SPINACH = "Spinach"
    MAIZE = "Maize"
    JACK_BEAN = "Jack bean"
    PEPPER = "Pepper"
    SQUASH = "Squash"
    BEETROOT = "Beetroot"
    RICE = "Rice"
    WATER_YAM = "Water yam"
    STRAWBERRY = "Strawberry"
    WATERMELON = "Watermelon"
    FINGER_MILLET = "Finger millet"
    HOLY_BASIL = "Holy basil"
    EASTERN_COTTONWOOD = "Eastern cottonwood"
    SHALLOT = "Shallot"
    PEANUT = "Peanut"
    SAFED_MUSLI = "Safed musli"
    WHITE_LEADTREE = "White leadtree"
    CHICKPEA = "Chickpea"
    COFFEE_ROBUSTA = "Coffee robusta"
    ALFALFA = "Alfalfa"
    GARLIC = "Garlic"
    TORRELS_EUCALYPTUS = "Torrel's eucalyptus"
    PEARL_MILLET = "Pearl millet"
    BOTTLE_GOURD = "Bottle gourd"
    FENUGREEK = "Fenugreek"
    APRICOT = "Apricot"
    PERENNIAL_RYEGRASS = "Perennial ryegrass"
    LEEK = "Leek"
    LABLAB_BEAN = "Lablab bean"
    COCO_YAM = "Coco yam"
    MANGO = "Mango"
    RUBBER = "Rubber"
    PEAR = "Pear"
    ROOT_PARSLEY = "Root parsley"
    KALE = "Kale"
    ALCHORNEA = "Alchornea"
    CORIANDER = "Coriander"
    DYERS_WOAD = "Dyer's woad"
    LETTUCE = "Lettuce"
    ROUGH_LEMON = "Rough lemon"
    BEAN = "Bean"
    MELON = "Melon"
    TOMATO = "Tomato"
    MARIGOLD = "Marigold"
    EGGPLANT = "Eggplant"
    TURNIP = "Turnip"
    COWPEA = "Cowpea"
    HAIRY_INDIGO = "Hairy indigo"
    BROCCOLI = "Broccoli"
    RATTAN_GRASS = "Rattan grass"
    POTATO = "Potato"
    OLIVE = "Olive"
    CARROT = "Carrot"
    CASSAVA = "Cassava"
    MUSTER_JOHN_HENRY = "Muster John Henry"
    CHICORY = "Chicory"
    SWEET_POTATO = "Sweet potato"
    PEACH = "Peach"
    OKRA = "Okra"
    DURUM_WHEAT = "Durum wheat"
    SAPOTA = "Sapota"
    JUJUBE = "Jujube"
    COLLARD = "Collard"
    JACKFRUIT = "Jackfruit"
    GLICIRIDIA = "Gliciridia"
    RADISH = "Radish"
    PIGEON_PEA = "Pigeon pea"
    AFRICAN_YAM_BEAN = "African yam bean"
    WHEAT = "Wheat"
    MILLET = "Millet"
    CAULIFLOWER = "Cauliflower"
    SOYBEAN = "Soybean"
    CELERY = "Celery"
    RED_CLOVER = "Red clover"
    CEYLON_SPINACH = "Ceylon spinach"
    SIAMESE_CASSIA = "Siamese cassia"
    GMELINA = "Gmelina"
    SORREL = "Sorrel"
    FENNEL = "Fennel"
    CUCUMBER = "Cucumber"
    BANANA = "Banana"
    SUNFLOWER = "Sunflower"
    WITHANIA = "Withania"
    OAT = "Oat"
    TURMERIC = "Turmeric"
    ROCKET = "Rocket"
    ONION = "Onion"
    GREATER_YAM = "Greater yam"
    BRUSSELS_SPROUT = "Brussels sprout"
    COFFEE_ARABICA = "Coffee arabica"
    BLACK_GRAM = "Black gram"
    CHIVE = "Chive"
    APPLE = "Apple"
    GROUNDNUT = "Groundnut"
    CROTALARIA = "Crotalaria"
    PARSLEY = "Parsley"
    RICE_BEAN = "Rice bean"
    FOXTAIL_MILLET = "Foxtail millet"
    CHINABERRYTREE = "Chinaberrytree"
    PEA = "Pea"
    ZUCCHINI = "Zucchini"
    SORGHUM = "Sorghum"
    FAVA_BEAN = "Fava bean"
    CLUSTER_BEAN = "Cluster bean"
    DILL = "Dill"
    DACTYLADENIA_BARTERI = "Dactyladenia barteri"
    BAMBARA_GROUNDNUT = "Bambara groundnut"
    WOMANS_TONGUE = "Woman's tongue"
    CABBAGE = "Cabbage"
    MUSTARD = "Mustard"
    INDIAN_JUJUBE = "Indian jujube"
    PUMPKIN = "Pumpkin"


class ClimateZones(BaseEnum):
    # From Koppen-Geiger classification
    TROPICAL_RAINFOREST = "Af"
    TROPICAL_MONSOON = "Am"
    TROPICAL_SAVANNA = "As"
    HOT_DESERT = "BWh"
    COLD_DESERT = "BWk"
    HOT_STEPPE = "BSh"
    COLD_STEPPE = "BSk"
    MEDITERRANEAN_HOT_SUMMER = "Csa"
    MEDITERRANEAN_WARM_SUMMER = "Csb"
    HUMID_SUBTROPICAL_CWA = "Cwa"  # Added suffix to differentiate from Cfa
    SUBTROPICAL_HIGHLAND = "Cwb"
    HUMID_SUBTROPICAL_CFA = "Cfa"  # Added suffix to differentiate from Cwa
    OCEANIC = "Cfb"
    HUMID_CONTINENTAL_CLIMATE = 'Dwa'
    HOT_SUMMER_HUMID_CONTINENTAL = "Dfa"
    WARM_SUMMER_HUMID_CONTINENTAL = "Dfb"
    SUBARCTIC = "Dfc"
    TUNDRA = "ET"
    ICE_CAP = "EF"
    NA = "NA"

    @classmethod
    def from_value(cls: Type[Enum], value: Any):
        if value == 'Aw':
            value = 'As'
        if value == 'Dsb':
            value = "Dwa"
        return super().from_value(value)


class SoilTextures(BaseEnum):
    # USDA soil texture classifications
    CLAY = "Clay"
    SILTY_CLAY = "Silty clay"
    SANDY_CLAY = "Sandy clay"
    CLAY_LOAM = "Clay loam"
    SILTY_CLAY_LOAM = "Silty clay loam"
    SANDY_CLAY_LOAM = "Sandy clay loam"
    LOAM = "Loam"
    SILT_LOAM = "Silty loam"
    SANDY_LOAM = "Sandy loam"
    SILT = "Silt"
    LOAMY_SAND = "Loamy sand"
    SAND = "Sand"
    MULTIPLE_SOIL = "Multiple soil"
    NA = "NA"


class IntercroppingDesigns(BaseEnum):
    REPLACEMENT = "Replacement"
    ADDITIVE = "Additive"
    NA = "NA"


class IntercroppingPatterns(BaseEnum):
    STRIP = "Strip"
    ROW = "Row"
    MIXED = "Mixed"
    AGROFORESTRY = "AF"
    NA = "NA"


E = TypeVar('E', bound=Enum)


def get_one_hot_encoding(value: E, enum: Type[E]) -> List[bool]:
    """
    Convert an Enum value to a one-hot encoded boolean list.

    :param value: The enum value to encode
    :param enum: The enum class containing all possible values
    :return: A list of boolean values where True indicates the position of the input value in the enum

    Example:
    ```python
    class Colors(Enum):
        RED = "red"
        BLUE = "blue"
        GREEN = "green"

    result = get_one_hot_encoding(Colors.BLUE, Colors)
    # Returns: [False, True, False]
    ```
    """
    return [v == value for v in enum]


@dataclass
class TensorConvertible(ABC):
    """
    Abstract base class for objects that can be converted to PyTorch tensors.

    This class defines a common interface for converting dataclass instances
    to tensor representations, ensuring consistent tensor conversion across
    different data types in the system.
    """

    @abstractmethod
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert the object's data to a PyTorch tensor.

        :param device: The device to place the tensor on (CPU/GPU)
        :return: A tensor representation of the object's data
        :raises NotImplementedError: If the subclass doesn't implement this method
        """
        pass


@dataclass
class ExperimentalSite(TensorConvertible):
    climate_zone: ClimateZones
    soil_texture: SoilTextures
    soil_ph: float  # If -1.0, it means "NA"

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert site data to tensor representation.

        :param device: The device to place the tensor on (CPU/GPU)
        :return: Tensor containing encoded climate zone, soil texture, and pH
        """
        climate_zone_encoding = get_one_hot_encoding(self.climate_zone, ClimateZones)
        soil_texture_encoding = get_one_hot_encoding(self.soil_texture, SoilTextures)
        encoding = climate_zone_encoding + soil_texture_encoding + [self.soil_ph]
        return torch.tensor(encoding, device=device)


@dataclass
class IntercroppingDescriptors(TensorConvertible):
    design: IntercroppingDesigns
    pattern: IntercroppingPatterns

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert intercropping descriptors to tensor representation.

        :param device: The device to place the tensor on (CPU/GPU)
        :return: Tensor containing encoded design and pattern information
        """
        design_encoding = get_one_hot_encoding(self.design, IntercroppingDesigns)
        pattern_encoding = get_one_hot_encoding(self.pattern, IntercroppingPatterns)
        encoding = design_encoding + pattern_encoding
        return torch.tensor(encoding, device=device)


@dataclass
class CropManagementPractices(TensorConvertible):
    nitrogen_rate: float  # -1.0 for "NA"
    greenhouse: bool = False
    organic_fertilizer: bool = False
    mineral_fertilizer: bool = False
    pesticide: bool = True
    irrigation: bool = True

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert crop management practices to tensor representation.

        :param device: The device to place the tensor on (CPU/GPU)
        :return: Tensor containing encoded management practices
        """
        encoding = [getattr(self, value.name) for value in fields(self) if value.name != 'nitrogen_rate']
        encoding.append(self.nitrogen_rate)
        return torch.tensor(encoding, device=device)

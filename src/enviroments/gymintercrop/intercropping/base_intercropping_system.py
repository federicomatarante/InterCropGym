from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from .intercropping_parameters import IntercroppingParameters
from ..utils.crop_metrics import CropMetrics
from ..utils.simulation_parameters import SimulationParameters


class BaseIntercroppingSystem(ABC):
    """
    Abstract base class for implementing intercropping systems.

    This class defines the interface for modeling interactions between crops
    in an intercropping system. Subclasses must implement the abstract methods
    to define specific interaction behaviors.
    """

    def __init__(self, params: IntercroppingParameters = None):
        """
        Initialize the intercropping system with configurable parameters.

        Args:
            params: Configuration parameters for interactions. If None, uses defaults.
        """
        self.params = params or IntercroppingParameters()

    @abstractmethod
    def _calculate_light_competition(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        """
        Calculate effects of light competition between crops.

        Considers:
        - Leaf Area Index (LAI) shadowing effects
        - Canopy structure differences
        - Light interception efficiency

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Dictionary containing competition effects and additional data.
            - effect_on_1: [0.0, 1.0] where 1.0 means no shading effect
            - effect_on_2: [0.0, 1.0] where 1.0 means no shading effect
            - taller_crop_factor: [-1.0, 1.0] where positive values indicate crop 1 is taller
        """
        pass

    @abstractmethod
    def _calculate_water_competition(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, float]:
        """
        Calculate water resource competition effects.

        Considers:
        - Transpiration rates
        - Root water uptake
        - Soil water content
        - Water stress impacts

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Dictionary containing competition effects and additional data
            - effect_on_1: [0.0, 1.0] where 1.0 means no water competition
            - effect_on_2: [0.0, 1.0] where 1.0 means no water competition
            - water_comp_1: [0.0, 1.0] representing crop 1's share of water resources
            - water_comp_2: [0.0, 1.0] representing crop 2's share of water resources
        """
        pass

    @abstractmethod
    def _calculate_nitrogen_interaction(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, float]:
        """
        Calculate nitrogen transfer and competition effects.

        Considers:
        - N-fixing capabilities (for legumes)
        - Soil nitrogen content
        - N uptake rates
        - N transfer between crops

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Dictionary containing interaction effects and additional data
            - effect_on_1: [0.0, 1.0] where 1.0 means no negative N interaction
            - effect_on_2: [0.0, 1.0] where 1.0 means no negative N interaction
            - transfer_1to2: [0.0, ∞) amount of N transferred from crop 1 to 2
            - transfer_2to1: [0.0, ∞) amount of N transferred from crop 2 to 1
            - n_ratio_1: [0.0, ∞) ratio of N uptake to soil N for crop 1
            - n_ratio_2: [0.0, ∞) ratio of N uptake to soil N for crop 2
        """
        pass

    @abstractmethod
    def _calculate_root_interaction(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        """
        Calculate root system interaction effects.

        Considers:
        - Root depth profiles
        - Root density and distribution
        - Root competition or complementarity
        - Mycorrhizal networks

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Dictionary containing interaction effects and additional data
            - effect_on_1: [0.0, 1.0] where 1.0 means no negative root interaction
            - effect_on_2: [0.0, 1.0] where 1.0 means no negative root interaction
            - overlap_factor: [0.0, 1.0] degree of root zone overlap
            - density_effect_1: [0.0, 1.0] crop 1's share of root space
            - density_effect_2: [0.0, 1.0] crop 2's share of root space
            Note: density_effect_1 + density_effect_2 = 1.0
        """
        pass

    @abstractmethod
    def _calculate_biomass_effects(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        """
        Calculate overall biomass and growth effects.

        Considers:
        - Space competition
        - Resource allocation
        - Growth rates
        - Yield components

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Dictionary containing biomass effects and additional data
            - effect_on_1: [0.0, 1.0] where 1.0 means no growth reduction
            - effect_on_2: [0.0, 1.0] where 1.0 means no growth reduction
            - biomass_ratio_1: [0.0, 1.0] crop 1's share of total biomass
            - biomass_ratio_2: [0.0, 1.0] crop 2's share of total biomass
            Note: biomass_ratio_1 + biomass_ratio_2 = 1.0
        """
        pass

    def calculate_intercropping_effects(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Tuple[CropMetrics, CropMetrics, SimulationParameters, SimulationParameters]:
        """
        Calculate and combine all intercropping interaction effects.

        This method orchestrates the calculation of individual effects and
        combines them to produce the final modified states and metrics.

        :param metrics_1: Performance metrics for first crop
        :param metrics_2: Performance metrics for second crop
        :param state_1: State parameters for first crop
        :param state_2: State parameters for second crop
        :return: Tuple containing (new_metrics_1, new_metrics_2, new_state_1, new_state_2)
                representing modified metrics and states for both crops after interactions
        """
        # Calculate individual effects
        light_effects = self._calculate_light_competition(metrics_1, metrics_2, state_1, state_2)
        water_effects = self._calculate_water_competition(metrics_1, metrics_2, state_1, state_2)
        nitrogen_effects = self._calculate_nitrogen_interaction(metrics_1, metrics_2, state_1, state_2)
        root_effects = self._calculate_root_interaction(metrics_1, metrics_2, state_1, state_2)
        biomass_effects = self._calculate_biomass_effects(metrics_1, metrics_2, state_1, state_2)

        # Create new instances for modification
        new_metrics_1 = CropMetrics(**{k: v for k, v in metrics_1.__dict__.items()})
        new_metrics_2 = CropMetrics(**{k: v for k, v in metrics_2.__dict__.items()})
        new_state_1 = SimulationParameters(**{k: v for k, v in state_1.__dict__.items()})
        new_state_2 = SimulationParameters(**{k: v for k, v in state_2.__dict__.items()})

        # Implement combination of effects in subclass
        self._combine_effects(
            new_metrics_1, new_metrics_2, new_state_1, new_state_2,
            light_effects, water_effects, nitrogen_effects, root_effects, biomass_effects
        )

        return new_metrics_1, new_metrics_2, new_state_1, new_state_2

    @abstractmethod
    def _combine_effects(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters,
            light_effects: Dict[str, Any],
            water_effects: Dict[str, Any],
            nitrogen_effects: Dict[str, Any],
            root_effects: Dict[str, Any],
            biomass_effects: Dict[str, Any]
    ) -> None:
        """
        Combine all calculated effects to modify the crop metrics and states.

        This method defines how different interaction effects are combined and
        applied to modify the crops' states and metrics in-place.

        :param metrics_1: Performance metrics for first crop to be modified
        :param metrics_2: Performance metrics for second crop to be modified
        :param state_1: State parameters for first crop to be modified
        :param state_2: State parameters for second crop to be modified
        :param light_effects: Results from light competition calculations
        :param water_effects: Results from water competition calculations
        :param nitrogen_effects: Results from nitrogen interaction calculations
        :param root_effects: Results from root interaction calculations
        :param biomass_effects: Results from biomass effects calculations
        """
        pass

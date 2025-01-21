from typing import Dict, Any

import numpy as np

from .base_intercropping_system import BaseIntercroppingSystem
from ..utils.crop_metrics import CropMetrics
from ..utils.simulation_parameters import SimulationParameters


class StandardIntercroppingSystem(BaseIntercroppingSystem):
    """
    Standard implementation of intercropping system with common interaction patterns
    between crops, such as competition for light, water, and nutrients, as well as
    potential beneficial interactions.
    """

    def _calculate_light_competition(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, float]:
        # Calculate relative canopy positions based on stem weight
        taller_crop_factor = np.tanh(state_1.WST / (state_2.WST + 1e-6))

        # Calculate shading effects using sigmoid function for smooth transition
        shading_on_2 = self.params.light_competition_factor * (
                1 / (1 + np.exp(-2 * (metrics_1.lai - metrics_2.lai)))
        ) * taller_crop_factor

        shading_on_1 = self.params.light_competition_factor * (
                1 / (1 + np.exp(-2 * (metrics_2.lai - metrics_1.lai)))
        ) * (1 - taller_crop_factor)

        return {
            'effect_on_1': 1.0 - shading_on_1,
            'effect_on_2': 1.0 - shading_on_2,
            'taller_crop_factor': taller_crop_factor
        }

    def _calculate_water_competition(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, float]:
        # Calculate total water demand
        total_transpiration = metrics_1.tran + metrics_2.tran

        # Calculate root competition based on root depth and mass
        root_factor_1 = state_1.ROOTD * state_1.WRT
        root_factor_2 = state_2.ROOTD * state_2.WRT
        total_root_factor = root_factor_1 + root_factor_2

        # Combine transpiration and root effects
        if total_transpiration > 0 and total_root_factor > 0:
            water_comp_1 = (
                    (metrics_1.tran / total_transpiration) *
                    (root_factor_1 / total_root_factor)
            )
            water_comp_2 = (
                    (metrics_2.tran / total_transpiration) *
                    (root_factor_2 / total_root_factor)
            )
        else:
            water_comp_1 = water_comp_2 = 0.5

        # Calculate water stress effects
        water_effect_1 = 1.0 - (self.params.water_competition_factor * (1 - water_comp_1))
        water_effect_2 = 1.0 - (self.params.water_competition_factor * (1 - water_comp_2))

        return {
            'effect_on_1': water_effect_1,
            'effect_on_2': water_effect_2,
            'water_comp_1': water_comp_1,
            'water_comp_2': water_comp_2
        }

    def _calculate_nitrogen_interaction(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        # Calculate N availability ratio for each crop
        n_ratio_1 = state_1.NUPTT / (state_1.TNSOIL + 1e-6)
        n_ratio_2 = state_2.NUPTT / (state_2.TNSOIL + 1e-6)

        # Calculate potential N transfer (simulating legume-nonlegume interactions)
        n_transfer_1to2 = max(0, (n_ratio_1 - n_ratio_2) * self.params.nitrogen_transfer_factor)
        n_transfer_2to1 = max(0, (n_ratio_2 - n_ratio_1) * self.params.nitrogen_transfer_factor)

        # Calculate competition effects
        n_effect_1 = 1.0 - (self.params.nitrogen_transfer_factor * n_transfer_1to2)
        n_effect_2 = 1.0 - (self.params.nitrogen_transfer_factor * n_transfer_2to1)

        return {
            'effect_on_1': n_effect_1,
            'effect_on_2': n_effect_2,
            'transfer_1to2': n_transfer_1to2,
            'transfer_2to1': n_transfer_2to1,
            'n_ratio_1': n_ratio_1,
            'n_ratio_2': n_ratio_2
        }

    def _calculate_root_interaction(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        # Calculate root zone overlap
        min_depth = min(state_1.ROOTD, state_2.ROOTD)
        max_depth = max(state_1.ROOTD, state_2.ROOTD)
        overlap_factor = min_depth / max_depth if max_depth > 0 else 1.0

        # Calculate root density effects
        root_density_1 = state_1.WRT / (state_1.ROOTD + 1e-6)
        root_density_2 = state_2.WRT / (state_2.ROOTD + 1e-6)
        total_density = root_density_1 + root_density_2

        if total_density > 0:
            density_effect_1 = root_density_1 / total_density
            density_effect_2 = root_density_2 / total_density
        else:
            density_effect_1 = density_effect_2 = 0.5

        # Combine effects
        root_effect_1 = 1.0 - (self.params.root_interaction_factor *
                               overlap_factor * (1 - density_effect_1))
        root_effect_2 = 1.0 - (self.params.root_interaction_factor *
                               overlap_factor * (1 - density_effect_2))

        return {
            'effect_on_1': root_effect_1,
            'effect_on_2': root_effect_2,
            'overlap_factor': overlap_factor,
            'density_effect_1': density_effect_1,
            'density_effect_2': density_effect_2
        }

    def _calculate_biomass_effects(
            self,
            metrics_1: CropMetrics,
            metrics_2: CropMetrics,
            state_1: SimulationParameters,
            state_2: SimulationParameters
    ) -> Dict[str, Any]:
        # Calculate relative biomass proportions
        total_biomass = metrics_1.tagbm + metrics_2.tagbm
        if total_biomass > 0:
            biomass_ratio_1 = metrics_1.tagbm / total_biomass
            biomass_ratio_2 = metrics_2.tagbm / total_biomass
        else:
            biomass_ratio_1 = biomass_ratio_2 = 0.5

        # Calculate growth rate effects
        growth_effect_1 = 1.0 - (self.params.biomass_competition_factor *
                                 (1 - biomass_ratio_1))
        growth_effect_2 = 1.0 - (self.params.biomass_competition_factor *
                                 (1 - biomass_ratio_2))

        return {
            'effect_on_1': growth_effect_1,
            'effect_on_2': growth_effect_2,
            'biomass_ratio_1': biomass_ratio_1,
            'biomass_ratio_2': biomass_ratio_2
        }

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
        # Calculate combined growth effects
        growth_effect_1 = np.mean([
            light_effects['effect_on_1'],
            water_effects['effect_on_1'],
            nitrogen_effects['effect_on_1'],
            root_effects['effect_on_1'],
            biomass_effects['effect_on_1']
        ])
        growth_effect_2 = np.mean([
            light_effects['effect_on_2'],
            water_effects['effect_on_2'],
            nitrogen_effects['effect_on_2'],
            root_effects['effect_on_2'],
            biomass_effects['effect_on_2']
        ])

        # Apply growth effects to biomass-related metrics
        biomass_vars = ['wlvg', 'wrt', 'wso', 'wst', 'tagbm']
        for var in biomass_vars:
            setattr(metrics_1, var, getattr(metrics_1, var) * growth_effect_1)
            setattr(metrics_2, var, getattr(metrics_2, var) * growth_effect_2)

        # Apply specific effects to related variables
        # Light effects
        metrics_1.tgrowth *= light_effects['effect_on_1']
        metrics_2.tgrowth *= light_effects['effect_on_2']

        # Water effects
        metrics_1.tranrf *= water_effects['effect_on_1']
        metrics_2.tranrf *= water_effects['effect_on_2']

        # Nitrogen effects
        state_1.TNSOIL = (state_1.TNSOIL * nitrogen_effects['effect_on_1'] +
                          nitrogen_effects['transfer_2to1'])
        state_2.TNSOIL = (state_2.TNSOIL * nitrogen_effects['effect_on_2'] +
                          nitrogen_effects['transfer_1to2'])

        # Root effects
        metrics_1.nuptt *= root_effects['effect_on_1']
        metrics_2.nuptt *= root_effects['effect_on_2']

        # Update nitrogen-related state variables
        n_scaling_1 = nitrogen_effects['effect_on_1']
        n_scaling_2 = nitrogen_effects['effect_on_2']

        state_1.NUPTT *= n_scaling_1
        state_2.NUPTT *= n_scaling_2

        # Update NNI based on modified nitrogen status
        state_1.NNI = max(0.0, min(1.0, state_1.NNI * n_scaling_1))
        state_2.NNI = max(0.0, min(1.0, state_2.NNI * n_scaling_2))

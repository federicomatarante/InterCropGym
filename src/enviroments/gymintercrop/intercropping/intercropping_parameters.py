from dataclasses import dataclass


@dataclass
class IntercroppingParameters:
    """Parameters that control intercropping interactions between crops"""
    light_competition_factor: float = 0.15  # How much LAI of one crop affects the other
    water_competition_factor: float = 0.1  # Competition for water resources
    nitrogen_transfer_factor: float = 0.05  # N transfer between crops (e.g., legume to non-legume)
    root_interaction_factor: float = 0.08  # How much root systems interact
    biomass_competition_factor: float = 0.12  # Competition for space and resources


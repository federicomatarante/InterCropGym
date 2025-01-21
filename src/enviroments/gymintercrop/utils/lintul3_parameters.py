from dataclasses import dataclass


@dataclass
class LINTUL3Parameters:
    """
    Dataclass representing crop state variables in the LINTUL3 crop model.
    The model must implement a set_variable(name, value) and a get_variable(name) method that accepts the following values:
        - ANLV: Nitrogen in leaves (g/m²)
        - ANRT: Nitrogen in roots (g/m²)
        - ANSO: Nitrogen in storage organs (g/m²)
        - ANST: Nitrogen in stems (g/m²)
        - CUMPAR: Cumulative Photosynthetically Active Radiation (MJ/m²)
        - LAI: Leaf Area Index (m²/m²)
        - NLOSSL: Nitrogen loss from leaves (g/m²)
        - NLOSSR: Nitrogen loss from roots (g/m²)
        - NNI: Nitrogen Nutrition Index
        - NUPTT: Total nitrogen uptake (g/m²)
        - ROOTD: Root depth (m)
        - TAGBM: Total Above Ground Biomass (g/m²)
        - TGROWTH: Total growth (g/m²)
        - TNSOIL: Total nitrogen in soil (g/m²)
        - WDRT: Weight of dead roots (g/m²)
        - WLVD: Weight of dead leaves (g/m²)
        - WLVG: Weight of green leaves (g/m²)
        - WRT: Total root weight (g/m²)
        - WSO: Weight of storage organs (g/m²)
        - WST: Weight of stems (g/m²)
    """
    ANLV: float  # Nitrogen in leaves (g/m²)
    ANRT: float  # Nitrogen in roots (g/m²)
    ANSO: float  # Nitrogen in storage organs (g/m²)
    ANST: float  # Nitrogen in stems (g/m²)
    CUMPAR: float  # Cumulative Photosynthetically Active Radiation (MJ/m²)
    LAI: float  # Leaf Area Index, leaf area per unit of soil area (m²/m²)
    NLOSSL: float  # Nitrogen loss from leaves (g/m²)
    NLOSSR: float  # Nitrogen loss from roots (g/m²)
    NNI: float  # Nitrogen Nutrition Index, indicates nitrogen sufficiency
    NUPTT: float  # Total nitrogen uptake by the crop (g/m²)
    ROOTD: float  # Root depth, penetration depth of roots (m)
    TAGBM: float  # Total Above Ground Biomass, includes leaves, stems, and storage organs (g/m²)
    TGROWTH: float  # Total growth, cumulative over time (g/m²)
    TNSOIL: float  # Total nitrogen available in soil (g/m²)
    WDRT: float  # Weight of dead roots (g/m²)
    WLVD: float  # Weight of dead leaves (g/m²)
    WLVG: float  # Weight of green leaves (g/m²)
    WRT: float  # Total weight of roots (g/m²)
    WSO: float  # Weight of storage organs (g/m²)
    WST: float  # Weight of stems (g/m²)

    @staticmethod
    def from_model(model):
        """
        Create a SimulationParameters instance from a model's crop state.

        :param model: The crop model to extract the crop state from. See class documentation for more.
        :return: An instance of SimulationParameters
        """
        return LINTUL3Parameters(
            ANLV=model.get_variable('ANLV'),
            ANRT=model.get_variable('ANRT'),
            ANSO=model.get_variable('ANSO'),
            ANST=model.get_variable('ANST'),
            CUMPAR=model.get_variable('CUMPAR'),
            LAI=model.get_variable('LAI'),
            NLOSSL=model.get_variable('NLOSSL'),
            NLOSSR=model.get_variable('NLOSSR'),
            NNI=model.get_variable('NNI'),
            NUPTT=model.get_variable('NUPTT'),
            ROOTD=model.get_variable('ROOTD'),
            TAGBM=model.get_variable('TAGBM'),
            TGROWTH=model.get_variable('TGROWTH'),
            TNSOIL=model.get_variable('TNSOIL'),
            WDRT=model.get_variable('WDRT'),
            WLVD=model.get_variable('WLVD'),
            WLVG=model.get_variable('WLVG'),
            WRT=model.get_variable('WRT'),
            WSO=model.get_variable('WSO'),
            WST=model.get_variable('WST')
        )

    def update_model(self, model):
        """
        Update the crop model's states with the values from the SimulationParameters instance.
        :param model: The crop model to update. See class documentation for more.
        """
        # The model has a set_variable method as we can see in the dir() output
        model.set_variable('ANLV', self.ANLV)
        model.set_variable('ANRT', self.ANRT)
        model.set_variable('ANSO', self.ANSO)
        model.set_variable('ANST', self.ANST)
        model.set_variable('CUMPAR', self.CUMPAR)
        model.set_variable('LAI', self.LAI)
        model.set_variable('NLOSSL', self.NLOSSL)
        model.set_variable('NLOSSR', self.NLOSSR)
        model.set_variable('NNI', self.NNI)
        model.set_variable('NUPTT', self.NUPTT)
        model.set_variable('ROOTD', self.ROOTD)
        model.set_variable('TAGBM', self.TAGBM)
        model.set_variable('TGROWTH', self.TGROWTH)
        model.set_variable('TNSOIL', self.TNSOIL)
        model.set_variable('WDRT', self.WDRT)
        model.set_variable('WLVD', self.WLVD)
        model.set_variable('WLVG', self.WLVG)
        model.set_variable('WRT', self.WRT)
        model.set_variable('WSO', self.WSO)
        model.set_variable('WST', self.WST)

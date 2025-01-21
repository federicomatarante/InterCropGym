import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class CropState:
    """
    Dataclass representing various crop growth and state.
    The dataframes used to interact with the CropState must contain the following keys:
    columns (case sensitive):
        LAI: Leaf Area Index (m² leaf/m² soil)
        WLVD: Weight of Dead Leaves
        WLVG: Weight of Green Leaves
        WRT: Weight of Roots
        WSO: Weight of Storage Organs
        WST: Weight of Stems
        TAGBM: Total Above Ground BioMass
        TGROWTH: Total Growth rate
        NUPTT: Nitrogen UPTake Total
        TRAN: TRANspiration rate
        TIRRIG: Total IRRIGation
        TNSOIL: Total Nitrogen in SOIL
        TRAIN: Total RAINfall
        TRANRF: TRANspiration Reduction Factor
        TRUNOF: Total RUNOFF
        TTRAN: Total TRANspiration
        WC: Water Content
    Also each row is indexed with a pd.TimeStamp object.
    """
    lai: float  # Leaf Area Index (m² leaf/m² soil)
    wlvd: float  # Weight of dead leaves
    wlvg: float  # Weight of green leaves
    wrt: float  # Weight of roots
    wso: float  # Weight of storage organs
    wst: float  # Weight of stems
    tagbm: float  # Total Above Ground BioMass
    tgrowth: float  # Total Growth rate
    nuptt: float  # Nitrogen UPTake Total
    tran: float  # TRANspiration rate
    tirrig: float  # Total IRRIGation
    tnsoil: float  # Total Nitrogen in SOIL
    train: float  # Total RAINfall
    tranrf: float  # TRANspiration Reduction Factor
    trunof: float  # Total RUNOFF
    ttran: float  # Total TRANspiration
    wc: float  # Water Content
    dvs: float  # Development Stage
    date: pd.Timestamp

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> 'CropState':
        """
        Create a CropMetrics instance from a pandas DataFrame's last column.
        :param df: dataframe to extract data from. See class documentation for more info.
        """
        # Create a mapping of lowercase column names to actual column names
        col_map = {col.lower(): col for col in df.columns}

        # List of required attributes in lowercase
        required_attrs = [attr.lower() for attr in cls.__dataclass_fields__.keys() if
                          attr not in ['date']]

        # Check for missing columns
        missing_cols = [attr for attr in required_attrs if attr not in col_map]
        if missing_cols:
            raise ValueError(f"Expected DataFrame with columns {required_attrs}, missing {missing_cols}")

        # Create kwargs dictionary with values from the DataFrame
        kwargs = {
            attr: float(df[col_map[attr.lower()]].iloc[-1])
            for attr in cls.__dataclass_fields__.keys() if attr not in ['date']
        }

        return cls(**kwargs, date=df.index[-1])

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert the CropMetrics instance to a pandas DataFrame.
        Returns a DataFrame with a single row containing all metrics and their descriptions.
        See class documentation for more info.
        """
        params = {
            'LAI': self.lai,
            'WLVD': self.wlvd,
            'WLVG': self.wlvg,
            'WRT': self.wrt,
            'WSO': self.wso,
            'WST': self.wst,
            'TAGBM': self.tagbm,
            'TGROWTH': self.tgrowth,
            'NUPTT': self.nuptt,
            'TRAN': self.tran,
            'TIRRIG': self.tirrig,
            'DVS': self.dvs,
            'TNSOIL': self.tnsoil,
            'TRAIN': self.train,
            'TRANRF': self.tranrf,
            'TRUNOF': self.trunof,
            'TTRAN': self.ttran,
            'WC': self.wc
        }
        return pd.DataFrame([
            params
        ], index=[self.date])

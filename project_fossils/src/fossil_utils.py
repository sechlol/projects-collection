import math
import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.typing import NDArray

from shapely import wkt
from dataclasses import dataclass


class Interpolator:
    """
    Helper class that maps a range of numbers [from_value, to_value] into another range [lower_limit, upper_limit]
    It is useful to assign data values to a continuous range of colors, or give a size to markers on a plot
    """
    def __init__(self, from_value: float, to_value: float, lower_limit: float, upper_limit: float):
        self.from_value = from_value
        self.to_value = to_value
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def interpolate(self, value: float) -> float:
        return (value - self.lower_limit) / (self.upper_limit - self.lower_limit) * (self.to_value - self.from_value) + self.from_value


@dataclass
class TimeUnitDef:
    name: str
    min: float
    max: float
    index: int


# Lists all the time units, from the oldest to the newest
TIME_UNITS_LIST = [
    TimeUnitDef(name="pre-MN", index=1, max=999, min=23),
    TimeUnitDef(name="MN1", index=2, max=23, min=21.7),
    TimeUnitDef(name="MN2", index=3, max=21.7, min=19.5),
    TimeUnitDef(name="MN3", index=4, max=19.5, min=17.2),
    TimeUnitDef(name="MN4", index=5, max=17.2, min=16.4),
    TimeUnitDef(name="MN5", index=6, max=16.4, min=14.2),
    TimeUnitDef(name="MN6", index=7, max=14.2, min=12.85),
    TimeUnitDef(name="MN7-8", index=8, max=12.85, min=11.2),
    TimeUnitDef(name="MN9", index=9, max=11.2, min=9.9),
    TimeUnitDef(name="MN10", index=10, max=9.9, min=8.9),
    TimeUnitDef(name="MN11", index=11, max=8.9, min=7.6),
    TimeUnitDef(name="MN12", index=12, max=7.6, min=7.1),
    TimeUnitDef(name="MN13", index=13, max=7.1, min=5.3),
    TimeUnitDef(name="MN14", index=14, max=5.3, min=5),
    TimeUnitDef(name="MN15", index=15, max=5, min=3.55),
    TimeUnitDef(name="MN16", index=16, max=3.55, min=2.5),
    TimeUnitDef(name="MN17", index=17, max=2.5, min=1.9),
    TimeUnitDef(name="MQ18", index=18, max=1.9, min=0.85),
    TimeUnitDef(name="MQ19", index=19, max=0.85, min=0.01),
    TimeUnitDef(name="post-MN", index=20, max=0.01, min=0),
]

TU_MAP_BY_NAME = {tu.name: tu for tu in TIME_UNITS_LIST}
TU_MAP_BY_INDEX = {tu.index: tu for tu in TIME_UNITS_LIST}


def to_GeoDataFrame(df: pd.DataFrame) -> gpd.GeoDataFrame:
    # Creates a GeoDataFrame from the original data. GeoPandas needs to create points from Lon/lat coordinates first
    if "LONG" in df.columns and "LAT" in df.columns:
        geometries = gpd.points_from_xy(df["LONG"], df["LAT"])
    else:
        geometries = df['geometry'].apply(wkt.loads)

    # Need to set a value for CRS (Coordinate reference system) to perform a spatial join. In this case,
    # crs="EPSG:4326" to match the crs of the world dataset
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


def get_tu_names() -> list[str]:
    return [tu.name for tu in get_sorted_tu()]


def tu_name_to_index(tu_name: str) -> int:
    return TU_MAP_BY_NAME[tu_name].index


def tu_index_to_name(tu_index: int) -> str:
    return TU_MAP_BY_INDEX[tu_index].name


def get_tu_indexes() -> list[int]:
    return [tu.index for tu in get_sorted_tu()]


def get_tu_by_name(tu_name: str) -> TimeUnitDef:
    return TU_MAP_BY_NAME[tu_name]


def get_tu_by_index(tu_index: int) -> TimeUnitDef:
    return TU_MAP_BY_INDEX[tu_index]


def get_sorted_tu() -> list[TimeUnitDef]:
    return TIME_UNITS_LIST


def color_gradient(size: int) -> list[tuple[float, float, float]]:
    rg = np.linspace(0, 1, size)
    b = np.linspace(1, 0, size)
    return [(rg[i].item(), rg[i].item(), b[i].item()) for i in range(0, size)]


def sigmoid(x: float | NDArray, c0: float, c1: float) -> float | NDArray:
    # Used for plotting a sigmoid curve. It works either with numpy arrays and with single values
    return 1.0 / (1.0 + math.e ** -(c0 + c1 * x))

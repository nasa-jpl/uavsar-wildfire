#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@source: Charles Z Marshak (https://github.com/simard-landscape-lab)
"""
import numpy as np
from affine import Affine
from rasterio.features import shapes
from typing import Tuple
from rasterio.warp import (reproject,
                           Resampling)

def get_geopandas_features_from_array(arr: np.ndarray,
                                      transform: Affine,
                                      label_name: str = 'label',
                                      mask: np.ndarray = None,
                                      connectivity: int = 4) -> list:
    """
    Obtains a list of geopandas features in which contigious integers are
    grouped as polygons for use as:
        df =  gpd.GeoDataFrame.from_features(geo_features)
    Parameters
    ----------
    arr : np.ndarray
        The array of integers to group into contiguous polygons. Note some
        labels that are connected through diagonals May be separated depending
        on connectivity.
    transform : Affine
        Rasterio transform related to arr
    label_name : str
        The label name used for each different polygonal feature, default is
        `label`.
    mask : np.ndarray
        Nodata mask in which true values indicate where nodata is located.
    connectivity : int
        4- or 8- connectivity of the polygonal features.  See rasterio:
        https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.shapes
        And see: https://en.wikipedia.org/wiki/Pixel_connectivity
    Returns
    -------
    list:
        List of features to use for constructing geopandas dataframe with
        gpd.GeoDataFrame.from_features
    """
    # see rasterio.features.shapes - needs all false values to be no data areas
    if mask is None:
        mask = np.zeros(arr.shape, dtype=bool)
    feature_list = list(shapes(arr,
                               mask=~mask,
                               transform=transform,
                               connectivity=connectivity))
    geo_features = list({'properties': {label_name: (value)},
                         'geometry': geometry}
                        for i, (geometry, value) in enumerate(feature_list))
    return geo_features


def reproject_arr_to_match_profile(src_array: np.ndarray,
                                   src_profile: dict,
                                   ref_profile: dict,
                                   nodata: str = None,
                                   resampling='bilinear') \
                                           -> Tuple[np.ndarray, dict]:
    """
    Reprojects an array to match a reference profile providing the reprojected
    array and the new profile.  Simply a wrapper for rasterio.warp.reproject.

    Parameters
    ----------
    src_array : np.ndarray
        The source array to be reprojected.
    src_profile : dict
        The source profile of the `src_array`
    ref_profile : dict
        The profile that to reproject into.
    nodata : str
        The nodata value to be used in output profile. If None, the nodata from
        src_profile is used in the output profile.  See
        https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py#L13-L24.
    resampling : str
        The type of resampling to use. See all the options:
        https://github.com/mapbox/rasterio/blob/08d6634212ab131ca2a2691054108d81caa86a09/rasterio/enums.py#L28-L40

    Returns
    -------
    Tuple[np.ndarray, dict]:
        Reprojected Arr, Reprojected Profile

    Notes
    -----
    src_array needs to be in gdal (i.e. BIP) format that is (# of channels) x
    (vertical dim.) x (horizontal dim).  Also, works with arrays of the form
    (vertical dim.) x (horizontal dim), but output will be: 1 x (vertical dim.)
    x (horizontal dim).
    """
    height, width = ref_profile['height'], ref_profile['width']
    crs = ref_profile['crs']
    transform = ref_profile['transform']

    reproject_profile = ref_profile.copy()

    nodata = nodata or src_profile['nodata']
    src_dtype = src_profile['dtype']
    count = src_profile['count']

    reproject_profile.update({'dtype': src_dtype,
                              'nodata': nodata,
                              'count': count})

    dst_array = np.zeros((count, height, width))

    resampling = Resampling[resampling]

    reproject(src_array,
              dst_array,
              src_transform=src_profile['transform'],
              src_crs=src_profile['crs'],
              dst_transform=transform,
              dst_crs=crs,
              dst_nodata=nodata,
              resampling=resampling)
    return dst_array.astype(src_dtype), reproject_profile


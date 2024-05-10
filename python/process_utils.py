#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wen Tao Lin
"""
from nd_utils import (interpolate_nn, scale_img)
from rio_utils import (get_geopandas_features_from_array)
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import felzenszwalb
from shapely.geometry import Point
from osgeo import gdal
from rasterio.crs import CRS
import pandas as pd
import numpy as np
import geopandas as gpd

# Preprocessing
def fwd(img):
    return 10 * np.log10(img)


def bwd(img):
    return 10**(img / 10)


def tv_denoise(img: np.ndarray,
               weight: float) -> np.ndarray:
    img_db = fwd(img)
    img_db_tv = denoise_tv_chambolle(img_db, weight=weight)
    img_tv = bwd(img_db_tv)
    return img_tv


def preprocess_data(data: np.ndarray, 
                    interpolation: bool, 
                    weight: float = 5.0) -> np.ndarray:
    """
    Preprocess the post-RTC data for classification

    Parameters
    ----------
    data : np.ndarray
        The raw post-RTC data to be preprocessed.
    interpolation: bool
        whether or not to interpolation the missing data.
    weight : float, optional
        The weight parameter in skimage.restoration.denoise_tv_chambolle. 
        The greater the weight, the more denoising (at the expense of image accuracy).
        For more info: https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle
        The default is 5.0.
        
    Returns
    -------
    data_tv : np.ndarray 
        Preprocessed array with the same shape as input np.array.

    """
    if (interpolation == True):
        background_mask = (data == 0)
    elif (interpolation == False):
        background_mask = (data <= 0)

    data[data <= 0] = np.nan
    data_nn = interpolate_nn(data)
    np.clip(data_nn, 1e-3, 1, out=data_nn)
    data_tv = tv_denoise(data_nn, weight)
    data_tv[background_mask] = np.nan

    return data_tv


def preprocess_for_merge(data: np.ndarray) -> np.ndarray:
    """
    Preprocess the post-RTC data for merging

    Parameters
    ----------
    data : np.ndarray
        The raw post-RTC data to be preprocessed.
        
    Returns
    -------
    data_scaled : np.ndarray
        The preprocessed image

    """
    # mask out missing data before clipping
    background_mask = (data <=0)
    np.clip(data, 1e-3, 1, out=data)

    # scaling img to ensure consistency within image when merging with another image
    data_scaled = scale_img(data, 1e-3, 1)
    data_scaled[background_mask] = np.nan # set the mask to NaN

    return data_scaled

    # # Attempt to use interpolation
    # background_mask = (data == 0)
    # missing_mask = (data < 0)
    # data[background_mask] = np.nan
    # data[missing_mask] = np.nan
    # data_nn = interpolate_nn(data)
    # np.clip(data_nn, 1e-3, 1, out=data_nn)
    # data_scaled = scale_img(data_nn, 1e-3, 1)

    # data_fill = np.copy(data_scaled)
    # data_fill[background_mask] = np.nan

    # data_reg = np.copy(data_fill)
    # data_reg[missing_mask] = np.nan
    # return data_reg, data_fill

    # data_fill : np.ndarray 
    #     Preprocessed array with the same shape as input np.array for perimeter generation
    # data_reg : np.ndarray 
    #     Preprocessed array with the same shape as input np.array for burn severity generation


def inc_filter(img: np.ndarray):
    """
    Convert incidence angles from radian to degree and filter out invalid radian.

    Parameters
    ----------
    img : np.ndarray
        The inc image to be converted
        
    Returns
    -------
    img_deg : np.ndarray 
        Incidence angle in degrees with the 'bad' pixels filled with 999

    """
    img_deg = np.rad2deg(img)
    mask = (img_deg <=0) # note bad incidence angle will be masked when merging
    img_deg[mask] = 999
    return img_deg


def weighted_inc_merge(img_0: np.ndarray, 
                       img_1: np.ndarray, 
                       inc_0: np.ndarray, 
                       inc_1: np.ndarray):
    """
    Merge two UAVSAR flights over the same area using the weighted average based on incidence angle.
    Used to merge flights with opposite direction.

    Parameters
    ----------
    img_0 : np.ndarray
        First backscatter image 
    img_1 : np.ndarray
        Second backscatter image 
    inc_0 : np.ndarray
        Incidence angle values of the first image
    inc_1 : np.ndarray
        Incidence angle values of the second image

    Returns
    -------
    img_merged : np.ndarray 
        The merged image of the two flight directions

    """
    # Mask out missing backscatter values and bad incidence angles
    mask_img_0 = np.isnan(img_0) | (inc_0 < 20) | (inc_0 > 65)
    mask_img_1 = np.isnan(img_1) | (inc_1 < 20) | (inc_1 > 65)

    # Set the weight
    w0 = inc_1 / (inc_0 + inc_1)
    w1 = inc_0 / (inc_0 + inc_1)

    # Peforms merging
    img_merged = (img_0 * w0) + (img_1 * w1)
    img_merged = np.where(mask_img_0, img_1, img_merged)
    img_merged = np.where(mask_img_1, img_0, img_merged)
    
    return img_merged


def merge_by_lowest_inc(img_0, img_1, inc_0, inc_1):
    """
    Merge two UAVSAR flights over the same area based on the lowest incidence angle
    Used to merge flights with opposite direction.

    Parameters
    ----------
    img_0 : np.ndarray
        First backscatter image 
    img_1 : np.ndarray
        Second backscatter image 
    inc_0 : np.ndarray
        Incidence angle values of the first image
    inc_1 : np.ndarray
        Incidence angle values of the second image

    Returns
    -------
    img_merged : np.ndarray 
        The merged image of the two flight directions

    """
    # Mask out missing backscatter values
    mask_img_0 = np.isnan(img_0)
    mask_img_1 = np.isnan(img_1)

    # Compare incidence angles and merge
    img_merge = img_0 * (inc_0 < inc_1).astype(int) + img_1 * (inc_0 >= inc_1).astype(int)
    img_merge = np.where(mask_img_0, img_1, img_merge)
    img_merge = np.where(mask_img_1, img_0, img_merge)

    return img_merge
    

def superpixel_segmentation(hv_0: np.ndarray,
                            hv_1: np.ndarray,
                            min_size: int=500) -> np.ndarray:
    """
    Perform graph-based image segmentation using Felzenszwalb’s Algorithm implemented on scikit-learn

    Parameters
    ----------
    hv_0 : np.ndarray
        An (m x n) array, containing a single band pre-fire image.
    hv_1 : np.ndarray
        An (m x n) array, containing a single band pre-fire image.
    min_size : int, optional
        Minimum component for Felzenszwalb's algorithm. Enforced using postprocessing. 
        For more info: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
        The default is 500.

    Returns
    -------
    superpixel_labels : np.ndarray
        A 2D (A x B) matrix of labels such that each pixel has a unique integer label.

    """
    X = np.stack([hv_0, hv_1], axis=2)
    mask = np.isnan(X)
    X[mask] = -999
    superpixel_labels = felzenszwalb(X, 
                                     scale=1, 
                                     sigma=0, 
                                     min_size=min_size)
    return superpixel_labels


# Modeling
def get_superpixel_model_labels(data: np.ndarray, 
                                model) -> np.ndarray:
    """
    Classify the raster image using the features extracted from superpixel segmentation.
    This function allows for clustering with NaN values in the input array by masking.

    Parameters
    ----------
    data : np.ndarray
        A (m x n) array representing the features extracted from superpixel segmentation. 
        `n` should be equal to the number of features.
    model : sklearn.cluster
        A scikit-learn clustering algorithm.

    Returns
    -------
    labeled_data : np.ndarray
        A (m x 1) array storing the clustered label.
        Note: the labels are sorted in ascending order based on the value 
        of the cluster center for the first extracted feature.
        In the case of n_clusters=2 for burned area classification, label '1' represents fire area.

    Notes
    -------
    Use np.hstack((extracted_feat_1, extracted_feat_2, ...)) to create the data parameter.
    
    """
    # Create a mask for NaN values and cluster using non-NaN values
    mask = np.isnan(data).any(axis=1)
    masked_data = data[~mask]
    model.fit(masked_data)

    # Get cluster centroids and their values
    centroids = model.cluster_centers_
    centroid_values = centroids[:, 0]

    # Sort centroids' values and create mapping
    sorted_indices = np.argsort(centroid_values)
    class_mapping = {sorted_indices[i]: i for i in range(model.n_clusters)}

    # Assign class labels to original cluster labels
    original_labels = model.labels_
    updated_labels = np.array([class_mapping[cluster_label] for cluster_label in original_labels])

    # Put the clustered labels into a (m x 1) array
    mask_reshape = mask.reshape(-1,1)
    labeled_data = np.empty(mask_reshape.shape)
    labeled_data[:] = np.nan
    labeled_data[~mask_reshape] = updated_labels

    return labeled_data


def get_model_labels(data: np.ndarray, 
                     model) -> np.ndarray:
    """
    Classify the raster image using the backscatter values read from a single band.
    The function allows for clustering with NaN values in the input array by masking.

    Parameters
    ----------
    data : np.ndarray
        A (m x n) matrix where each cell corresponds to a pixel in the image.
    model : sklearn.cluster
        A scikit-learn clustering algorithm.

    Returns
    -------
    final_data : np.ndarray
        A (m x n) array similiar to `data` except each cell is filled with the classified label.
    
    Notes
    -------
    The output labels are sorted in ascending order based on the value of the
    cluster center for the first extracted feature.

    """
    # Create a mask for NaN values and cluster using non-NaN values
    mask = np.isnan(data)
    masked_data = data[~mask]
    masked_data = masked_data.reshape(-1, 1)
    model.fit(masked_data)

    # Get cluster centroids and their values
    centroids = model.cluster_centers_
    centroid_values = centroids[:, 0]

    # Sort centroids' values and create mapping
    sorted_indices = np.argsort(centroid_values)
    class_mapping = {sorted_indices[i]: i for i in range(model.n_clusters)}

    # Assign class labels to original cluster labels
    original_labels = model.labels_
    updated_labels = np.array([class_mapping[cluster_label] for cluster_label in original_labels])

    # Put the clustered labels into a (m x n) array corresponding to the input array shape
    mask_reshape = mask.reshape(-1,1)
    labeled_data = np.empty(mask_reshape.shape)
    labeled_data[:] = np.nan
    labeled_data[~mask_reshape] = updated_labels
    final_data = labeled_data.reshape(mask.shape)

    return final_data


# Post Processing
def convert_labels_to_gdf(classes: np.ndarray, 
                          profile: dict) -> gpd.GeoDataFrame:
    """
    Group an array of contigious integers as polygons and output the polygons as a GeoDataFrame.
    
    Parameters
    ----------
    classes : np.ndarray
        The array of integers to group into contiguous polygons.
    profile : dict
        Rasterio profile related to classes.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame contining the contigious integers as polygons.

    """
    # convert dtype to uint8 for geopandas features
    nan_mask = np.isnan(classes)
    classes[nan_mask] = 255 # np.nan cannot be converted to int
    classes = classes.astype('uint8')

    # mask out np.nan
    mask = (classes == 255)
    
    # obtain the gpd features and create a gdf
    gpd_features = get_geopandas_features_from_array(classes, profile['transform'], 'class', mask, 4)
    gdf = gpd.GeoDataFrame.from_features(gpd_features, crs=CRS.from_epsg(4326)) 

    return gdf


def filter_by_area(gdf: gpd.GeoDataFrame, 
                   min_area_sq_km: int=1) -> gpd.GeoDataFrame:
    """
    Filter out polygons with area less than the input min_area_sq_meters.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing the polygons of the clustering result
    min_area_sq_km : int, optional
        All polygons with area less than this value is filtered out.
        The unit is kilometer square. The default is 1.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A filtered gdf containing only polygons with areas greater than the input.
        A new column is added to the gdf representing the area in sq km.

    """
    sq_m = min_area_sq_km * 1000000
    gdf = gdf.to_crs('epsg:3857') # change crs so can filter by area
    gdf = gdf[gdf.area > sq_m]
    gdf['area_sq_km'] = gdf.geometry.area / 1000000 # convert from sq_m to sq_km
    gdf = gdf.to_crs('epsg:4326') # change crs back to long/lat
    
    return gdf


def find_intersection(gdf:gpd.GeoDataFrame,
                      longitude: float, 
                      latitude: float, 
                      radius: float=0.005) -> gpd.GeoDataFrame:
    """
    Find the polygon in the input gdf that intersect with an user inputted coordinate.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing the polygons with the fire perimeters
    longitude : float
        longitude of a coordinate within the fire area. 
        Used to help identify the true perimeter.
    latitude : float
        latitude of a coordinate within the fire area.
        Used to help identify the true perimeter.
    radius : float, optional
        Buffer distance in degree relative to epsg:4326.
        The larger the value, the larger circular polygon the gdf is intersecting.
        The default is 0.005.

    Returns
    -------
    target_gdf : gpd.GeoDataFrame
        The GeoDataFrame containing the polygon of the fire perimeter. 
        Ideally, there should only be 1 polygon in the output gdf, 
        which corresponds to the fire perimeter aassociated with the user input coordinate.
        
    Notes
    -------
    Change the longitude and latitude if no polygon is in the returned gdf.

    """
    gdf = gdf.to_crs('epsg:4326') # change crs back to long/lat
    target_point = Point(longitude,latitude) # define a point within the fire area

    # generate a circular polygon based on the input point 
    target_buffer = target_point.buffer(radius)
    target_gdf = gdf[gdf.intersects(target_buffer)]
    
    return target_gdf
    

def merge_geojson(paths: list) -> gpd.GeoDataFrame:
    """
    Merge multiple GeoJSON files into one.

    Parameters
    ----------
    paths : list
        A list storing the paths to the generated GeoJSONs

    Returns
    -------
    output_gdf : gpd.GeoDataFrame
        A single GeoJSON file merging the input files as a single file

    """
    
    merged_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(path) for path in paths], ignore_index=True))
    merged_geom = merged_gdf.unary_union
    output_gdf = gpd.GeoDataFrame(geometry=[merged_geom])
    output_gdf.crs = merged_gdf.crs
    output_gdf = output_gdf.to_crs('epsg:3857')
    output_gdf['area_sq_km'] = output_gdf.geometry.area / 1000000
    output_gdf = output_gdf.to_crs('epsg:4326')
    
    return output_gdf
    

def merge_image(path_to_images: list, 
                header: str) -> None:
    """
    Merge outputted burn severity raster images into one GeoTIFF if a fire involves multiple flight lines

    Parameters
    ----------
    path_to_images : List
        A List of dataset objects or filenames.
    header : str
        The header of the output file name. 
        Output names would be '<header>_merged.tif'

    Returns
    -------
    None
        Outputs GeoTiff named '<header>_merged.tif' in the working directory.
        It should be the merged of all the flight lines inputted. 
    
    """
    header = str(header)

    # merge input files 
    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', srcNodata=0, VRTNodata=0)
    vrt = gdal.BuildVRT('merged.vrt', path_to_images, options=vrt_options)
    gdal.Translate(header + '_merged.tif', vrt)
    vrt = None
    print('Merged file name:')
    print(header + '_merged.tif')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wen Tao Lin

Contains a function `gen_perimeter_sbs` that automates mapping in one-go.
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from pathlib import Path
from sklearn.cluster import KMeans
from shapely.geometry import Polygon
from shapely.geometry import Point
from process_utils import (superpixel_segmentation,
                            tv_denoise, 
                            preprocess_for_merge,
                            inc_filter,
                            weighted_inc_merge,
                            get_superpixel_model_labels,
                           convert_labels_to_gdf,
                           filter_by_area,
                           get_model_labels)
from nd_utils import (get_superpixel_means_as_features,
                      get_superpixel_stds_as_features,
                      get_array_from_features)
from rio_utils import (reproject_arr_to_match_profile)
from crop_utils import (crop_image_by_geojson_shp)

# Function to read UAVSAR images
def open_one(path):
    with rasterio.open(path) as ds:
        band = ds.read(1)
        profile = ds.profile
        transform = ds.transform
    return band, profile, transform

# Function to denoise image using total-variation denoising
def denoise(data, weight):
    mask = np.isnan(data)
    data[mask] = 9999
    data_tv = tv_denoise(data, weight)
    data_tv[mask] = np.nan

    return data_tv

# Function to make a buffered point based on user-input to help locate area of interest
def make_center_point(longitude, latitude):
    buffer_point = Point(longitude, latitude)
    buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_point])

    return buffer_gdf


def gen_perimeter_sbs(
    data_path: str,
    root_name: str,
    resampling: str,
    longitude: float,
    latitude: float,
    min_size: int = 15,
    weight: float = 5.0,
    min_area_sq_km: float = 1,
    top_k_index: int = 10,
    superpixel_path: str = None
    ):
    """
    Generate UAVSAR fire perimeter and burn severity based 
    on the given cropped flight line and incidence angle.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the files.
    root_name : str
        Rootname to name all the output files.
    resampling : str
        The type of resampling to use for reprojection. See all the options:
        https://github.com/mapbox/rasterio/blob/08d6634212ab131ca2a2691054108d81caa86a09/rasterio/enums.py#L28-L40
    longitude : float
        Longitude of the fire area. Will be used to help select polygons for perimeter.
    latitude : float
        Latitude of the fire area. Will be used to help select polygons for perimeter.
    min_size: int, optional
        Minimum component size for segmentation. Enforced using postprocessing. This will greatly affect the output parameter.
        Parameter for skimage.segmentation.felzenszwalb. 
        For more info: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
        The default is 15.
    weight : float, optional
        The weight parameter in skimage.restoration.denoise_tv_chambolle. 
        The greater the weight, the more denoising (at the expense of image accuracy).
        For more info: https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle
        The default is 5.0.
    min_area_sq_km : float, optional
        All polygons with area less than this value is filtered out for fire perimeter.
        The unit is kilometer square. The default is 1.
    top_k_index : int, optional
        The top k largest post-filtered polygons will be returned for user selection for fire perimeter.
        The default is 10.
    superpixel_path: str, optional
        The path to .npy file with containing the superpixel label.
        The default is None and a .npy file will be created.

    Returns
    -------
    None
        Outputs
            - Preprocessed merged flights
            - Superpixel labels, if not exist
            - UAVSAR flight image intersection with the generated perimeter
            - UAVSAR Fire Perimeters (Reg, Concave, Convex)
            - UAVSAR Burn Severity (Based on the selected perimeter)
    """
    # Set the data directory
    data_dir = Path(data_path)

    # Locate flight images and incidence angle files
    imgs = sorted(list(data_dir.glob('./*_cropped_*.tif')))
    incs = sorted(list(data_dir.glob('./*_inc_*.tif')))

    # Display the available options to the user
    print("Available flight image files:")
    for i, img_path in enumerate(imgs):
        print(f"{i}: {img_path}")

    # Let the user choose the base flight line for reprojection
    selected_index = int(input("Enter the index of the image you want to select as the base-flight for reprojection. The rest of the images will be reproject to this profile : "))

    # Open the files
    img_bands, img_profiles, img_transforms = zip(*map(open_one, imgs))
    inc_bands, inc_profiles, _ = zip(*map(open_one, incs))

    # Select transform and profile based on inputted base-flight
    transform = img_transforms[selected_index]
    profile = img_profiles[selected_index]

    img_bands = list(img_bands)
    inc_bands = list(inc_bands)

    print("Preprocessing images to merge...")

    # Preprocess flight images and inc
    for i in range(len(img_bands)):
        img_bands[i] = preprocess_for_merge(img_bands[i]) # mask out the missing data after RTC. Scale the values to ensure consistency
        inc_bands[i] = inc_filter(inc_bands[i]) # convert angle from radian to degree. Filter out the bad angles

    # Resample the array to match all to a profile
    for i in range(len(img_bands)):
        img_bands[i], _ = reproject_arr_to_match_profile(img_bands[i], 
                                                        img_profiles[i], 
                                                        profile, 
                                                        resampling=resampling)
        img_bands[i] = img_bands[i][0]

        inc_bands[i], _ = reproject_arr_to_match_profile(inc_bands[i], 
                                                        inc_profiles[i], 
                                                        profile, 
                                                        resampling=resampling)
        inc_bands[i] = inc_bands[i][0]

    # Merge the opposite direction flights
    merged_0 = weighted_inc_merge(img_bands[0], img_bands[2], inc_bands[0], inc_bands[2])
    merged_1 = weighted_inc_merge(img_bands[1], img_bands[3], inc_bands[1], inc_bands[3])

    # Denoise merged flights
    merged_0_tv = denoise(merged_0, weight)
    merged_1_tv = denoise(merged_1, weight)

    # Output the preprocessed merged flight
    merged_out_path_0 = data_path + root_name + '_merged_0.tif'
    merged_out_path_1 = data_path + root_name + '_merged_1.tif'  

    with rasterio.open(merged_out_path_0, "w", **profile) as dest:
        dest.write(merged_0_tv, 1)
    print(merged_out_path_0 + " is outputted.")
    with rasterio.open(merged_out_path_1, "w", **profile) as dest:
        dest.write(merged_1_tv, 1)
    print(merged_out_path_1 + " is outputted.")

    # Load superpixel labels or perform superpixel segementation
    if superpixel_path is None:
        print("Performing superpixel segmentation...")
        superpixel_labels = superpixel_segmentation(merged_0_tv, merged_1_tv, min_size=min_size)
        superpixel_out_path = data_path + root_name + f'_superpixel_labels_minsize{min_size}.npy'
        np.save(superpixel_out_path, superpixel_labels)
        print(superpixel_out_path + " is created and saved.")
    else: 
        superpixel_labels = np.load(superpixel_path)
        print(superpixel_path + " is loaded.")

    # Calculate log ratio based on preprocessed images
    log_ratio_perim = np.log10(merged_0_tv/merged_1_tv)

    # Extract features from the segmentation
    mean_features = get_superpixel_means_as_features(superpixel_labels, log_ratio_perim)
    std_features = get_superpixel_stds_as_features(superpixel_labels, log_ratio_perim)

    # Select the K-Means model with n_clusters=2, representing fire and non-fire
    kmeans_n2 = KMeans(n_clusters=2, n_init=10,random_state=1)

    print("Running Perimeter Classification...")

    # Converting classification to a gdf
    labeled_data = get_superpixel_model_labels(np.hstack((mean_features, std_features)), kmeans_n2)
    fire_classes = get_array_from_features(superpixel_labels, labeled_data)
    gdf = convert_labels_to_gdf(fire_classes, profile)

    # Filtering out the false positive regions and desired number of polygons
    gdf = filter_by_area(gdf, min_area_sq_km)
    fire_gdf = gdf[gdf['class'] == 1] # by the ordering of the labels, '1' will be burned areas
    fire_gdf = fire_gdf.sort_values(by='area_sq_km', ascending=False)
    fire_gdf = fire_gdf.reset_index(drop=True)
    fire_gdf = fire_gdf[0:top_k_index]

    # Plot the GeoDataFrame and allow user to select the desired polygons for perimeter
    fig, ax = plt.subplots(figsize=(8, 8))
    show(log_ratio_perim, ax=ax, transform=transform, cmap='viridis', interpolation='none')
    fire_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.2)
    for idx, polygon in fire_gdf.iterrows():
        x, y = polygon.geometry.centroid.x, polygon.geometry.centroid.y
        ax.text(x, y, str(idx), fontsize=12, color='white', ha='center', va='center')

    # Create a buffer point of the input fire coordinate
    make_center_point(longitude, latitude).plot(ax=ax, color='blue', markersize=25)

    plt.title('Possible Fire Polygons overlapped with HV Log Ratio')
    plt.show()

    # Ask the user for polygons to keep as part of the output perimeter
    user_input = input("Enter a list of index to keep (e.g., 0,1,5,6) as part of perimeter. Enter -1 to keep all.")

    # Save the polygons based on user input
    index_to_keep = [int(x) for x in user_input.split(",")]
    if index_to_keep == [-1]:
        target_gdf = fire_gdf
    else:
        target_gdf = fire_gdf.loc[index_to_keep]

    # Filenames to be outputted (concave is edit later once ratio is determined)
    output_fill = data_path + root_name + '_uavsar_perimeter.geojson'
    output_convex = data_path + root_name + '_uavsar_perimeter_convex.geojson'

    # Create and output a gdf with holes fill
    no_holes = gpd.GeoSeries([Polygon(p.exterior) for p in target_gdf["geometry"]])
    merged_polygon = no_holes.unary_union
    fill_gdf = gpd.GeoDataFrame(geometry=[merged_polygon])
    fill_gdf.crs = gdf.crs
    fill_gdf = fill_gdf.to_crs('epsg:3857')
    base_area = fill_gdf.geometry.area
    fill_gdf['area_sq_km'] = fill_gdf.geometry.area / 1000000
    fill_gdf = fill_gdf.to_crs('epsg:4326')

    fill_gdf.to_file(output_fill, driver='GeoJSON')
    print(output_fill + " is outputted.")

    # Create and output a gdf with concave hull
    # Loop through potential ratios for concave, until we find the smallest ratio with at least 100% of original area
    parameter_values = [i * 0.005 for i in range(0, 21)] 
    for p in parameter_values:
        concave_series = fill_gdf.concave_hull(p)
        concave_gdf = gpd.GeoDataFrame(geometry=concave_series)
        concave_gdf = concave_gdf.to_crs('epsg:3857')
        concave_area = concave_gdf.geometry.area
        concave_ratio = p
        if concave_area[0]/base_area[0] > 1: # concave perimeter needs to have an area greater than the original perimeter
            print(f"Concave Hull Ratio: {p}")
            concave_ratio = p
            break
    concave_gdf['area_sq_km'] = concave_gdf.geometry.area / 1000000
    concave_gdf = concave_gdf.to_crs('epsg:4326')
    output_concave = f"{data_path}{root_name}_uavsar_perimeter_concave_{concave_ratio}.geojson"
    concave_gdf.to_file(output_concave, driver='GeoJSON')
    print(output_concave + " is outputted.")

    # Create and output a gdf with convex hull
    convex_series = fill_gdf.convex_hull
    convex_gdf = gpd.GeoDataFrame(geometry=convex_series)
    convex_gdf = convex_gdf.to_crs('epsg:3857')
    convex_gdf['area_sq_km'] = convex_gdf.geometry.area / 1000000
    convex_gdf = convex_gdf.to_crs('epsg:4326')
    convex_gdf.to_file(output_convex, driver='GeoJSON')
    print(output_convex + " is outputted.")

    # Plot the GeoDataFrames
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    fill_gdf.plot(ax=axs[0], cmap='viridis')
    axs[0].set_title('UAVSAR Perimeter 1')

    concave_gdf.plot(ax=axs[1], cmap='viridis')
    axs[1].set_title('UAVSAR Perimeter 2')

    convex_gdf.plot(ax=axs[2], cmap='viridis')
    axs[2].set_title('UAVSAR Perimeter 3')

    plt.show()

    # Ask the user to choose one a perimeter to outline burn severity
    choice = input("Choose a UAVSAR Perimeter to outline the burn severity map (enter 1, 2, or 3): ")

    # Check user input and save the selected GeoDataFrame as a variable
    if choice == '1':
        selected_gdf_path = output_fill
    elif choice == '2':
        selected_gdf_path = output_concave
    elif choice == '3':
        selected_gdf_path = output_convex
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

    print("Extracting flight line based off UAVSAR Perimeter...")

    # Extract UAVSAR data based off the generated UAVSAR Perimeter
    img_paths = [merged_out_path_0, merged_out_path_1]
    output_names = [img[:-4] + '_uavsar_intersect.tif' for img in img_paths]
    for i in range(len(img_paths)):

        # Crop and output
        crop_image_by_geojson_shp(selected_gdf_path,
                                img_paths[i], 
                                output_names[i])

    print("Running Burn Severity Classification...")

    # Load the UAVSAR flight intersection with perimeter
    intersect = sorted(list(data_dir.glob('./*_uavsar_intersect*')))
    intersect_bands, intersect_profiles, _ = zip(*map(open_one, intersect))

    # Compute log-ratio
    log_ratio_sbs = np.log10(intersect_bands[0]/intersect_bands[1])

    print("Running classification model...")

    # Select the K-Means model with n_clusters=4, representing the 4 severity classes
    # 1=Unburned, 2=Low, 3=Moderate, 4=High
    kmeans_n4 = KMeans(n_clusters=4, 
                    n_init=10, 
                    random_state=1)

    # Run K-means
    burn_classes = get_model_labels(log_ratio_sbs, kmeans_n4)
    burn_classes = burn_classes + 1

    # Output Burn Severity
    sbs_out_path = data_path + root_name + '_uavsar_sbs'
    if choice == '1':
        sbs_out_path += '.tif'
    elif choice == '2':
        sbs_out_path += '_concave.tif'
    elif choice == '3':
        sbs_out_path += '_convex.tif'

    with rasterio.open(sbs_out_path, "w", **intersect_profiles[0]) as dest:
        dest.write(burn_classes, 1)
    print(sbs_out_path + " is outputted.")
    print("Thank you for using this product.")
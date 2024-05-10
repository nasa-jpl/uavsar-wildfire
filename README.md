# uavsar_wildfire_classification
This product uses UAVSAR polarimetric data to generate **fire perimeter** and **burn severity** mappings. Specifically, HV polarization backscatter is used. 

Please apply radiometric terrain correction (RTC) to the raw UAVSAR prior to using this product. And generate a HDR environment for the incidence angle data. Both can be performed by following the [radiometric_terrain_correction](https://github.jpl.nasa.gov/UAVSAR-Fire-Research/radiometric_terrain_correction) repository.

Make sure conda is set up prior to use. Else, mambaforge can be installed [here](https://github.com/conda-forge/miniforge#mambaforge), or conda can be installed [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Setup (Mac)
1. Open terminal
2. Clone this repository: 

    `git clone git@github.jpl.nasa.gov:UAVSAR-Fire-Research/uavsar_wildfire_classification.git`
    
3. Create a virtual environment using conda via:

    `conda create --name classification python=3.11`
	
	  Make sure to hit `y` to confirm that the listed packages can be downloaded for this environment.
    
4. Activate the virtual environment: 

	  `conda activate classification`.

5. Install requirements: (Note: this will take a while)

	`conda env update -f environment.yml`
  
6. Create a new jupyter kernel: 

	`python -m ipykernel install --user --name classification`
	
Make sure the kernel is `classification` when using jupyter-notebook

This has not been tested for windows OS but, the setup steps should be similiar to that of Mac except the steps are ran on Anaconda Prompt.
Check [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more information about setup.

## Data Info
This product requires incidence angle images and the post-RTC HV images for the area of interest. We will use both directions of a UAVSAR flight line. There should be minimum of 4 UAVSAR flights involved. They include:
- pre-fire data of both directions (2 flights)
- post-fire data of both directions (2 flights)

In the case that a fire require more than one UAVSAR coverage area(e.g. Bobcat north and south), the number of flights will increment in multiples of 4. However, we can only run one flight line at a time and then combine the outputs.

All we need are the incidence angle(`.inc`) and the HV backscatter data. Two important things:
- *The incidence angle should have a HDR environment so it can be opened*
- *The HV images here should have already undergone radiometric terrain correction*.

## Product Info
Activate the jupyter-notebook by going to the `notebook` folder and run `jupyter-notebook` from there. Or open the folder from VS-code.

### To Run the Product:
1. Navigate to `crop_flight_img.ipynb` in the `notebook` folder.

2. Crop the post-RTC HV image and incidence angle image to the area of interest. This should output 8 files (4 inc, 4 hv) for each coverage area.

3. For each coverage area, create a folder. The folder should contain the following to make up the pre & post fire data for both directions:
	- 4 cropped raw incidence angle files - must contain `_cropped_` in the name
	- 4 cropped RTC files (hv) - must contain `_inc_` in the name

	All the 8 files must be in `.tif` format. The naming and formatting is done automatically in `crop_flight_img.ipynb` notebook.

4. Run the `gen_perimeter_sbs` function in the `gen_perim_sbs.ipynb` notebook for each coverage area. The folder created in **Step 3** will be one of the input parameters.

5. The outputs (perimeter and burn severity mapping) of **Step 4** will be saved in the same folder that was fed in as parameter.

## Notebook Info

- `crop_flight_img.ipynb`: Used to crop incidence angle images and post-RTC HV images.

- `gen_perim_sbs.ipynb`: Used to generate both fire perimeter and burn severity maps. Requires the cropped images.

## Python Files

- `classify.py`: Contains the script for perimeter and burn severity generation

- `crop_utils.py`: Contains functions regarding cropping raster images

- `process_utils.py`: Contains functions regarding data preprocessing, modeling, and gdf processing

- `nd_utils.py`: Contains functions utilizing scipy.ndimage. Sourced from the [Simard Lanscape Lab](https://github.com/simard-landscape-lab).

- `rio_utils.py`: Contains functions utilizing rasterio. Sourced from the [Simard Lanscape Lab](https://github.com/simard-landscape-lab).


## Old Notebooks Folder & Instructions
This folder contains the notebooks from the old product. Below is the instruction for running the old product:

The notebooks should be used in the following order: 

*The raster image that is going to be cropped should have already undergone radiometric terrain correction*

**Fire Perimeter Generation**
1. `crop_image.ipynb`: Crop the raster image by a circle
2. `preprocessing_segmentation.ipynb`: Preprocess the cropped image and perform superpixel segmentation. The outputs for both are saved and used in the next step
3. `perimeter_generation.ipynb`: Generate the fire perimeter map

**Burn Severity Map Generation**
1. `crop_image.ipynb`: Crop the raster image by a fire perimeter in GeoJSON or shapefile
2. `burn_severity_generation.ipynb`: Generate the burn severity map (preprocessing will happen in this notebook)

**Notebooks Demonstrations**

- `crop_image.ipynb`
	- Crop raster image by circle defined by user input coordinates, geojson, and shapefile
	- Reproject raster images to another coordinate reference system

- `preprocessing_segmentation.ipynb`
	- Preprocess the images
	- Performs superpixel segmentation (partition the image into multiple segments)
	- Outputs are saved to safe runtime in the future. Superpixel segmentation should take a while to run

- `perimeter_generation.ipynb`
	- Involves user interaction to generate and select fire polygons 
	- Merge output GeoJSON if a fire involve multiple flight lines

- `burn_severity_generation.ipynb`
	- Generate the burn severity map
	- Visualizes the distribution of the log ratio values for each burn severity class through boxplot

**Each flight line should be processed separately and outputs could be using the `merge_geojson`(perimeter) or `merge_image`(severity) functions in `process_utils.py`**


## 

Copyright 2023, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

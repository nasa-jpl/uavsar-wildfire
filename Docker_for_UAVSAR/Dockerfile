# Use the official Jupyter base image with Python 3
FROM jupyter/scipy-notebook:latest

# Switch to root user to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gdal-bin \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the jovyan user
USER jovyan

# Create a directory for the project
WORKDIR /home/jovyan/uavsar-wildfire

# Clone the UAVSAR Wildfire repository
RUN git clone https://github.com/nasa-jpl/uavsar-wildfire.git ./uavsar-wildfire
RUN git clone https://github.com/nasa-jpl/uavsar-wildfire-rtc.git ./uavsar-wildfire-rtc

# Use conda to install GDAL and other dependencies
RUN conda install -c conda-forge \
    gdal \
    rasterio \
    pyproj \
    netCDF4 \
    h5py \
    geopandas \
    shapely

# Install Python dependencies directly with pip
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scipy \
    requests \
    jupyter

# Expose the port for Jupyter
EXPOSE 8888

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]

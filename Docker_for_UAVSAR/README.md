
# UAVSAR Wildfire Docker Setup

This guide will help you set up and run the Docker container for the UAVSAR Wildfire notebooks.

## Prerequisites

- Docker must be installed on your machine.

## Step-by-Step Guide

### 1. Install Docker

Docker is required to build and run the container. Follow the instructions for your operating system:

- **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)
- **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
- **Linux**: [Docker Engine for Linux](https://docs.docker.com/engine/install/)

For Windows and macOS, Docker Desktop is recommended. For Linux, follow the instructions to install Docker Engine.

### 2. Verify Docker Installation

After installing Docker, verify the installation by running the following command in your terminal or command prompt:

```sh
docker --version
```

You should see the version of Docker installed.

### 3. Create a Directory for the Project

Create a directory on your local machine where you will store the Dockerfile and related files. For example:

```sh
mkdir ~/uavsar-wildfire-docker
cd ~/uavsar-wildfire-docker
```

### 4. Ensure the Dockerfile is in the new directory

Confirm that the Dockerfile is in the directory you created. If you don't have the Dockerfile, you can create one with the following content:

```dockerfile
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
```

### 5. Build the Docker Image

Open a terminal or command prompt in the directory where your Dockerfile is located and run the following command to build the Docker image **This may take some time be patient**:

```sh
docker build -t uavsar-wildfire-notebook -f Dockerfile .
```

This will create a Docker image named `uavsar-wildfire-notebook`.

### 6. Run the Docker Container

Run the Docker container using the following command:

```sh
docker run -p 8888:8888 uavsar-wildfire-notebook
```

This command maps the container's port 8888 to your local machine's port 8888.

### 7. Access the Jupyter Notebook

Once the container is running, you should see output in your terminal that includes a URL for accessing Jupyter Notebook or you can go to `http://localhost:8888/` within your browser. This will give you access to the Jupyter Notebook interface running inside the Docker container.

### 8. (Optional) Persist Data Between Container Runs

If you want to persist data between container runs or access your local files inside the container, you can mount a volume. Here’s an example of how to do this:

```sh
docker run -p 8888:8888 -v /path/to/local/directory:/home/jovyan/work uavsar-wildfire-notebook
```

Replace `/path/to/local/directory` with the path to the directory on your host machine that you want to mount inside the container. This will make the directory accessible inside the container at `/home/jovyan/work`.

### Summary

1. **Install Docker** from [Docker's official website](https://www.docker.com/get-started).
2. **Create a directory** for your project.
3. **Create and populate the Dockerfile** in this directory.
4. **Build the Docker image** with `docker build -t uavsar-wildfire-notebook .`.
5. **Run the Docker container** with `docker run -p 8888:8888 uavsar-wildfire-notebook`.
6. **Access Jupyter Notebook** at the provided URL.
7. (Optional) **Mount a volume** to persist data between container runs.

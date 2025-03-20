#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kris Mannino
"""
from pathlib import Path
def edit_paths(include: list[str],
               exclude: list[str],    
               paths: list[Path]) -> list[Path]:
    """
    Crops a raster image based on the shape of the first polygon in the 
    inputted GeoJSON or shapefile.

    Parameters
    ----------
    include : list[str]
        List of strings that must exist in Path.
    exclude : list[str]
        List of strings that if exists in Path, Path is removed.
    paths: list[Path]
        List of initial Paths.

    Returns
    -------
    list[Path]
        Returns adjust Path list with inclusions and exclusions.

    Notes
    -------

    """    
    if include:
        filtered_paths = [
            p for p in paths 
            if any(term in str(p) for term in include)
        ]
    else:
        filtered_paths = paths
        
    if exclude:
        filtered_paths = [
            p for p in filtered_paths 
            if not any(term in str(p) for term in exclude)
        ]
        
    return filtered_paths

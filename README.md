# RadixCO2Reduction

Detection of CO2 reduction measures using remote sensing.


## Setup Project

In order to install the project environment locally, run `./tasks/init.sh` from Terminal in the project's root folder. 
When successful, you can activate the project's environment by running `source activate radix-co2-reduction-env`.

A list of project-tasks exist, which can be summon using the `invoke` command:
- `invoke bump` Bump the major, minor, patch, or post-release part of this package's version.
- `invoke docs` Generate this package's docs.
-`invoke lab` Run Jupyter Lab.
- `invoke lint` Lint this package. 
- `invoke test` `Test this package.
- `invoke conda.create` Recreate the conda environment.
- `invoke conda.update` Update the conda environment.
- `invoke --list` Get overview of all the `invoke` commands.

It is possible to extend the `invoke` list by adding your own task in the `./tasks/tasks.py` file.


## Google Earth Engine

This repository makes use of the [Google Earth Engine API](https://earthengine.google.com/) to extract satellite imagery of fields, on which the predications are based.
In order to make use of this API, you'll need to [sign up](https://signup.earthengine.google.com/) first (the API is free). 
All code related to the GEE can be found in the `src/radix_co2_reduction/earth_engine` folder.


## Field Detection

In order to convert a geographic coordinate `(longitude, latitude)` into a field, we've implemented and trained a [Mask R-CNN network in PyTorch](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) on the instance segmentation task.
Our implementation can be found in the `src/radix_co2_reduction/field_detection` folder.


## Tillage Classification

The objective of the *Tillage Classification* task is to predict if tillage has been done over a continuous time period for a specific field.
In order to make this prediction, we've used both the GEE code (to extract field-level data) as the Field Detection code (to extract field boundaries).
The code used to make this prediction can be found in the `src/radix_co2_reduction/tillage_detection` folder.

You can access a **pipeline** of this code in the `src/radix_co2_reduction/tillage.py` file, which performs the following steps:
 1) Take in a coordinate `(longitude, latitude)` and a time-frame `(starting date, end-date)` in `YYYY-MM-DD` format.
 2) Extract the field boundaries of the field marked by the coordinate.
 3) Sample the detected field over the specified time-frame.
 4) Make tillage-predictions based on the sampled data.

The result of this pipeline is a boolean value that indicates if the field has been tilled (`True`) or not (`False`).

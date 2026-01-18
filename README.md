# Air–Sea Heat Flux Computation and Probablity Distribution of Turbulent Heat Fluxes

## Overview
This repository contains the code used in the scientific paper:

> ** Revisited heat budget and probability distributions of turbulent heat fluxes in the Mediterranean Sea<img width="468" height="13" alt="image" src="https://github.com/user-attachments/assets/29a7322f-f2de-408e-88a8-0780b0c841cb" />
  **  
> *[Ocean Science, 2025]*

The code is designed to compute **air–sea heat fluxes** using atmospheric variables and sea surface temperature (SST) datasets. It supports the calculation and output of different heat flux components, including sensible heat flux, latent heat flux, longwave radiation, and short wave radiation. The code  provides the results to .npy or NetCDF files.

---

## Purpose
The purpose of this repository is to:
- Ensure **reproducibility** of the results presented in the paper
- Provide a transparent implementation of the heat flux calculations
- Enable reuse and extension of the methodology for related air–sea interaction studies

---

## Data Description
The code operates on datasets:
- Atmospheric  (e.g., near-surface air temperature, humidity, wind speed, mean sea-level pressure, cloud coverage)
- Sea surface temperature (SST)

**Note:**  
Input datasets are not included in this repository due to size and/or data policy restrictions. Users should obtain the required datasets from their original sources and preprocess them as described in the paper.

---

## Code Structure
- scripts/ # Main source code
- README.md 
- LICENSE # License information
- .gitignore # Ignored files and directories

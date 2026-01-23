# pySimpleMask

[![PyPI version](https://img.shields.io/pypi/v/pysimplemask.svg)](https://pypi.python.org/pypi/pysimplemask)
[![Build Status](https://img.shields.io/travis/AdvancedPhotonSource/pySimpleMask.svg)](https://travis-ci.com/AdvancedPhotonSource/pySimpleMask)
[![Documentation Status](https://readthedocs.org/projects/pysimplemask/badge/?version=latest)](https://pysimplemask.readthedocs.io/en/latest/?version=latest)

**pySimpleMask** is a graphical user interface (GUI) tool designed for creating masks and Q-partition maps for scattering patterns, specifically facilitating SAXS, WAXS, and XPCS data reduction.

## Features

*   **Versatile Data Support**: Load scattering data from various formats including HDF5, IMM, TIFF, and binary files.
*   **Interactive Masking**:
    *   **Drawing Tools**: Create masks using polygons, circles, rectangles, and lines.
    *   **Thresholding**: Automatically mask pixels based on intensity limits (low/high).
    *   **Blemish Maps**: Apply pre-existing blemish (bad pixel) files.
    *   **Outlier Removal**: Automatically detect and mask outliers using SAXS 1D azimuthal average comparisons.
    *   **Manual Selection**: Click to mask specific pixels or regions.
*   **Partition Generation**:
    *   Compute Q-Phi partitions (Dynamic/Static).
    *   Generate X-Y partitions.
    *   Support for custom mapping modes.
*   **Visualization**: Real-time visualization of scattering patterns, masks, and SAXS 1D profiles.
*   **Output**:
    *   Save generated masks as TIFF files.
    *   Save full partition maps and metadata in Nexus-compatible HDF5/XPCS formats.

## Installation

### From PyPI
```bash
pip install pysimplemask
```

### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/AdvancedPhotonSource/pySimpleMask.git
   cd pysimplemask
   ```
2. Install the package:
   ```bash
   pip install .
   ```

## Usage

To launch the GUI, simply run the following command in your terminal:

```bash
pysimplemask
```

You can also specify a starting path for data loading:

```bash
pysimplemask --path /path/to/your/data
```

## Workflow

1.  **Load Data**: Click "Select Raw" or "Load" to open your scattering data file.
2.  **Define Mask**:
    *   Use the "Mask" tabs to apply different masking techniques (Draw, Threshold, Blemish, etc.).
    *   Combine multiple masking methods as needed.
    *   Use "Evaluate" to preview the mask and "Apply" to finalize it.
3.  **Compute Partition**:
    *   Go to the "Partition" tab.
    *   Select the desired mode (e.g., Q-Phi) and configure parameters (number of bins, symmetry).
    *   Click "Compute Partition" to generate the maps.
4.  **Save Results**:
    *   Click "Save" to export the mask (TIFF) or the full partition data (HDF5).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

*   **Author**: Miaoqi Chu (mqichu@anl.gov)
*   **License**: BSD 3-Clause license

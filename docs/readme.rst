============
pySimpleMask
============

**pySimpleMask** is a graphical user interface (GUI) tool designed for creating masks and Q-partition maps for scattering patterns, specifically facilitating SAXS, WAXS, and XPCS data reduction.

Features
--------

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

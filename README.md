# DNA-distribution-from-microscopy-images

This is a simple Python code that calculates the distribution of cells in the cell cycle using microscopic images. The code requires DAPI images of a known resolution in TIF format, as well as a range of accepted nuclear areas. It uses the mathematical model of cell cycle distribution proposed by M. Fox (1980; https://doi.org/10.1002/cyto.990010114) to determine the proportion of cells in the G1, S, and G2 phases. The results are generated as a CSV file. An HTML report is also created to help verify the quality of the segmentation and the fit of the data to the model. 

If the segmentation is improper, the thresholding algorithm can be changed (this version uses Otsu thresholding) and the distance parameter in watershed segmentation can be adjusted. 

Please note that the code assumes image operations are conducted with four workers. You can adjust this number accordingly. 

This code requires Python 3.12!

Quick Start
1. Install Python 3.12 from python.org/downloads
2. Create virtual environment: `py -3.12 -m venv venv312`
3. Activate: `venv312\Scripts\Activate.ps1` (PowerShell) or `activate.bat` (cmd)
4. Install deps: `pip install -r requirements.txt`
5. Run: `python cell_distribution_in_cell_cycle.py input_dir output_dir pixel_size min_area max_area` (pixel size in μm, min_area and max_area in μm^2)

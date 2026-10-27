# DNA-distribution-from-microscopy-images

This is a simple Python code that calculates the distribution of cells in the cell cycle using microscopic images. The code requires DAPI images of a known resolution in TIF format, as well as a range of accepted nuclear areas. It uses the Dean-Jett-Fox model of the DNA content distribution (Dean & Jett 1974; M. Fox 1980, https://doi.org/10.1002/cyto.990010114) to determine the proportion of cells in the G1, S, and G2/M phases. The results are generated as a CSV file. An HTML report is also created to help verify the quality of the segmentation and the fit of the data to the model.

If the segmentation is improper, the thresholding algorithm can be changed (this version uses Otsu thresholding) and the distance parameter in watershed segmentation can be adjusted (`--min-distance`).

This code requires Python 3.12!

## Quick Start
1. Install Python 3.12 from python.org/downloads
2. Create virtual environment: `py -3.12 -m venv venv312`
3. Activate: `venv312\Scripts\Activate.ps1` (PowerShell) or `activate.bat` (cmd)
4. Install deps: `pip install -r requirements.txt`
5. Run: `python cell_distribution_in_cell_cycle.py input_dir output_dir pixel_size min_area max_area` (pixel size in μm, min_area and max_area in μm²)

Outputs masks, histograms, CSV results, and interactive HTML report in `output_dir`.

### Options
| Option | Meaning |
|---|---|
| `--workers N` | parallel workers for image processing (default 4) |
| `--min-distance PX` | watershed seed spacing in pixels (default: half the radius of the smallest accepted nucleus, at least 3) |
| `--synchronous` | add Fox's Gaussian term to S phase, for synchronised populations |
| `--free-g2-cv` | fit the CV of the G2/M peak separately (default: same CV as G1) |
| `--ratio MIN MAX` | allowed G2/G1 intensity ratio (default 1.7 2.3) |

## Measurement
Integrated intensity is background-corrected. Uneven illumination should be flat-field corrected before running the script.

## Model
For DNA content x:

    G1(x) = n1 · N(x; μ1, CV1·μ1)
    G2(x) = n2 · N(x; μ2, CV2·μ2),       μ2 = r·μ1
    S(x)  = ∫ s(x') · N(x; x', CV(x')·x') dx'   over x' from μ1 to μ2

s(x') is the DNA content density of S-phase cells before measurement noise: a second-order polynomial (Dean & Jett), plus a Gaussian if `--synchronous` is used (Fox). The polynomial is written in the Bernstein basis with non-negative coefficients, which keeps it non-negative. An exponential debris term absorbs fragments and out-of-focus objects below G1; it is not counted in the fractions.

The model is fitted by extended maximum likelihood on the individual nuclei, within a window around G1 and G2.

`Results.csv` reports, per image, the fractions, the G1 position, the G2/G1 ratio, the CVs, the number of nuclei inside and outside the fit window, and whether the optimiser converged. Check the G2/G1 ratio (close to 2 when segmentation and background are right) and the residual panel of each `_model.png`.

### Precision
With simulated data from the same model, the typical error of a fraction is about ±3% (G1) and ±5% (S) with 400 nuclei, and about ±3% (G1) and ±9% (S) with 150 nuclei. Analyse at least a few hundred nuclei per sample, or pool images. Early-S cells sit under the G1 peak, so populations with a large early-S fraction tend to have G1 underestimated and S overestimated.

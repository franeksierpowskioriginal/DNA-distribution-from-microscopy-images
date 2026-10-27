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

Example, with 1.82 μm pixels, nuclei of 30–350 μm², masks enlarged by 2 pixels and 4 parallel workers:

    python cell_distribution_in_cell_cycle.py overview_tif results 1.82068 30 350 --expand 2 --workers 4

All `.tif`/`.tiff` files in `input_dir` are analysed. Every image must be a single-channel (DAPI) 2D image.

### Options
| Option | Meaning |
|---|---|
| `--workers N` | parallel workers for image processing (default 4) |
| `--min-distance PX` | watershed seed spacing in pixels (default: half the radius of the smallest accepted nucleus, at least 3) |
| `--expand N` | enlarge each nuclear mask by N pixels before summing the intensity (default 0); masks grow only into background, never into neighbouring nuclei. Useful when nuclei are only a few pixels wide |
| `--synchronous` | add Fox's Gaussian term to S phase, for synchronised populations |
| `--free-g2-cv` | fit the CV of the G2/M peak separately (default: same CV as G1) |
| `--ratio MIN MAX` | allowed G2/G1 intensity ratio (default 1.7 2.3) |

## Outputs
In `output_dir`:

| File | Content |
|---|---|
| `Results.csv` | one row per image: number of nuclei, fractions G1/S/G2, G1 position, G2/G1 ratio, CVs, nuclei inside/outside the fit window, optimiser convergence |
| `report.html` | interactive report: summary bar plot, results table, and the per-image plots below |
| `Cell_cycle_barplot.png` | stacked bar plot of the fractions for all images |
| `<image>.csv` | one row per nucleus: centroid (`Y`, `X`), min/max/mean intensity, raw integrated intensity, background, background-corrected integrated intensity (`IntDen_bgsub`, used for the analysis), segmented area (`Area`) and area after `--expand` (`Area_measured`); areas in pixels |
| `<image>_mask.png` | segmented nuclei |
| `<image>_spatial.png` | each nucleus's intensity relative to the median, at its position in the image |
| `<image>_histogram.png` | DNA content histogram |
| `<image>_initial_model.png` | model at the starting values of the fit |
| `<image>_model.png` | fitted model with its components and residuals |

## Measurement
- Nuclei are segmented by Gaussian smoothing, Otsu thresholding, hole filling and a watershed on the distance transform. Nuclei touching the image border and objects outside the area range are removed.
- Nuclear area limits are applied to the segmented masks (before any `--expand`). A minimum area well above 0 keeps small debris out of the analysis.
- Integrated intensity is background-corrected: the median of pixels far from any nucleus is subtracted from every pixel (`IntDen_bgsub` column). Uneven illumination should be flat-field corrected before running the script.
- With `--expand N`, intensities are summed over masks enlarged by N pixels, so that the blurred edge of each nucleus is included. This matters when nuclei are only a few pixels wide: there, moving the mask outline by one pixel changes the integrated intensity considerably. Check the G2/G1 ratio when choosing N.

## Model
For DNA content x:

    G1(x) = n1 · N(x; μ1, CV1·μ1)
    G2(x) = n2 · N(x; μ2, CV2·μ2),       μ2 = r·μ1
    S(x)  = ∫ s(x') · N(x; x', CV(x')·x') dx'   over x' from μ1 to μ2

s(x') is the DNA content density of S-phase cells before measurement noise: a second-order polynomial (Dean & Jett), plus a Gaussian if `--synchronous` is used (Fox). The polynomial is written in the Bernstein basis with non-negative coefficients, which keeps it non-negative. An exponential debris term absorbs fragments and out-of-focus objects below G1; it is not counted in the fractions.

The model is fitted by extended maximum likelihood on the individual nuclei, within a window around G1 and G2.

## Checking the quality of a result
Look at these before trusting the fractions of an image:

- **CV of G1** (`CV_G1` in `Results.csv`). This is the width of the G1 peak and sets how well S can be separated from G1 and G2. Below about 10% the S fraction is well defined. Between 10 and 20% the G1 fraction is still reasonable, but the split between S and G2/M becomes unstable. A CV at the upper limit (30%) means the fit is not usable.
- **G2/G1 ratio** (`G2_G1_ratio`). It should be close to 2. A value at the limit (1.7 or 2.3) points to a segmentation, background or focus problem, or to cells that are not separated (doublets look like G2).
- **Residuals** in `_model.png`. They should scatter around 0 without a pattern. With tens of thousands of nuclei even small deviations from the Gaussian peak shape give large residuals, so look at the pattern rather than at the chi² value.
- **Spatial map** (`_spatial.png`). It should look uniform. A gradient (often towards the edge of a well) or tile-shaped blocks mean uneven focus or illumination; the peaks then get broader and the fractions less reliable.

### Precision
With simulated data from the same model, the typical error of a fraction is about ±3% (G1) and ±5% (S) with 400 nuclei, and about ±3% (G1) and ±9% (S) with 150 nuclei. Analyse at least a few hundred nuclei per sample, or pool images. Early-S cells sit under the G1 peak, so populations with a large early-S fraction tend to have G1 underestimated and S overestimated.

### Imaging recommendations
- Nuclei should be at least ~15 pixels across (for typical mammalian nuclei a pixel size of ≤ 0.65 μm, e.g. a 10× or 20× objective). Low-magnification whole-well scans give broad peaks.
- Use autofocus for every field, avoid the edge of the well, and use flat-field correction for tiled images.
- Avoid saturated pixels and confluent areas where nuclei touch.
- For a direct measurement of S phase, label it (e.g. a short EdU pulse) in addition to DAPI.

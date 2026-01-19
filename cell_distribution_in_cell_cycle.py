import sys
import os
import glob
import numpy as np
import pandas as pd
import numpy.ma as ma
import skimage.io
import skimage
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import datetime
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import iqr
import datapane as dp
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
import imageio.v2 as imageio
import tifffile as tiff


def compute_histogram(values, q=0.99):
    h = (2*iqr(values))/(len(values)**(1/3))
    range = np.max(values) - np.min(values)
    no_of_bins = round(range/h)
    upper = np.quantile(values, q)
    return np.histogram(values, bins=no_of_bins, range=(0, upper))

def s_phase_component(x, x_vals, A, B, C, Ns, x_s, sigma_s, sigma1_S, x1):
    """
    Compute F_s(x) with a two-term discrete summation (j = 1, 2).

    Parameters
    ----------
    x : float
        Evaluation point:
    x_vals : array-like of length 2
        Values [x_1, x_2]:
    A, B, C : float
        Polynomial coefficients:
    Ns : float
        Gaussian normalization:
    x_s : float
        Center of the signal Gaussian:
    sigma_s : float
        Width of the signal Gaussian:
    sigma1 : float
        Base smoothing width:
    x1 : float
        Reference scale (x_1 in the formula):

    Returns
    -------
    float
        Value of F_s(x):
    """

    x_vals = np.asarray(x_vals)

    def f(xj):
        return (
            A
            + B * xj
            + C * xj**2
            + Ns / (np.sqrt(2.0 * np.pi) * sigma_s)
            * np.exp(-(xj - x_s)**2 / (2.0 * sigma_s**2))
        )

    Fs = 0.0
    for xj in x_vals:
        sigma_eff = sigma1_S * (xj / x1)
        Fs += (
            f(xj)
            / (np.sqrt(2.0 * np.pi) * sigma_eff)
            * np.exp(-(x - xj)**2 / (2.0 * sigma_eff**2))
        )
        # enforce S only between x1_s and x2_s
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    mask = (x >= x_min) & (x <= x_max)
    Fs[~mask] = 0.0

    return Fs

def process_image(filepath, name, output_dir, min_nucleus_area, max_nucleus_area):
    start = datetime.datetime.now()
    #load the image
    img = skimage.io.imread(filepath)

    #gaussian filter
    img = img.astype(np.float32, copy=False)
    end = datetime.datetime.now()
    duration = end - start
    print("Upload completed, execution time: ", duration)
    
    start = datetime.datetime.now()
    img_proc = skimage.filters.gaussian(img, sigma=0.8, preserve_range=True)
    end = datetime.datetime.now()
    duration = end - start
    print("Gaussian filtering performed, execution time: ", duration)
    
    start = datetime.datetime.now()
    #calculate threshold and make a binary mask
    thresh_otsu = threshold_otsu(img_proc)
    binary_otsu = img_proc > thresh_otsu
    end = datetime.datetime.now()
    duration = end - start
    print("Threshold calculated, execution time: ", duration)

    start = datetime.datetime.now()   
    #Fill holes, perform binary opening
    mask_filled = ndi.binary_fill_holes(binary_otsu)
    mask_opened = ndi.binary_opening(mask_filled)
    
    end = datetime.datetime.now()
    duration = end - start
    print("Binary calculation completed, execution time: ", duration)
    
    #Perform watershed
    start = datetime.datetime.now()
    distance_transform = ndi.distance_transform_edt(mask_opened)

    sigma = 1
    distance_transform_smoothened = ndi.gaussian_filter(distance_transform, sigma)

    distance = 3
    coordinates = peak_local_max(distance_transform_smoothened, min_distance=distance, labels=mask_opened)

    peak_mask = np.zeros_like(mask_opened, dtype = bool)
    peak_mask[tuple(coordinates.T)] = True
    peak_mask = ma.masked_values(peak_mask, False)
    labeled_seeds, _ = ndi.label(peak_mask)

    labels = watershed(-distance_transform_smoothened, labeled_seeds, mask=mask_opened)
    
    end = datetime.datetime.now()
    duration = end - start
    print("Watershed completed, execution time: ", duration)
    
    start = datetime.datetime.now()
    #Remove the cells at the border
    labels_copy = np.copy(labels)
    labels_list = np.unique(labels_copy)
    labels_list = np.delete(labels_list, 0)

    x1 = np.unique(labels_copy[0,:])
    x2 = np.unique(labels_copy[-1,:])
    y1 = np.unique(labels_copy[:,0])
    y2 = np.unique(labels_copy[:,-1])

    labels_to_remove = np.unique(np.concatenate([x1, x2, y1, y2]))
    labels_to_remove = np.delete(labels_to_remove, 0)

    for x in labels_to_remove:
        current = labels_copy == x
        labels_copy[current] = 0
        
    #Remove mask smaller than min and greater than max
    labels_flat = labels_copy.ravel()
    areas = np.bincount(labels_flat)

    keep = (areas >= min_nucleus_area) & (areas <= max_nucleus_area)
    keep[0] = False  # background always removed

    labels_filtered = labels_copy.copy()
    labels_filtered[~keep[labels_copy]] = 0
    
    end = datetime.datetime.now()
    duration = end - start
    print("Filtering out of incorrect masks completed, execution time: ", duration)

    start = datetime.datetime.now()
    # Save the resulting masks
    mask_path = os.path.join(output_dir, f"{name}_mask.tif")
    tiff.imwrite(mask_path, labels_filtered.astype(np.uint32, copy=False), compression="zlib")
    
    end = datetime.datetime.now()
    duration = end - start
    print("png generation completed, execution time: ", duration)

    start = datetime.datetime.now()
    #Quantify nuclei intensities
    labels = labels_filtered.ravel()
    image = img.ravel()

    mask = labels > 0
    labels = labels[mask]
    image = image[mask]

    if labels.size == 0:
        return "All nuclei were removed, the intensity quantification can't be performed."
    max_label = labels.max()

    area = np.bincount(labels)
    rawintden = np.bincount(labels, weights=image)

    mean = np.zeros_like(rawintden, dtype=float)
    np.divide(rawintden, area, out=mean, where=area > 0)

    min_vals = np.full(max_label + 1, np.inf)
    max_vals = np.full(max_label + 1, -np.inf)

    np.minimum.at(min_vals, labels, image)
    np.maximum.at(max_vals, labels, image)

    df = pd.DataFrame({
        "ID": np.arange(len(area)),
        "Min": min_vals,
        "Max": max_vals,
        "Mean_int": mean,
        "RawIntDen": rawintden,
        "Area": area
    }).iloc[1:]  # remove background

    csv_path = os.path.join(output_dir, f"{name}.csv")
    
    #Remove debris
    df = df[(df['Area'] >= min_nucleus_area) & (df['Area'] <= max_nucleus_area)]
    
    if df.empty:
        print(f"No valid nuclei found for {name}, skipping CSV generation")
        return "No valid nuclei found"

    csv_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    
    end = datetime.datetime.now()
    duration = end - start
    print("Nuclei intensities calculated, execution time: ", duration)
    
    start = datetime.datetime.now()
    #Generate a histogram showing the dispersion of nuclei intensities
    nuclei_intensities = []
    nuclei_intensities = df["RawIntDen"].astype(float).values
    counts, bin_edges = compute_histogram(nuclei_intensities, q=0.99)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    upper_lim = bin_edges[-1]
    step = round(upper_lim/5, -2)
    plot_path = os.path.join(output_dir, f"{name}_histogram.png")
    plt.figure(figsize = (8,8))
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], align='center')
    plt.xlabel("Integrated intensity")
    plt.ylabel("Cell count")
    plt.xticks(np.arange(start = 0,stop = upper_lim, step = step))
    plt.title("DNA Content Histogram")
    plt.savefig(plot_path)
    plt.close()
    end = datetime.datetime.now()
    duration = end - start
    print("Histogram generation completed, execution time: ", duration)
    
def djf_model(
    x,
    N1, mu1, sigma1,
    N2, mu2, sigma2,
    A, B, C, Ns, x_s, sigma_s, sigma1_S,
    k1=1.05, k2=0.95, n_pts=25
):
    sigma1 = max(sigma1, 1e-12)
    sigma2 = max(sigma2, 1e-12)
    sigma_s = max(sigma_s, 1e-12)
    sigma1_S = max(sigma1_S, 1e-12)

    # Same normalization you already use
    G1 = (N1/(np.sqrt(2*np.pi)*sigma1)) * np.exp(-(x - mu1)**2 / (2*sigma1**2))
    G2 = (N2/(np.sqrt(2*np.pi)*sigma2)) * np.exp(-(x - mu2)**2 / (2*sigma2**2))

    # Tie endpoints + reference
    x1_ref = max(mu1, 1e-12)
    x1_s = mu1 * k1
    x2_s = mu2 * k2
    if x2_s <= x1_s:
        # fallback (rare when mu2 ~ mu1)
        x1_s = mu1 * 1.02
        x2_s = mu2 * 0.98

    x_vals = np.linspace(x1_s, x2_s, n_pts)

    S = s_phase_component(
        x,
        x_vals=x_vals,
        A=A, B=B, C=C,
        Ns=Ns, x_s=x_s, sigma_s=sigma_s,
        sigma1_S=sigma1_S, x1=x1_ref
    )

    return G1 + S + G2

def estimate_initial_p0(x, y, q=0.99):
    
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_smooth = ndi.gaussian_filter1d(y.astype(float), sigma=2)
    
    # 1. Find G1, G2 peaks (first/second major peaks)
    peaks, _ = find_peaks(y_smooth, prominence=0.05*np.max(y_smooth))
    if len(peaks) < 2:
        peaks = np.linspace(0, len(y)-1, 2, dtype=int)  # Fallback
    
    g1_peak = peaks[np.argmax(y_smooth[peaks[:len(peaks)//2]])]
    g2_peak = peaks[len(peaks)//2 + np.argmax(y_smooth[peaks[len(peaks)//2:]])]
    
    # G1 params
    mu1 = x[g1_peak]
    N1 = y_smooth[g1_peak] * np.std(x[y_smooth > y_smooth[g1_peak]*0.5]) * np.sqrt(2*np.pi)
    mask1 = y_smooth > y_smooth[g1_peak]*0.5
    sigma1 = np.std(x[mask1]) if np.any(mask1) else (x[-1]-x[0])/20
    
    # G2 params  
    mu2 = x[g2_peak]
    N2 = y_smooth[g2_peak] * np.std(x[y_smooth > y_smooth[g2_peak]*0.5]) * np.sqrt(2*np.pi)
    mask2 = y_smooth > y_smooth[g2_peak]*0.5
    sigma2 = np.std(x[mask2]) if np.any(mask2) else (x[-1]-x[0])/20
    
    # 2. S-phase params: x_vals = [x1,x2] between G1/G2
    x1_s = mu1 * 1.05
    x2_s = mu2 * 0.95
    if x2_s <= x1_s:
        x1_s = mu1 * 1.02
        x2_s = mu2 * 0.98

    valley = (x > x1_s) & (x < x2_s)
    if np.count_nonzero(valley) >= 5:
        c2, c1, c0 = np.polyfit(x[valley], y_smooth[valley], 2)
        A, B, C = float(c0), float(c1), float(c2)
        valley_mean = float(np.mean(y_smooth[valley]))
    else:
        A, B, C = float(np.min(y_smooth)), 0.0, 0.0
        valley_mean = float(np.min(y_smooth))

    # Make Ns conservative to avoid initial spike
    Ns = max(valley_mean * 0.2, 1e-6)

    x_s = float(0.5*(x1_s + x2_s))
    sigma_s = max(float((x2_s - x1_s)/10), 1e-6)
    sigma1_S = max(float(0.6*np.mean([sigma1, sigma2])), 1e-6)

    p0 = [N1, mu1, sigma1, N2, mu2, sigma2, A, B, C, Ns, x_s, sigma_s, sigma1_S]
    p0 = np.asarray(p0, float)
    p0[~np.isfinite(p0)] = 1e-6
    return p0.tolist()

def djf_components_from_params(x, params, k1=1.05, k2=0.95, n_pts=25):
    (N1, mu1, sigma1,
     N2, mu2, sigma2,
     A, B, C, Ns, x_s, sigma_s, sigma1_S) = params
    
    sigma1 = max(float(sigma1), 1e-12)
    sigma2 = max(float(sigma2), 1e-12)
    sigma_s = max(float(sigma_s), 1e-12)
    sigma1_S = max(float(sigma1_S), 1e-12)

    G1 = (N1 / (np.sqrt(2*np.pi) * sigma1)) * np.exp(-(x - mu1)**2 / (2*sigma1**2))
    G2 = (N2 / (np.sqrt(2*np.pi) * sigma2)) * np.exp(-(x - mu2)**2 / (2*sigma2**2))
    
    x1_ref = max(float(mu1), 1e-12)
    x1_s = float(mu1) * k1
    x2_s = float(mu2) * k2
    if x2_s <= x1_s:
        x1_s = float(mu1) * 1.02
        x2_s = float(mu2) * 0.98

    x_vals = np.linspace(x1_s, x2_s, n_pts)
    
    S  = s_phase_component(
        x,
        x_vals=x_vals,
        A=A, B=B, C=C,
        Ns=Ns, x_s=x_s, sigma_s=sigma_s,
        sigma1_S=sigma1_S, x1=x1_ref)
    
    return G1, S, G2

def quantify_cell_cycle(filepath, name, output_dir, rows):
    start = datetime.datetime.now()
    try:
        df = pd.read_csv(filepath)

        required_cols = ['RawIntDen', 'Area']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} in {filepath}, skipping")
            return

        if df.empty or len(df) == 0:
            print(f"No data in {filepath}, skipping")
            return

        nuclei_intensities = df["RawIntDen"].astype(float).dropna().values
        if len(nuclei_intensities) == 0:
            print(f"No valid RawIntDen values in {filepath}, skipping")
            return

        counts, bin_edges = compute_histogram(nuclei_intensities, q=0.99)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    end = datetime.datetime.now()
    duration = end - start
    print("Loading the data completed, execution time: ", duration)
    
    start = datetime.datetime.now()
    print("Fitting the model to the data")
    p0 = estimate_initial_p0(bin_centers, counts)
    print("p0:", p0)
    print("any nonfinite:", any([not np.isfinite(v) for v in p0]))
    print("min p0:", np.min(p0))
    
    lower = [0]*13
    upper = [np.inf]*13

    # Allow polynomial coeffs A,B,C to be negative:
    # Param order: [N1,mu1,sigma1, N2,mu2,sigma2, x1_s,x2_s, A,B,C, Ns,x_s,sigma_s,sigma1_S,x1_ref]
    lower[6]  = -np.inf  # A
    lower[7]  = -np.inf  # B
    lower[8] = -np.inf  # C

    params, cov = curve_fit(
        djf_model, bin_centers, counts,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000
    )
    fitted = djf_model(bin_centers, *params)
    model_function = djf_model(bin_centers, *p0)
    end = datetime.datetime.now()
    duration = end - start
    print("Fitting completed, execution time: ", duration)
    
    start = datetime.datetime.now()
    print("Calculating model components")
    #fitted model components
    G1, S, G2 = djf_components_from_params(bin_centers, params)
    G1_p0, S_p0, G2_p0 = djf_components_from_params(bin_centers, p0)
    
    end = datetime.datetime.now()
    duration = end - start
    print("Model components calculation completed, execution time: ", duration)
    
    final_model_path = os.path.join(output_dir, f"{name}_model.png")
    initial_model_path = os.path.join(output_dir, f"{name}_initial_model.png")
    
    start = datetime.datetime.now()
    print("Generating the plots")
    
    plt.figure(figsize=(8, 8))
    
    plt.plot(bin_centers, counts, 'o', label='Data')
    plt.plot(bin_centers, fitted, '-', label='DJF fit')
    plt.plot(bin_centers, G1, '-', label='G1')
    plt.plot(bin_centers, S, '-', label='S')
    plt.plot(bin_centers, G2, '-', label='G2')
    plt.legend()
    plt.xlabel('DAPI intensity')
    plt.ylabel('Counts')
    plt.title('Fit Results')
    plt.savefig(final_model_path)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(bin_centers, counts, 'o', label='Data')
    plt.plot(bin_centers, model_function, '-', label='Initial model')
    plt.plot(bin_centers, G1_p0, '-', label='G1_p0')
    plt.plot(bin_centers, S_p0, '-', label='S_p0')
    plt.plot(bin_centers, G2_p0, '-', label='G2_p0')
    plt.legend()
    plt.xlabel('DAPI intensity')
    plt.ylabel('Counts')
    plt.title('Initial Model Components')
    plt.savefig(initial_model_path)
    plt.close()
    
    end = datetime.datetime.now()
    duration = end - start
    print("Plots generated, execution time: ", duration)
    
    start = datetime.datetime.now()
    print("Calculating the fractions of cells and saving the results")    
    total_area = np.trapz(fitted, bin_centers)
    frac_G1 = np.trapz(G1, bin_centers) / total_area
    frac_S  = np.trapz(S,  bin_centers) / total_area
    frac_G2 = np.trapz(G2, bin_centers) / total_area
    
    end = datetime.datetime.now()
    duration = end - start
    print("Calculation completed, execution time: ", duration)
    return frac_G1, frac_S, frac_G2
    
def plot_cell_cycle(results_csv_path, output_dir):
    
    df = pd.read_csv(results_csv_path)
    
    df['row'] = df.index
    
    x = np.arange(len(df))  # [0, 1, 2, ...]
    #width = 0.8

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x, df['frac_G1'], label='G1', color='#1f77b4')
    ax.bar(x, df['frac_S'], bottom=df['frac_G1'], label='S', color='#ff7f0e')
    ax.bar(x, df['frac_G2'], bottom=df['frac_G1'] + df['frac_S'], label='G2', color='#2ca02c')


    ax.set_xlabel('Sample (row number)')
    ax.set_ylabel('Fraction of cells')
    ax.set_title('Cell cycle phase distribution by sample')
    ax.set_xticks(x)
    ax.set_xticklabels(df['row'].astype(str))
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'Cell_cycle_barplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stacked bar plot saved: {plot_path}")
    
def process_image_wrapper(args):
    return process_image(*args)

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    size_of_1px = float(sys.argv[3])
    min_nucleus_area = float(sys.argv[4])
    max_nucleus_area = float(sys.argv[5])
    max_nucleus_area = max_nucleus_area/size_of_1px*2
    min_nucleus_area = min_nucleus_area/size_of_1px*2

    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    tasks = [(filepath, os.path.basename(filepath).split('.')[0], output_dir, 
              min_nucleus_area, max_nucleus_area) for filepath in tif_files]
    
    print("Processing images...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_image_wrapper, tasks))
    
    print("Processing CSVs for cell cycle quantification...")
    csv_files = sorted(glob.glob(os.path.join(output_dir, "*.csv")))
    valid_csv_files = [f for f in csv_files if not f.endswith('Results.csv')]
    
    rows = []
    
    for filepath in valid_csv_files:
        filename = os.path.basename(filepath).split('.')[0]
        print(f"Quantifying cell cycle phases in: {filename}")
        n_cells = len(pd.read_csv(filepath))
        frac_row = quantify_cell_cycle(filepath, filename, output_dir, rows)
        rows.append([filename, n_cells, *frac_row])
    
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "n_cells", "frac_G1", "frac_S", "frac_G2"])
        results_path = os.path.join(output_dir, "Results.csv")
        df.to_csv(results_path, index=False)
        print(f"Results saved: {len(rows)} samples analyzed")
        plot_cell_cycle(results_path, output_dir)

        # Generate datapane report (handles 0, 1, or many samples)
        results_df = pd.read_csv(results_path)
        summary_table = dp.Table(results_df)
        results_png = os.path.join(output_dir, "Cell_cycle_barplot.png")
        results_plot = dp.Media(file=results_png)

        if len(results_df) == 0:
            print("No samples for detailed report")
            report_blocks = [dp.Text("No valid samples found")]
        elif len(results_df) == 1:
            # Single sample: show directly (no Select needed)
            row = results_df.iloc[0]
            sample_id = row["ID"]
            n_cells = int(row["n_cells"])

            mask_path = os.path.join(output_dir, f"{sample_id}_mask.tif")
            hist_path = os.path.join(output_dir, f"{sample_id}_histogram.png")
            final_model = os.path.join(output_dir, f"{sample_id}_model.png")
            initial_model = os.path.join(output_dir, f"{sample_id}_initial_model.png")

            single_sample = dp.Group(
                dp.Text(f"### Sample: {sample_id}"),
                dp.Text(f"Total nuclei: **{n_cells}**"),
                dp.Media(file=mask_path, caption="Nuclei masks"),
                dp.Media(file=hist_path, caption="DAPI intensity histogram"),
                dp.Media(file=initial_model, caption="Initial model components"),
                dp.Media(file=final_model, caption="Fitted model"),
            )
            report_blocks = [single_sample]
        else:
            # Multiple samples: use Select dropdown
            sample_blocks = []
            for _, row in results_df.iterrows():
                sample_id = row["ID"]
                n_cells = int(row["n_cells"])

                mask_path = os.path.join(output_dir, f"{sample_id}_mask.tif")
                hist_path = os.path.join(output_dir, f"{sample_id}_histogram.png")
                final_model = os.path.join(output_dir, f"{sample_id}_model.png")
                initial_model = os.path.join(output_dir, f"{sample_id}_initial_model.png")

                sample_blocks.append(
                    dp.Group(
                        dp.Text(f"### Sample: {sample_id}"),
                        dp.Text(f"Total nuclei: **{n_cells}**"),
                        dp.Media(file=mask_path, caption="Nuclei masks"),
                        dp.Media(file=hist_path, caption="DAPI intensity histogram"),
                        dp.Media(file=initial_model, caption="Initial model components"),
                        dp.Media(file=final_model, caption="Fitted model"),
                        label=sample_id,
                    )
                )
            report_blocks = [dp.Select(blocks=sample_blocks, type=dp.SelectType.DROPDOWN)]

        report = dp.Report(
            dp.Text("# Cell cycle analysis report"),
            dp.Text("## Summary"),
            results_plot,
            summary_table,
            dp.Text("## Per-sample details"),
            *report_blocks  # Unpacks the list
        )

        report.save(path=os.path.join(output_dir, "report.html"), open=True)
        print("Report saved: report.html")
    else:
        print("No valid data for cell cycle analysis")

if __name__ == "__main__":
    main()
    

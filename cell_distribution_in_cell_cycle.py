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

def compute_histogram(values, q=0.99):
    h = (2*iqr(values))/(len(values)**(1/3))
    range = np.max(values) - np.min(values)
    no_of_bins = round(range/h)
    upper = np.quantile(values, q)
    return np.histogram(values, bins=no_of_bins, range=(0, upper))

def s_phase_component(x, mu1, mu2, sigma_s, A_s, n_u = 100):
    u = np.linspace(mu1, mu2, n_u)
    gaussian = np.exp(-0.5 * ((x[:, None] - u[None, :]) / sigma_s)**2)
    S = np.trapezoid(gaussian, u)
    return A_s * S / S.max()

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
    mask_path = os.path.join(output_dir, f"{name}_mask.png")
    plt.figure(figsize = (8,8))
    plt.imshow(labels_filtered, cmap="nipy_spectral")
    plt.title('Nuclei masks')
    plt.savefig(mask_path)
    plt.close()
    
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
    
def djf_model(x, N1, mu1, sigma1, N2, mu2, sigma2, A_s, sigma_s):
    
    G1 = N1 * np.exp(-(x - mu1)**2 / (2*sigma1**2))
    G2 = N2 * np.exp(-(x - mu2)**2 / (2*sigma2**2))
    S = s_phase_component(x, mu1, mu2, sigma_s, A_s)    
    return G1 + S + G2

def estimate_initial_p0(x, y):
    peaks, _ = find_peaks(y, prominence=0.05 * np.max(y))
    if len(peaks) == 0:
        raise ValueError("No peaks found in the histogram.")
      
    #G1 assumptions
    g1_index = peaks[np.argmax(y[peaks])]
    mu1 = x[g1_index]
    N1 = y[g1_index]

    half_max = N1 / 2
    left = np.where(y[:g1_index] < half_max)[0]
    right = np.where(y[g1_index:] < half_max)[0]
    if left.size > 0 and right.size > 0:
        l = x[left[-1]]
        r = x[g1_index + right[0]]
        sigma1 = (r - l) / 2.3548  # FWHM to sigma
    else:
        sigma1 = (x[-1] - x[0]) / 20  # fallback

    # G2 assumptions
    mu2 = 1.75 * mu1
    N2 = N1 * 0.5
    sigma2 = sigma1 * 1.1
    
    # S assumptions
    sigma_s = (mu2 - mu1) / 4
    s_range = (x > mu1 * 1.05) & (x < mu2 * 0.95)
    A_s = np.min(y[s_range])


    return [N1, mu1, sigma1, N2, mu2, sigma2, A_s, sigma_s]

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
    params, cov = curve_fit(djf_model, bin_centers, counts, p0=p0, bounds=((0, 0, 0, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
    fitted = djf_model(bin_centers, *params)
    model_function = djf_model(bin_centers, *p0)
    end = datetime.datetime.now()
    duration = end - start
    print("Fitting completed, execution time: ", duration)
    
    start = datetime.datetime.now()
    print("Calculating model components")
    #fitted model components
    G1 = params[0] * np.exp(-(bin_centers - params[1])**2 / (2*params[2]**2))
    G2 = params[3] * np.exp(-(bin_centers - params[4])**2 / (2*params[5]**2))
    S = s_phase_component(bin_centers, params[1], params[4], params[7], params[6])
    
    #intitial model components
    
    G1_p0 = p0[0] * np.exp(-(bin_centers - p0[1])**2 / (2*p0[2]**2))
    G2_p0 = p0[3] * np.exp(-(bin_centers - p0[4])**2 / (2*p0[5]**2))
    S_p0 = s_phase_component(bin_centers, p0[1], p0[4], p0[7], p0[6])
    
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
    total_area = np.trapezoid(fitted, bin_centers)
    frac_G1 = np.trapezoid(G1, bin_centers) / total_area
    frac_S  = np.trapezoid(S,  bin_centers) / total_area
    frac_G2 = np.trapezoid(G2, bin_centers) / total_area
    
    rows.append([name, frac_G1, frac_S, frac_G2])  
    end = datetime.datetime.now()
    duration = end - start
    print("Calculation completed, execution time: ", duration)
    
def process_image_wrapper(args):
    return process_image(*args)

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    size_of_1px = float(sys.argv[3])
    max_nucleus_area = float(sys.argv[4])
    min_nucleus_area = float(sys.argv[5])
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
        quantify_cell_cycle(filepath, filename, output_dir, rows)
    
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "frac_G1", "frac_S", "frac_G2"])
        results_path = os.path.join(output_dir, "Results.csv")
        df.to_csv(results_path, index=False)
        print(f"Results saved: {len(rows)} samples analyzed")
    else:
        print("No valid data for cell cycle analysis")

if __name__ == "__main__":
    main()

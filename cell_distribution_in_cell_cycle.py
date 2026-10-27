"""
Cell cycle distribution from DAPI microscopy images.

Pipeline
--------
1. Segment nuclei (Gaussian smoothing, Otsu threshold, distance-transform watershed).
2. Measure background-corrected integrated DAPI intensity per nucleus.
3. Fit a Dean-Jett-Fox (DJF) model to the DNA-content distribution and report
   the fractions of cells in G1, S and G2/M.

The DJF model (Dean & Jett 1974; Fox 1980, https://doi.org/10.1002/cyto.990010114)
-----------------------------------------------------------------------------
    G1(x) = n1 * Normal(x; mu1, cv1*mu1)
    G2(x) = n2 * Normal(x; mu2, cv2*mu2),            mu2 = r * mu1, r ~ 2
    S(x)  = integral_{mu1}^{mu2} s(x') * Normal(x; x', cv(x')*x') dx'

s(x') is the DNA-content density of S-phase cells *before* measurement noise.
Dean & Jett use a second-order polynomial; Fox (1980) adds a Gaussian to the
polynomial to describe the S-phase bump of synchronised populations (enable it
with --synchronous). The broadening is applied to s(x') as a whole, so S-phase
cells legitimately spill over the G1 and G2 peaks; the S component is NOT cut
off at mu1 and mu2.

Fitting is done by extended (Poisson) maximum likelihood on the individual
nuclei rather than least squares on a histogram: with the few hundred nuclei
typical of an image, histogram fits depend heavily on the binning. The
histogram is only used for display. The fractions are the areas of the three
components, so they always sum to 1.
"""

import argparse
import datetime
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import gaussian_kde, iqr
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

INTENSITY_COL = "IntDen_bgsub"   # column used for the cell-cycle analysis
BINNED_ABOVE = 1000              # use the binned likelihood above this many nuclei
SQRT_2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _timer(label, start):
    print(f"{label}, execution time: {datetime.datetime.now() - start}")
    return datetime.datetime.now()


def segment_nuclei(img, min_nucleus_area, max_nucleus_area, min_distance):
    """Otsu threshold + distance-transform watershed.

    Returns (labels, labels_all, binary): labels holds the accepted nuclei; labels_all
    holds every segmented object (also the rejected ones), which enlarged masks must
    not grow into; binary is the raw threshold mask, used to find background.
    """
    # Smoothing is used for segmentation only; intensities are measured on the raw image.
    img_proc = skimage.filters.gaussian(img, sigma=0.8, preserve_range=True)
    binary = img_proc > threshold_otsu(img_proc)
    mask = ndi.binary_opening(ndi.binary_fill_holes(binary))

    distance = ndi.gaussian_filter(ndi.distance_transform_edt(mask), 1)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
    seeds = np.zeros(mask.shape, dtype=np.int32)
    seeds[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels_all = watershed(-distance, seeds, mask=mask)

    labels = labels_all.copy()
    # Remove nuclei touching the border
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    labels[np.isin(labels, border[border > 0])] = 0
    # Remove objects outside the accepted area range
    areas = np.bincount(labels.ravel())
    keep = (areas >= min_nucleus_area) & (areas <= max_nucleus_area)
    keep[0] = False
    labels[~keep[labels]] = 0
    return labels, labels_all, binary


def expand_masks(labels, labels_all, n_px):
    """Grow every accepted mask outward by n_px pixels.

    Growth goes only into pixels that belong to no segmented object, so a mask never
    takes pixels from a neighbouring nucleus (or from a rejected object); where two
    masks grow towards each other the gap is shared between them. Small nuclei lose
    part of their blurred edge to the threshold, and including it makes the
    integrated intensity much less dependent on where exactly the threshold fell.
    """
    out = labels.copy()
    free = labels_all == 0
    cross = ndi.generate_binary_structure(2, 1)
    square = np.ones((3, 3), bool)
    for i in range(n_px):
        # alternate 4- and 8-neighbour steps: grows roughly circular, not square
        grown = ndi.grey_dilation(out, footprint=cross if i % 2 == 0 else square)
        take = free & (out == 0) & (grown > 0)
        out[take] = grown[take]
    return out


def measure_nuclei(img, labels, labels_all, binary, expand_px=0):
    """Per-nucleus measurements; intensities summed over masks enlarged by expand_px."""
    # Background: median of pixels well away from any foreground object.
    # Integrated intensity must be background-corrected, otherwise every nucleus
    # gets an offset proportional to its area and the G2/G1 ratio is no longer ~2.
    bg_region = ~ndi.binary_dilation(binary, iterations=max(5, expand_px + 3))
    background = float(np.median(img[bg_region])) if np.any(bg_region) else 0.0

    meas = expand_masks(labels, labels_all, expand_px) if expand_px > 0 else labels
    n_lab = int(labels.max()) + 1
    if n_lab <= 1:
        return pd.DataFrame(), background

    seg_area = np.bincount(labels.ravel(), minlength=n_lab)
    lab = meas.ravel()
    sel = lab > 0
    lab, pix = lab[sel], img.ravel()[sel]
    area = np.bincount(lab, minlength=n_lab)
    rawintden = np.bincount(lab, weights=pix, minlength=n_lab)
    min_vals = np.full(n_lab, np.inf)
    max_vals = np.full(n_lab, -np.inf)
    np.minimum.at(min_vals, lab, pix)
    np.maximum.at(max_vals, lab, pix)
    yy, xx = np.nonzero(labels)
    ll = labels[yy, xx]
    cy = np.bincount(ll, weights=yy, minlength=n_lab) / np.maximum(seg_area, 1)
    cx = np.bincount(ll, weights=xx, minlength=n_lab) / np.maximum(seg_area, 1)

    df = pd.DataFrame({
        "ID": np.arange(n_lab),
        "Y": cy,
        "X": cx,
        "Min": min_vals,
        "Max": max_vals,
        "Mean_int": np.divide(rawintden, area, out=np.zeros(n_lab), where=area > 0),
        "RawIntDen": rawintden,
        "Background": background,
        INTENSITY_COL: rawintden - background * area,
        "Area": seg_area,               # segmented area (used for the area filter)
        "Area_measured": area,          # area after enlargement
    })
    df = df.iloc[1:]                    # row 0 is the background
    return df[df["Area"] > 0], background


def process_image(filepath, name, output_dir, min_nucleus_area, max_nucleus_area,
                  min_distance, expand_px=0):
    """Segment nuclei in one DAPI image and save per-nucleus measurements.

    Areas are in pixels. Returns the path of the per-image CSV, or None.
    """
    t = datetime.datetime.now()
    img = skimage.io.imread(filepath).astype(np.float32, copy=False)
    if img.ndim != 2:
        raise ValueError(f"{name}: expected a single-channel 2D image, got shape {img.shape}")
    t = _timer(f"[{name}] Image loaded", t)

    labels, labels_all, binary = segment_nuclei(img, min_nucleus_area, max_nucleus_area,
                                                min_distance)
    t = _timer(f"[{name}] Segmentation completed", t)

    # The PNG is 800 px wide: plot a subsampled label image (a full 9000 x 9000 label
    # image would need several GB of memory inside matplotlib).
    step = max(1, max(labels.shape) // 2000)
    plt.figure(figsize=(8, 8))
    plt.imshow(labels[::step, ::step], cmap="nipy_spectral", interpolation="nearest",
               extent=(0, labels.shape[1], labels.shape[0], 0))
    plt.title("Nuclei masks")
    plt.savefig(os.path.join(output_dir, f"{name}_mask.png"))
    plt.close()

    df, background = measure_nuclei(img, labels, labels_all, binary, expand_px)
    if df.empty:
        print(f"[{name}] No valid nuclei found, skipping CSV generation")
        return None

    csv_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    t = _timer(f"[{name}] Nuclei intensities calculated ({len(df)} nuclei, masks enlarged by "
               f"{expand_px} px, background {background:.1f})", t)

    values = df[INTENSITY_COL].to_numpy(float)

    # Spatial QC: nucleus intensity relative to the median, at its position in the image.
    # A smooth gradient or blocks (tiles) mean uneven focus or illumination.
    rel = values / np.median(values)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["X"], df["Y"], c=rel, s=1, cmap="viridis", vmin=0.3, vmax=2.5,
                    rasterized=True)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title("Nucleus intensity / median (G1 ~1, G2 ~2)")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(output_dir, f"{name}_spatial.png"))
    plt.close(fig)

    edges = display_bin_edges(values)
    plt.figure(figsize=(8, 8))
    plt.hist(values, bins=edges)
    plt.xlabel("Integrated intensity (background-corrected)")
    plt.ylabel("Cell count")
    plt.title("DNA content histogram")
    plt.savefig(os.path.join(output_dir, f"{name}_histogram.png"))
    plt.close()
    _timer(f"[{name}] Histogram generated", t)
    return csv_path


def process_image_wrapper(args):
    return process_image(*args)


# ---------------------------------------------------------------------------
# Dean-Jett-Fox model
# ---------------------------------------------------------------------------

def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (SQRT_2PI * sd)


def _norm_mass(lo, hi, mu, sd):
    return ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd)


class DJFModel:
    """Dean-Jett-Fox model with an optional Fox (1980) Gaussian term in S phase.

    Intensities are handled in units of `scale` (the initial G1 estimate), so all
    parameters are of order 1 and the optimiser is well conditioned.

    Parameter vector (theta):
        n1, mu1, cv1, n2, r, [cv2], b0, b1, b2, [ns, us, ws]
    n1, n2       : number of cells in G1 and G2/M
    mu1          : G1 mean (units of scale); mu2 = r * mu1
    cv1, cv2     : coefficients of variation of G1 and G2 (cv2 = cv1 unless --free-g2-cv)
    b0, b1, b2   : S-phase polynomial (DNA-content density of S cells, in cells per
                   unit of S progress u = (x - mu1) / (mu2 - mu1)), written in the
                   Bernstein basis  s(u) = b0(1-u)^2 + 2 b1 u(1-u) + b2 u^2.
                   This is the same family of quadratics as A + Bx + Cx^2, but
                   b0, b1, b2 >= 0 guarantees s(u) >= 0 (no negative cell counts),
                   and b0/b2 are the S density at the start/end of S phase.
    ns, us, ws   : Fox Gaussian term ns * Normal(u; us, ws) (synchronous populations)
    nd, td       : debris, nd * exp(-x/td) / td: nuclear fragments, over-segmented or
                   out-of-focus objects. It is not a cell-cycle phase and is excluded
                   from the fractions.
    """

    def __init__(self, synchronous=False, free_g2_cv=False, ratio_bounds=(1.7, 2.3),
                 debris=True, n_quad=80):
        self.synchronous = synchronous
        self.debris = debris
        self.free_g2_cv = free_g2_cv
        self.ratio_bounds = ratio_bounds
        self.u = (np.arange(n_quad) + 0.5) / n_quad    # midpoint rule on [0, 1]
        self.du = 1.0 / n_quad

    # -- parameter handling -------------------------------------------------
    @property
    def names(self):
        names = ["n1", "mu1", "cv1", "n2", "r"]
        if self.free_g2_cv:
            names.append("cv2")
        names += ["b0", "b1", "b2"]
        if self.synchronous:
            names += ["ns", "us", "ws"]
        if self.debris:
            names += ["nd", "td"]
        return names

    def unpack(self, theta):
        p = dict(zip(self.names, theta))
        p.setdefault("cv2", p["cv1"])
        p.setdefault("ns", 0.0)
        p.setdefault("us", 0.5)
        p.setdefault("ws", 0.1)
        p.setdefault("nd", 0.0)
        p.setdefault("td", 0.3)
        return p

    def bounds(self, n_cells):
        b = {
            "n1": (0, 3 * n_cells), "mu1": (0.6, 1.4), "cv1": (0.01, 0.30),
            "n2": (0, 3 * n_cells), "r": self.ratio_bounds, "cv2": (0.01, 0.30),
            "b0": (0, 30 * n_cells), "b1": (0, 30 * n_cells), "b2": (0, 30 * n_cells),
            "ns": (0, 10 * n_cells), "us": (0.0, 1.0), "ws": (0.03, 0.5),
            "nd": (0, 3 * n_cells), "td": (0.05, 0.6),
        }
        return [b[k] for k in self.names]

    # -- model pieces ---------------------------------------------------------
    def _s_nodes(self, p):
        """S-phase density at the quadrature nodes, node positions and widths."""
        u = self.u
        s = p["b0"] * (1 - u) ** 2 + 2 * p["b1"] * u * (1 - u) + p["b2"] * u ** 2
        if p["ns"] > 0:
            s = s + p["ns"] * _norm_pdf(u, p["us"], p["ws"])
        mu1, mu2 = p["mu1"], p["r"] * p["mu1"]
        xk = mu1 + u * (mu2 - mu1)
        sdk = (p["cv1"] + u * (p["cv2"] - p["cv1"])) * xk   # CV interpolated G1 -> G2
        return s * self.du, xk, sdk                          # weights in cells

    def components(self, x, theta):
        """Densities (cells per intensity unit, in units of scale) of G1, S, G2 at x."""
        p = self.unpack(theta)
        x = np.asarray(x, float)
        mu1, mu2 = p["mu1"], p["r"] * p["mu1"]
        g1 = p["n1"] * _norm_pdf(x, mu1, p["cv1"] * mu1)
        g2 = p["n2"] * _norm_pdf(x, mu2, p["cv2"] * mu2)
        wk, xk, sdk = self._s_nodes(p)
        s = _norm_pdf(x[:, None], xk[None, :], sdk[None, :]) @ wk
        return g1, s, g2

    def areas(self, theta):
        """Number of cells in G1, S, G2 (full, untruncated components)."""
        p = self.unpack(theta)
        return p["n1"], self._s_nodes(p)[0].sum(), p["n2"]

    def expected_in_window(self, theta, lo, hi):
        p = self.unpack(theta)
        mu1, mu2 = p["mu1"], p["r"] * p["mu1"]
        wk, xk, sdk = self._s_nodes(p)
        return (p["n1"] * _norm_mass(lo, hi, mu1, p["cv1"] * mu1)
                + p["n2"] * _norm_mass(lo, hi, mu2, p["cv2"] * mu2)
                + wk @ _norm_mass(lo, hi, xk, sdk)
                + p["nd"] * (np.exp(-lo / p["td"]) - np.exp(-hi / p["td"])))

    def debris_density(self, x, theta):
        p = self.unpack(theta)
        return p["nd"] * np.exp(-np.asarray(x, float) / p["td"]) / p["td"]

    def nll(self, theta, x, lo, hi):
        """Extended maximum-likelihood negative log-likelihood (data truncated to [lo, hi])."""
        g1, s, g2 = self.components(x, theta)
        dens = np.maximum(g1 + s + g2 + self.debris_density(x, theta), 1e-300)
        return self.expected_in_window(theta, lo, hi) - np.sum(np.log(dens))

    def expected_in_bins(self, theta, edges):
        """Expected number of nuclei in each bin (exact integral of the model over the bin)."""
        p = self.unpack(theta)
        mu1, mu2 = p["mu1"], p["r"] * p["mu1"]
        g1 = p["n1"] * np.diff(ndtr((edges - mu1) / (p["cv1"] * mu1)))
        g2 = p["n2"] * np.diff(ndtr((edges - mu2) / (p["cv2"] * mu2)))
        wk, xk, sdk = self._s_nodes(p)
        s = np.diff(ndtr((edges[:, None] - xk[None, :]) / sdk[None, :]) @ wk)
        e = np.exp(-edges / p["td"])
        return g1 + g2 + s + p["nd"] * (e[:-1] - e[1:])

    def nll_binned(self, theta, edges, counts):
        """Binned version of nll (Poisson likelihood over fine bins), for large samples.
        With bins much narrower than the peaks it gives the same estimates as nll,
        but the cost no longer grows with the number of nuclei."""
        mu = np.maximum(self.expected_in_bins(theta, edges), 1e-300)
        return mu.sum() - counts @ np.log(mu)


def _narrow_kde(x):
    """KDE with half the Scott bandwidth: the DNA histogram is multimodal, and the
    default bandwidth smears the G1 peak into the S phase."""
    return gaussian_kde(x, bw_method=lambda k: 0.5 * k.scotts_factor())


def find_g1_peak(values):
    """Locate the G1 peak: the dominant mode, unless a sizeable peak sits at ~half of it
    (which happens when G2/M cells outnumber G1 cells)."""
    kde = _narrow_kde(values)
    grid = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 800)
    dens = kde(grid)
    mode = grid[np.argmax(dens)]
    half = (grid > 0.4 * mode) & (grid < 0.62 * mode)
    if np.any(half):
        i = np.argmax(np.where(half, dens, 0))
        is_local_max = 0 < i < len(dens) - 1 and dens[i] >= dens[i - 1] and dens[i] >= dens[i + 1]
        if is_local_max and dens[i] > 0.3 * dens.max():
            return grid[i]
    return mode


def fit_djf(values, synchronous=False, free_g2_cv=False, ratio_bounds=(1.7, 2.3), debris=True):
    """Fit the DJF model to per-nucleus integrated intensities.

    Returns a dict with the fitted model, parameters, fractions and fit window.
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise ValueError(f"only {len(values)} nuclei; at least 30 are needed for a cell-cycle fit")

    scale = find_g1_peak(values)
    if scale <= 0:
        raise ValueError("G1 peak position is not positive; check background correction")
    x_all = values / scale
    n = len(x_all)

    grid = np.linspace(np.percentile(x_all, 0.5), np.percentile(x_all, 99.5), 600)
    dens = _narrow_kde(x_all)(grid) * n                      # cells per unit x

    # G1 width from the left flank of the peak, which is least contaminated by S phase:
    # for a Gaussian, the median distance of the lower half from the mean is 0.674 SD.
    left = 1.0 - x_all[(x_all < 1.0) & (x_all > 0.6)]
    cv0 = float(np.clip(np.median(left) / 0.6745, 0.02, 0.2)) if len(left) >= 5 else 0.07
    h1 = float(dens[np.argmin(np.abs(grid - 1.0))])

    # G2 position: maximum of the density in the expected range
    in_g2 = (grid > ratio_bounds[0]) & (grid < ratio_bounds[1])
    r0 = float(grid[in_g2][np.argmax(dens[in_g2])]) if np.any(in_g2) else 2.0
    h2 = dens[in_g2].max() if np.any(in_g2) else 0.0

    # Fit window: excludes debris / fragments below G1 and aggregates above G2
    lo = max(1.0 - 5.0 * cv0, 0.45)
    hi = ratio_bounds[1] * (1 + 3.5 * cv0)
    x = x_all[(x_all >= lo) & (x_all <= hi)]

    valley = (grid > 1 + 2.5 * cv0) & (grid < r0 * (1 - 2.5 * cv0))
    s_level = float(np.median(dens[valley])) if np.any(valley) else 0.1 * h1
    s_level = s_level * (r0 - 1.0)                           # cells per unit of u

    model = DJFModel(synchronous=synchronous, free_g2_cv=free_g2_cv, ratio_bounds=ratio_bounds,
                     debris=debris)
    bounds = model.bounds(n)

    def start(r_start):
        p = {
            "n1": max(h1 * cv0 * SQRT_2PI, 1.0), "mu1": 1.0, "cv1": cv0,
            "n2": max(h2 * cv0 * r_start * SQRT_2PI, 1.0), "r": r_start, "cv2": cv0,
            "b0": s_level, "b1": s_level, "b2": s_level,
            "ns": s_level, "us": 0.3, "ws": 0.15,
            "nd": 0.02 * n, "td": 0.3,
        }
        th = np.array([p[k] for k in model.names], float)
        return np.clip(th, [b[0] for b in bounds], [b[1] for b in bounds])

    # The optimiser works on rescaled variables: cell counts divided by n, so that every
    # variable is of order 1, and the likelihood per nucleus. Without this, L-BFGS-B
    # stops long before the optimum because the count parameters are ~1000x larger
    # than the CVs.
    amp = np.array([k in ("n1", "n2", "b0", "b1", "b2", "ns", "nd") for k in model.names])
    D = np.where(amp, float(n), 1.0)
    z_bounds = [(b[0] / d, b[1] / d) for b, d in zip(bounds, D)]

    # Above BINNED_ABOVE nuclei the likelihood is evaluated on fine bins (width ~CV/10
    # of G1, i.e. far narrower than any peak) instead of on every nucleus.
    if len(x) > BINNED_ABOVE:
        n_bins = int(np.clip((hi - lo) / (cv0 / 10), 200, 2000))
        edges = np.linspace(lo, hi, n_bins + 1)
        counts = np.histogram(x, bins=edges)[0].astype(float)

        def objective(z):
            return model.nll_binned(z * D, edges, counts) / n
    else:
        def objective(z):
            return model.nll(z * D, x, lo, hi) / n

    # A few starts guard against local minima (e.g. G2 placed on an S-phase shoulder)
    starts = {float(np.clip(r, *ratio_bounds)) for r in (r0, 1.95, 2.0)}
    best = None
    for r_start in sorted(starts):
        th0 = start(r_start)
        res = minimize(objective, th0 / D, method="L-BFGS-B", bounds=z_bounds,
                       options={"maxiter": 20000, "maxfun": 200000, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or res.fun < best[0].fun:
            best = (res, th0)
    res, theta0 = best
    res.x = res.x * D

    a1, aS, a2 = model.areas(res.x)
    total = a1 + aS + a2
    p = model.unpack(res.x)
    return {
        "model": model, "theta": res.x, "theta0": theta0, "scale": scale,
        "window": (lo * scale, hi * scale), "x_window": x * scale, "values": values,
        "converged": bool(res.success), "message": res.message,
        "frac_G1": a1 / total, "frac_S": aS / total, "frac_G2": a2 / total,
        "G1_mean": p["mu1"] * scale, "G2_G1_ratio": p["r"],
        "CV_G1": p["cv1"], "CV_G2": p["cv2"],
        "n_in_window": len(x), "n_outside_window": n - len(x),
    }


def display_bin_edges(values, lo=None, hi=None):
    """Freedman-Diaconis bins for display only, 20-120 bins."""
    values = np.asarray(values, float)
    lo = np.min(values) if lo is None else lo
    hi = np.max(values) if hi is None else hi
    inside = values[(values >= lo) & (values <= hi)]
    h = 2 * iqr(inside) / max(len(inside), 1) ** (1 / 3) if len(inside) > 1 else 0
    n_bins = int(np.clip((hi - lo) / h if h > 0 else 40, 20, 120))
    return np.linspace(lo, hi, n_bins + 1)


def plot_fit(fit, path, which="theta", title="DJF fit"):
    model, scale = fit["model"], fit["scale"]
    theta = fit[which]
    lo, hi = fit["window"]
    x = fit["x_window"]

    edges = display_bin_edges(x, lo, hi)
    width = edges[1] - edges[0]
    counts, _ = np.histogram(x, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    xf = np.linspace(lo, hi, 600)
    g1, s, g2 = (c * width / scale for c in model.components(xf / scale, theta))   # cells per bin
    deb = model.debris_density(xf / scale, theta) * width / scale
    total = g1 + s + g2 + deb

    # expected counts per bin (exact integral via fine grid) for residuals
    fine = np.linspace(lo, hi, 20 * (len(edges) - 1) + 1)
    dens_fine = (sum(model.components(fine / scale, theta))
                 + model.debris_density(fine / scale, theta)) / scale
    cum = np.concatenate([[0], np.cumsum(0.5 * (dens_fine[1:] + dens_fine[:-1]) * np.diff(fine))])
    expected = np.diff(np.interp(edges, fine, cum))
    resid = (counts - expected) / np.sqrt(np.maximum(expected, 1.0))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
    ax.bar(centres, counts, width=width, color="0.8", edgecolor="0.6", label="Data")
    ax.plot(xf, total, "k-", lw=2, label="Model")
    ax.plot(xf, g1, "-", color="#1f77b4", label="G1")
    ax.plot(xf, s, "-", color="#ff7f0e", label="S")
    ax.plot(xf, g2, "-", color="#2ca02c", label="G2/M")
    if model.debris and deb.max() > 0.01 * total.max():
        ax.plot(xf, deb, ":", color="0.4", label="Debris")
    ax.set_ylabel(f"Cells per bin (bin width {width:.3g})")
    ax.set_title(title)

    if which == "theta":
        a1, aS, a2 = model.areas(theta)
        tot = a1 + aS + a2
        chi2 = float(np.sum(resid ** 2))
        dof = max(len(counts) - len(theta), 1)
        text = (f"G1 {a1 / tot:.1%}   S {aS / tot:.1%}   G2/M {a2 / tot:.1%}\n"
                f"G2/G1 = {fit['G2_G1_ratio']:.3f}   CV(G1) = {fit['CV_G1']:.1%}\n"
                f"n = {fit['n_in_window']} in window ({fit['n_outside_window']} outside)\n"
                f"Pearson chi2/dof = {chi2 / dof:.2f}")
        ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="upper center")

    axr.axhline(0, color="k", lw=0.8)
    axr.bar(centres, resid, width=width, color="0.5")
    axr.set_ylabel("Residual (SD)")
    axr.set_xlabel("Integrated DAPI intensity (background-corrected)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def quantify_cell_cycle(filepath, name, output_dir, synchronous=False, free_g2_cv=False,
                        ratio_bounds=(1.7, 2.3)):
    """Fit the DJF model to one per-image CSV. Returns a dict of results or None."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    col = INTENSITY_COL if INTENSITY_COL in df.columns else "RawIntDen"
    if col not in df.columns:
        print(f"No intensity column in {filepath}, skipping")
        return None
    if col == "RawIntDen":
        print(f"Warning: {filepath} has no background-corrected intensities; using RawIntDen")

    values = df[col].astype(float).dropna().to_numpy()
    t = datetime.datetime.now()
    try:
        fit = fit_djf(values, synchronous=synchronous, free_g2_cv=free_g2_cv,
                      ratio_bounds=ratio_bounds)
    except Exception as e:
        print(f"[{name}] Model fit failed: {e}")
        return None
    t = _timer(f"[{name}] Model fitted ({'converged' if fit['converged'] else 'NOT converged: ' + str(fit['message'])})", t)

    plot_fit(fit, os.path.join(output_dir, f"{name}_model.png"), "theta", "Fitted DJF model")
    plot_fit(fit, os.path.join(output_dir, f"{name}_initial_model.png"), "theta0",
             "Initial model (starting values)")
    _timer(f"[{name}] Plots generated", t)

    return {k: fit[k] for k in ("frac_G1", "frac_S", "frac_G2", "G1_mean", "G2_G1_ratio",
                                "CV_G1", "CV_G2", "n_in_window", "n_outside_window",
                                "converged")}


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------

def quantify_cell_cycle_wrapper(args):
    """Runs in a worker process: fit one sample and return its row of Results.csv."""
    filepath, name, output_dir, synchronous, free_g2_cv, ratio_bounds = args
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
    n_cells = len(pd.read_csv(filepath))
    res = quantify_cell_cycle(filepath, name, output_dir, synchronous=synchronous,
                              free_g2_cv=free_g2_cv, ratio_bounds=ratio_bounds)
    return {"ID": name, "n_cells": n_cells, **(res or {})}


def plot_cell_cycle(results_csv_path, output_dir):
    df = pd.read_csv(results_csv_path)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df) + 2), 6))
    g1, s, g2 = (df[c].fillna(0) for c in ("frac_G1", "frac_S", "frac_G2"))
    ax.bar(x, g1, label="G1", color="#1f77b4")
    ax.bar(x, s, bottom=g1, label="S", color="#ff7f0e")
    ax.bar(x, g2, bottom=g1 + s, label="G2/M", color="#2ca02c")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction of cells")
    ax.set_title("Cell cycle phase distribution by sample")
    ax.set_xticks(x)
    ax.set_xticklabels(df["ID"].astype(str), rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "Cell_cycle_barplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Stacked bar plot saved: {plot_path}")


def build_report(results_df, output_dir, pixel_size, min_area, max_area, expand_px=0):
    try:
        import datapane as dp
    except ImportError:
        print("datapane not installed; HTML report skipped (all PNG and CSV outputs are saved).")
        return

    def sample_group(row, label=None):
        sid = row["ID"]
        kwargs = {"label": label} if label else {}
        return dp.Group(
            dp.Text(f"### Sample: {sid}"),
            dp.Text(f"Total nuclei: **{int(row['n_cells'])}**"),
            dp.Media(file=os.path.join(output_dir, f"{sid}_mask.png"), caption="Nuclei masks"),
            dp.Media(file=os.path.join(output_dir, f"{sid}_spatial.png"), caption="Intensity across the image"),
            dp.Media(file=os.path.join(output_dir, f"{sid}_histogram.png"), caption="DAPI intensity histogram"),
            dp.Media(file=os.path.join(output_dir, f"{sid}_initial_model.png"), caption="Initial model"),
            dp.Media(file=os.path.join(output_dir, f"{sid}_model.png"), caption="Fitted model"),
            **kwargs,
        )

    fitted = results_df[results_df["frac_G1"].notna()]
    if len(fitted) == 0:
        blocks = [dp.Text("No samples could be fitted")]
    elif len(fitted) == 1:
        blocks = [sample_group(fitted.iloc[0])]
    else:
        blocks = [dp.Select(blocks=[sample_group(r, str(r["ID"])) for _, r in fitted.iterrows()],
                            type=dp.SelectType.DROPDOWN)]

    report = dp.Report(
        dp.Text("# Cell cycle analysis report"),
                dp.Text(f"Pixel size: **{pixel_size:g}** µm, nuclear area range: **{min_area:g}–{max_area:g}** µm², masks enlarged by **{expand_px}** px"),
        dp.Text("## Summary"),
        dp.Media(file=os.path.join(output_dir, "Cell_cycle_barplot.png")),
        dp.Table(results_df),
        dp.Text("## Per-sample details"),
        *blocks,
    )
    report.save(path=os.path.join(output_dir, "report.html"), open=True)
    print("Report saved: report.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Cell cycle distribution from DAPI images (DJF model).")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("pixel_size", type=float, help="pixel size in um")
    ap.add_argument("min_area", type=float, help="minimum nuclear area in um^2")
    ap.add_argument("max_area", type=float, help="maximum nuclear area in um^2")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel workers for image processing and model fitting (default 4)")
    ap.add_argument("--min-distance", type=int, default=None,
                    help="watershed seed spacing in pixels (default: derived from min_area)")
    ap.add_argument("--expand", type=int, default=0,
                    help="enlarge each nuclear mask by this many pixels before summing the "
                         "intensity (default 0). Useful when nuclei are only a few pixels wide.")
    ap.add_argument("--synchronous", action="store_true",
                    help="add Fox's Gaussian term to S phase (synchronised populations)")
    ap.add_argument("--free-g2-cv", action="store_true",
                    help="fit the G2 CV separately instead of setting it equal to the G1 CV")
    ap.add_argument("--ratio", type=float, nargs=2, default=(1.7, 2.3), metavar=("MIN", "MAX"),
                    help="allowed range of the G2/G1 intensity ratio (default 1.7 2.3)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # um^2 -> pixels: one pixel covers pixel_size^2 um^2
    px_area = args.pixel_size ** 2
    min_px = args.min_area / px_area
    max_px = args.max_area / px_area
    # Seeds closer than ~half the radius of the smallest accepted nucleus are merged
    min_distance = args.min_distance or max(3, int(round(0.5 * np.sqrt(min_px / np.pi))))
    print(f"Accepted nuclear area: {min_px:.0f}-{max_px:.0f} px; watershed min_distance = {min_distance} px")

    tif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif"))
                       + glob.glob(os.path.join(args.input_dir, "*.tiff")))
    tasks = [(f, os.path.splitext(os.path.basename(f))[0], args.output_dir, min_px, max_px,
              min_distance, args.expand) for f in tif_files]

    print(f"Processing {len(tasks)} images...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        csv_files = [c for c in executor.map(process_image_wrapper, tasks) if c]

    print(f"Quantifying cell cycle phases ({len(csv_files)} samples, {args.workers} workers)...")
    fit_tasks = [(f, os.path.splitext(os.path.basename(f))[0], args.output_dir,
                  args.synchronous, args.free_g2_cv, tuple(args.ratio)) for f in csv_files]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(quantify_cell_cycle_wrapper, fit_tasks))

    if not rows:
        print("No valid data for cell cycle analysis")
        return

    cols = ["ID", "n_cells", "frac_G1", "frac_S", "frac_G2", "G1_mean", "G2_G1_ratio",
            "CV_G1", "CV_G2", "n_in_window", "n_outside_window", "converged"]
    results_df = pd.DataFrame(rows).reindex(columns=cols)
    results_path = os.path.join(args.output_dir, "Results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved: {len(results_df)} samples, {results_df['frac_G1'].notna().sum()} fitted")
    plot_cell_cycle(results_path, args.output_dir)
    build_report(results_df, args.output_dir, args.pixel_size, args.min_area, args.max_area,
                 args.expand)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
    main()

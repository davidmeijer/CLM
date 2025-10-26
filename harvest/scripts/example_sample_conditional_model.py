#!/usr/bin/env python3

"""Sample conditional model with RetroMol fingerprint."""

import argparse
import os
import re
from typing import List, Generator, Tuple, Optional

from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from networkx import radius
import torch
import yaml
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from rdkit import Chem, RDLogger
from numpy.typing import NDArray
from tqdm import tqdm
from retromol.api import run_retromol_with_timeout
from retromol.fingerprint import (
    FingerprintGenerator,
    NameSimilarityConfig,
    polyketide_family_of,
    cosine_similarity
)
from retromol.io import Result, Input as RetroMolInput
from retromol.chem import Mol, smiles_to_mol, mol_to_smiles, mol_to_fpr
from retromol.streaming import run_retromol_stream, stream_table_rows
from retromol.rules import (
    get_path_default_matching_rules,
    get_path_default_reaction_rules,
    get_path_default_wave_config,
    load_rules_from_files,
)

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from clm.datasets import Vocabulary
from clm.models import ConditionalRNN


RDLogger.DisableLog('rdApp.*')


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sdf", type=str, required=True, help="Path to SDF file with background compounds")
    parser.add_argument("--force", action="store_true", help="Force re-computation of samples")
    parser.add_argument("--retromol", type=str, default=None, help="Path to saved RetroMol results for generated samples (path/to/JSONL)")
    return parser.parse_args()


def load_models(
    data_dir: str,
    train_dataset_name: str,
    enum_factor: int,
    rnn_type: str,
    n_layers: int,
    embedding_size: int,
    hidden_size: int,
    dropout: float,
    device: torch.device,
) -> List[ConditionalRNN]:
    """
    Load RNN models from the specified directory.

    :param modeldir: Directory containing model files
    :param train_dataset_name: Name of the training dataset
    :param enum_factor: Enumeration factor for model selection
    :param device: Device to load the models onto
    :param rnn_type: Type of RNN ("GRU" or "LSTM")
    :param n_layers: Number of RNN layers
    :param embedding_size: Size of the embedding layer
    :param hidden_size: Size of the hidden layers
    :param dropout: Dropout rate
    :return: List of loaded RNN models
    """
    assert rnn_type in ["GRU", "LSTM"], "rnn_type must be either 'GRU' or 'LSTM'"

    models: List[ConditionalRNN] = []

    # Find all vocab and model files
    vocab_dir_path = os.path.join(data_dir, f"{enum_factor}", "prior", "inputs")
    model_dir_path = os.path.join(data_dir, f"{enum_factor}", "prior", "models")

    vocab_file_pattern = re.compile(f"train_{train_dataset_name}_SMILES_\d+.vocabulary")
    model_file_pattern = re.compile(f"{train_dataset_name}_SMILES_\d+_\d+_model.pt")

    try:
        vocab_files = [f for f in os.listdir(vocab_dir_path) if vocab_file_pattern.match(f)]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Vocabulary directory not found: {vocab_dir_path}") from e

    try:
        model_files = [f for f in os.listdir(model_dir_path) if model_file_pattern.match(f)]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model directory not found: {model_dir_path}") from e
    
    assert len(vocab_files) == len(model_files), "Number of vocabulary files must match number of model files."

    # Match vocab files to model files based on the split number
    vocab_split_pattern = re.compile(f"train_{train_dataset_name}_SMILES_(\d+).vocabulary")
    model_split_pattern = re.compile(f"{train_dataset_name}_SMILES_(\d+)_\d+_model.pt")

    # Find the split numbers
    found_vocab_splits = {int(vocab_split_pattern.match(f).group(1)): f for f in vocab_files}
    found_model_splits = {int(model_split_pattern.match(f).group(1)): f for f in model_files}

    # Check if keys (split numbers) perfectly overlap
    overlapping_splits = set(found_vocab_splits.keys()).intersection(set(found_model_splits.keys()))
    assert len(overlapping_splits) == len(found_vocab_splits) == len(found_model_splits), "Mismatch in splits between vocab and model files."

    # Create mapping: split number -> (vocab file, model file)
    split_to_files = {
        split: {
            "vocab": found_vocab_splits[split], 
            "model": found_model_splits[split]
        } 
        for split in overlapping_splits
    }

    for _, files in sorted(split_to_files.items()):
        vocab_path = os.path.join(vocab_dir_path, files["vocab"])
        model_path = os.path.join(model_dir_path, files["model"])

        # Load vocabulary
        vocab = Vocabulary(vocab_file=vocab_path)

        # Load model
        model = ConditionalRNN(
            vocabulary=vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=512,
        )
        
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()

        models.append(model)
    
    return models


def get_fingerprint_generator() -> FingerprintGenerator:
    """
    Create and configure a FingerprintGenerator for polyketides.

    :return: configured FingerprintGenerator instance
    """
    path_default_matching_rules = get_path_default_matching_rules()
    collapse_by_name = [
        "glycosylation",
        "methylation",
        "chlorination",
        "bromination",
        "fluorination",
        "oxidation",
        "boronation",
        "amination",
    ]
    cfg = NameSimilarityConfig(family_of=polyketide_family_of, symmetric=True, family_repeat_scale=1)
    generator = FingerprintGenerator(
        matching_rules_yaml=path_default_matching_rules,
        collapse_by_name=collapse_by_name,
        name_similarity=cfg
    )
    return generator


def calc_fingerprint(generator: FingerprintGenerator, result: Result, num_bits: int, counted: bool) -> NDArray[np.int_]:
    """
    Calculate the fingerprint for a given Retromol result.

    :param generator: fingerprintGenerator instance
    :param result: Retromol Result object
    :param num_bits: number of bits in the fingerprint
    :param counted: whether to use counted fingerprints
    :return: fingerprint as a NumPy array (N, num_bits)
    """
    # TODO: check kmers_sizes and kmer_weights settings
    fp = generator.fingerprint_from_result(result, num_bits=num_bits, counted=counted)
    return fp


def get_all_kmers(seq, ks=(1, 2, 3)):
    """
    Return a flat list of unique k-mers (tuples of (name, SMILES)) 
    from both forward and reverse directions.
    """
    def kmers(s, k):
        return [tuple(s[i:i+k]) for i in range(len(s) - k + 1)]
    
    forward = [kmer for k in ks for kmer in kmers(seq, k)]
    reverse = [kmer for k in ks for kmer in kmers(list(reversed(seq)), k)]
    
    # Deduplicate while preserving order
    seen, unique = set(), []
    for kmer in forward + reverse:
        if kmer not in seen:
            seen.add(kmer)
            unique.append(kmer)

    return unique


def mols_from_sdf_field(sdf_path: str,) -> Generator[NDArray[np.int_], None, None]:
    """
    Iterate over an SDF and yield RDKit Mol objects.
    """
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    for mol_idx, mol in enumerate(suppl):
        if mol is not None:
            fp = mol_to_fpr(mol, rad=2, nbs=2048)
            yield fp
        # if mol_idx >= 1000:
        #     break  # for testing


def hdr_region_2d(points_xy: np.ndarray,
                  mass: float = 0.50,
                  grid_n: int = 256) -> dict:
    """
    Compute a 2D highest-density region (HDR) for 'points_xy' (N,2).
    Returns:
      - center: (x0, y0) KDE mode (argmax on a grid)
      - thr: density threshold so that {f >= thr} has ≈ 'mass' probability
      - area: estimated area of HDR (from grid)
      - radius: equivalent-circle radius = sqrt(area/pi)
      - idx_in_hdr: indices of input points that fall inside HDR
      - kde: fitted gaussian_kde callable
      - grid: (Xg, Yg, Z) density grid for optional plotting
    """
    assert 0 < mass < 1, "mass must be in (0,1)"
    xy = points_xy.T  # shape (2, N) for gaussian_kde
    kde = gaussian_kde(xy)

    # grid limits (robust to outliers)
    x, y = points_xy[:, 0], points_xy[:, 1]
    x_min, x_max = np.percentile(x, [1, 99])
    y_min, y_max = np.percentile(y, [1, 99])
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    y_pad = 0.05 * (y_max - y_min + 1e-12)
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # mode (grid argmax)
    idx_max = np.argmax(Z)
    mode_j, mode_i = np.unravel_index(idx_max, Z.shape)  # (row, col)
    x0, y0 = Xg[mode_j, mode_i], Yg[mode_j, mode_i]

    # find threshold thr so that integral over {Z >= thr} ~ mass
    z_sorted = np.sort(Z.ravel())[::-1]
    probs = z_sorted / z_sorted.sum()  # proportional; normalize on grid
    cumsum = np.cumsum(probs)
    # first index where cumulative mass exceeds target
    k = np.searchsorted(cumsum, mass)
    thr = z_sorted[min(k, len(z_sorted) - 1)]

    mask_hdr = (Z >= thr)
    # area estimate from grid cell area
    cell_area = (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
    area = mask_hdr.sum() * cell_area
    radius = np.sqrt(area / np.pi)

    # which original points lie inside HDR (by KDE threshold at their coordinates)
    z_points = kde(points_xy.T)  # density at each sample point
    # find numeric value of 'thr' in the same units as z_points:
    # thr is from grid Z, which is directly comparable to kde values.
    # We'll consider a point inside if its density >= thr.
    idx_in_hdr = np.where(z_points >= thr)[0]

    return {
        "center": (float(x0), float(y0)),
        "thr": float(thr),
        "area": float(area),
        "radius": float(radius),
        "idx_in_hdr": idx_in_hdr,
        "kde": kde,
        "grid": (Xg, Yg, Z),
    }


def nearest_neighbors_in_embedding(
    embedding: np.ndarray,
    target_coord: Tuple[float, float],
    labels: Optional[np.ndarray] = None,
    label_filter: Optional[int] = None,
    k: int = 10,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve nearest neighbors (indices and distances) from a 2D embedding around a target coordinate.

    Parameters
    ----------
    embedding : np.ndarray
        2D array of shape (N, 2), e.g., PCA coordinates.
    target_coord : tuple
        (x, y) coordinate around which to find neighbors.
    labels : np.ndarray, optional
        Optional array of integer labels for filtering (e.g., 0=background, 1=samples, 2=target).
    label_filter : int, optional
        If provided, restrict search to points with this label.
    k : int
        Number of neighbors to return (ignored if radius is given).
    radius : float, optional
        If provided, return all neighbors within this distance from target_coord.

    Returns
    -------
    indices : np.ndarray
        Indices (in `embedding`) of the nearest neighbors.
    distances : np.ndarray
        Corresponding Euclidean distances (sorted ascending).

    Example
    -------
    >>> idxs, dists = nearest_neighbors_in_embedding(embedding, (0.1, -0.3), labels=lbs_all, label_filter=1, k=5)
    >>> neighbors = [sampled[i - len(background_fps)] for i in idxs]
    """
    # Filter subset if label_filter given
    if labels is not None and label_filter is not None:
        mask = (labels == label_filter)
        points = embedding[mask]
        base_idxs = np.where(mask)[0]
    else:
        points = embedding
        base_idxs = np.arange(len(embedding))

    # Compute distances to target
    xy = np.asarray(target_coord)
    dists = np.linalg.norm(points - xy, axis=1)

    if radius is not None:
        # All neighbors within radius
        sel = np.where(dists <= radius)[0]
    else:
        # k nearest neighbors
        sel = np.argsort(dists)[:k]

    return base_idxs[sel], dists[sel]


def main() -> None:
    """
    Main function to sample from the conditional model.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    # Parse sdf
    save_path_background_fps = os.path.join(args.outdir, "background_fps.npy")
    if not os.path.exists(save_path_background_fps):
        background_fps = []
        for fp in tqdm(mols_from_sdf_field(args.sdf), desc="Parsing SDF file"):
            background_fps.append(fp)
        background_fps = np.array(background_fps)
        np.save(save_path_background_fps, background_fps)
    else:
        background_fps = np.load(save_path_background_fps)
    print(f"Parsed {len(background_fps)} compounds from SDF file with fingerprints of shape {background_fps.shape}")

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(
        data_dir=args.modeldir,
        train_dataset_name="retromol_512",
        enum_factor=10,
        rnn_type="LSTM",
        n_layers=3,
        embedding_size=128,
        hidden_size=1024,
        dropout=0,
        device=device,
    )
    print(f"Loaded {len(models)} models")

    # Set config fingerprinting
    num_bits = 512
    counted = False
    generator = get_fingerprint_generator()

    # Parse target compound
    smiles = r"O=C(C(C(C)C)NC(C(NC(C(CC(O)=O)NC1=O)=O)CC2=CC=C(O)C=C2)=O)NC(CC(N)=O)C(NC(CCC(O)=O)C(OCC(NC(C(C(C)C)NC(CC(O)CCCCCCCCCC)=O)=O)C(NCC(NC(C(NC1C(C)O)=O)CC3=CC=C(O)C=C3)=O)=O)=O)=O"
    # smiles = r"CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O"
    compound_fp_morgan = mol_to_fpr(smiles_to_mol(smiles), rad=2, nbs=2048).reshape(1, -1)
    print(f"Calculated Morgan fingerprint of shape {compound_fp_morgan.shape}")
    inp = RetroMolInput("target", smiles)
    result = run_retromol_with_timeout(inp)
    coverage = result.best_total_coverage()
    print(f"Parsed compound with {coverage:.2%} coverage")

    # Turn parsed compound into fingerprint
    compound_fps_retro = calc_fingerprint(generator, result, num_bits=num_bits, counted=counted)
    print(f"Calculated fingerprint of shape {compound_fps_retro.shape}")

    # Get fingerprint for candidate cluster
    cand_cluster_readout = [
        # ("3-hydroxytridecanoic acid", r"O=C(O)CC(O)CCCCCCCCCC"),
        # ("decanoic acid", r"O=C(O)CCCCCCCCC"),
        # ("D1", r"O=C(O)CCSO"),
        # ("D1", r"O=C(O)CCSO"),
        # ("D1", r"O=C(O)CCSO"),
        # ("D1", r"O=C(O)CCSO"),
        # ("D1", r"O=C(O)CCSO"),
        # ("D1", r"O=C(O)CCSO"),

        # pos 1
        ("valine", r"CC(C)[C@@H](C(=O)O)N"),
        # pos 2
        ("serine", r"C([C@@H](C(=O)O)N)O"),
        # pos 3
        ("alanine", r"C[C@@H](C(=O)O)N"),
        # ("glycine", r"C(C(=O)O)N"),
        # pos 4
        ("tyrosine", r"C1=CC(=CC=C1C[C@@H](C(=O)O)N)O"),
        # pos 5
        ("4R-E-butenyl-4R-methylthreonine", r"C/C=C/C[C@@H](C)[C@H]([C@@H](C(=O)O)N)O"),
        # ("threonine", r"C[C@H]([C@@H](C(=O)O)N)O"),
        # pos 6
        ("aspartic acid", r"C([C@@H](C(=O)O)N)C(=O)O"),
        # pos 7
        ("tyrosine", r"C1=CC(=CC=C1C[C@@H](C(=O)O)N)O"),
        # pos 8
        ("valine", r"CC(C)[C@@H](C(=O)O)N"),
        # pos 9
        ("asparagine", r"C([C@@H](C(=O)O)N)C(=O)N"),
        # pos 10
        ("aspartic acid", r"C([C@@H](C(=O)O)N)C(=O)O"),
        # ("glutamic acid", r"C(CC(=O)O)[C@@H](C(=O)O)N"),
    ]
    kmers = get_all_kmers(cand_cluster_readout, ks=(1,2,3))
    print(f"Extracted {len(kmers)} unique k-mers from candidate cluster")
    cand_cluster_fp = generator.fingerprint_from_kmers(kmers=kmers, num_bits=num_bits, counted=counted).reshape(1, -1)
    print(f"Calculated candidate cluster fingerprint of shape {cand_cluster_fp.shape}")

    # Calculate cosine similarity between compound and candidate cluster
    for i, compound_fp_retro in enumerate(compound_fps_retro):
        cosine_sim = cosine_similarity(compound_fp_retro, cand_cluster_fp)
        print(f"Cosine similarity between compound (fp_idx={i+1}) and candidate cluster: {cosine_sim:.4f}")

    # Samples
    save_path_sample_fps = os.path.join(args.outdir, "sampled_fps.npy")
    sampled_smiles_path = os.path.join(args.outdir, "sampled_smiles.txt")
    if args.force or (not os.path.exists(save_path_sample_fps) or not os.path.exists(sampled_smiles_path)):
        sampled = []
        n_seqs = 10000
        for model in models:
            with torch.inference_mode():
                with torch.no_grad():
                    model.eval()
                    descriptors = torch.tensor(cand_cluster_fp, device=device, dtype=torch.float32).repeat(n_seqs, 1)
                    samples = model.sample(
                        descriptors=descriptors,
                        n_sequences=n_seqs,
                        max_len=250,
                        return_smiles=True,
                        return_losses=False,
                    )
                    for sample in tqdm(samples, desc="Sampling sequences", leave=False):
                        try:
                            mol = smiles_to_mol(sample)
                            smiles = mol_to_smiles(mol)
                        except:
                            continue
                        sampled.append(smiles)
        print(f"Generated {len(sampled)}/{n_seqs*len(models)} valid samples from conditional models")

        # Calculate fingerprints for sampled compounds
        sampled_fps_morgan = np.array([mol_to_fpr(smiles_to_mol(smi), rad=2, nbs=2048) for smi in sampled])
        print(f"Calculated fingerprints for sampled compounds of shape {sampled_fps_morgan.shape}")
        np.save(save_path_sample_fps, sampled_fps_morgan)
        # save sampled smiles to txt file
        with open(sampled_smiles_path, "w") as f:
            for smi in sampled:
                f.write(smi + "\n")
    else:
        sampled_fps_morgan = np.load(save_path_sample_fps)
        with open(sampled_smiles_path, "r") as f:
            sampled = [line.strip() for line in f.readlines()]
    print(f"Loaded sampled fingerprints of shape {sampled_fps_morgan.shape}")

    # Concate data, create labels
    fps_all = np.vstack([background_fps, sampled_fps_morgan, compound_fp_morgan])
    lbs_all = np.array([0] * len(background_fps) + [1] * len(sampled_fps_morgan) + [2] * len(compound_fp_morgan))  # 0: background, 1: samples, 2: original
    print(f"Combined fingerprint array shape: {fps_all.shape}, labels shape: {lbs_all.shape}")

    # PCA and plot
    model = PCA(n_components=2)
    embedding = model.fit_transform(fps_all)
    ev = model.explained_variance_ratio_

    # Plot density of samples, plot original compound as scatter
    lbl_to_name = {0: "background (NPAtlas)", 1: "sampled (Harvest)", 2: "target"}
    plt.figure(figsize=(8, 6))

    lbl = 0
    idxs = np.where(lbs_all == lbl)
    x, y = embedding[idxs, 0], embedding[idxs, 1]
    hb = plt.hexbin(x, y, gridsize=40, bins='log', mincnt=1, cmap='Greys', linewidths=0.0, zorder=1, alpha=0.5)
    cbar = plt.colorbar(hb, pad=0.01)
    cbar.set_label(r'$\log_{10}(\mathrm{count})$', fontsize=12)

    lbl = 1
    color = "#1f77b4"
    idxs = np.where(lbs_all == lbl)
    x, y = embedding[idxs, 0], embedding[idxs, 1]
    sns.kdeplot(x=x.flatten(), y=y.flatten(), fill=False, alpha=1.0, levels=20, color=color, label=lbl_to_name[lbl], linewidths=1.5, linestyles='solid')

    lbl = 2
    color = "#2ca02c"
    idxs = np.where(lbs_all == lbl)
    x, y = embedding[idxs, 0], embedding[idxs, 1]
    plt.scatter(x, y, color=color, s=200, edgecolor="black", marker="X", label=lbl_to_name[lbl], zorder=5)

    # create proxy artists for legend
    background_patch = mpatches.Patch(color='lightgray', alpha=0.5, label=lbl_to_name[0])
    samples_patch = mpatches.Patch(color="#1f77b4", alpha=1.0, label=lbl_to_name[1])
    target_marker = Line2D(
        [0], [0],
        marker='X', color='w', markerfacecolor="#2ca02c",
        markeredgecolor='black', markersize=12, label=lbl_to_name[2]
    )

    plt.legend(
        handles=[background_patch, samples_patch, target_marker],
        frameon=True,
        fontsize=12,
        loc='upper right'
    )

    plt.xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=14)
    plt.ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=14)
    # plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sampled_space_conditional_model.png"), dpi=300)
    plt.close()

    # Calculate highest density point sampled fps
    mask_samples = (lbs_all == 1)
    pts = embedding[mask_samples, :2]  # (Ns, 2)
    hdr = hdr_region_2d(pts, mass=0.50, grid_n=256)
    (x0, y0), radius = hdr["center"], hdr["radius"]
    idx_in_hdr_local = hdr["idx_in_hdr"]  # indices w.r.t. 'pts'
    idx_in_hdr_global = np.where(mask_samples)[0][idx_in_hdr_local]  # indices w.r.t. 'embedding'
    k = 10
    d2 = np.sum((pts - np.array([x0, y0]))**2, axis=1)
    rep_local = np.argsort(d2)[:k]
    rep_global = np.where(mask_samples)[0][rep_local]
    print(f"HDR center (mode) in PCA: ({x0:.4f}, {y0:.4f})")
    print(f"HDR (mass=50%) equivalent radius: {radius:.4f}")
    print(f"Samples inside HDR: {len(idx_in_hdr_global)} / {pts.shape[0]}")
    print("Representative (nearest-to-center) sample indices (global):", rep_global.tolist())
    rep_smiles = [sampled[i - len(background_fps)] for i in rep_global]
    # print("Representative SMILES (k-nearest to HDR center):")
    # for s_idx, s in enumerate(rep_smiles):
    #     print(f" {s_idx+1:2d}. {s}")

    # Get nearest 10 sampled neighbors around target
    x0, y0 = embedding[-1, 0], embedding[-1, 1]  # last point is target
    idxs, dists = nearest_neighbors_in_embedding(
        embedding,
        target_coord=(x0, y0),
        labels=lbs_all,
        label_filter=1,  # restrict to sampled compounds
        k=10
    )
    # print("Nearest neighbors to target center:")
    # for i, (idx, dist) in enumerate(zip(idxs, dists), 1):
    #     smi = sampled[idx - len(background_fps)]  # adjust offset because background are first
    #     print(f"{i:2d}. idx={idx}, dist={dist:.4f}, SMILES={smi}")

    # read in sampled_smiles, and write out again with compound_id column and smiles column
    sampled_smiles = pd.DataFrame({
        "compound_id": range(len(sampled)),
        "smiles": sampled
    })
    sampled_smiles.to_csv(os.path.join(args.outdir, "sampled_smiles.csv"), index=False)

    # parse sampled compounds with RetroMol and calculatre retromol fingerprints
    # first put smiles into dataframe with identifier column
    if args.retromol:
        from retromol.helpers import iter_json
        from retromol.io import Result

        retro_fps = []
        retro_smi = []

        kmers = get_all_kmers(cand_cluster_readout, ks=(1,2,3))
        print(f"Extracted {len(kmers)} unique k-mers from candidate cluster")
        cand_cluster_fp = generator.fingerprint_from_kmers(kmers=kmers, num_bits=num_bits, counted=False).reshape(1, -1)
        print(f"Calculated candidate cluster fingerprint of shape {cand_cluster_fp.shape}")

        for d in iter_json(args.retromol, jsonl=True):
            r = Result.from_serialized(d["result"])
            fps = calc_fingerprint(generator, r, num_bits=num_bits, counted=False)
            if fps is not None:
                smi = r.get_input_smiles(remove_tags=True)
                for i in range(fps.shape[0]):
                    retro_fps.append(fps[i])
                    retro_smi.append(smi)

        retro_fps = np.vstack(retro_fps)
        retro_smi = np.array(retro_smi)
        print(f"Parsed {len(retro_smi)} RetroMol fingerprints from saved results of shape {retro_fps.shape}")

        # calculate cosine similarity between calc fingerprints and cand_cluster_fp, get top 10
        scored_smiles = []
        for i, compound_fp_retro in enumerate(retro_fps):
            sim = cosine_similarity(compound_fp_retro, cand_cluster_fp)
            scored_smiles.append((retro_smi[i], sim))

        scored_smiles = sorted(scored_smiles, key=lambda x: x[1], reverse=True)
        print("Top 10 RetroMol scored sampled compounds by cosine similarity to candidate cluster:")
        for i in range(20):
            smi, sim = scored_smiles[i]
            print(f"{i+1:2d}. sim={sim:.4f}, SMILES={smi}")

if __name__ == "__main__":
    main()

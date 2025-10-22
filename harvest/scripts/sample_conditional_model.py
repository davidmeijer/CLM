#!/usr/bin/env python3

"""Sample conditional model with RetroMol fingerprint."""

import argparse
import os
import re
from typing import List

from matplotlib.legend_handler import HandlerPathCollection
import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from retromol.api import run_retromol_with_timeout
from retromol.fingerprint import (
    FingerprintGenerator,
    NameSimilarityConfig,
    polyketide_family_of
)
from retromol.io import Result, Input as RetroMolInput
from retromol.rules import get_path_default_matching_rules
from retromol.chem import smiles_to_mol, mol_to_smiles, mol_to_fpr

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from clm.datasets import Vocabulary
from clm.models import ConditionalRNN


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")
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
    fp = generator.fingerprint_from_result(result, num_bits=num_bits, counted=counted)
    return fp


def main() -> None:
    """
    Main function to sample from the conditional model.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)
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

    def get_samples(smiles):
        # Parse SMILES of compound and regenerate compounds from its fingerprint
        num_bits = 512
        counted = False
        inp = RetroMolInput("erythromycin", smiles)
        result = run_retromol_with_timeout(inp)
        print(result.best_total_coverage())
        fp_gen = get_fingerprint_generator()
        fp_vector = calc_fingerprint(fp_gen, result, num_bits=num_bits, counted=counted)
        if fp_vector.shape[0] > 1: # pick first
            fp_vector = fp_vector[0:1, :]
        add_fp_vector = fp_gen.fingerprint_from_kmers(
            kmers=[[("chlorination", r"Cl"),]],
            num_bits=num_bits,
            counted=counted,
        )
        # add_fp_vector = fp_gen.fingerprint_from_kmers(
        #     kmers=[[(None, r"C([C@@H](C(=O)O)N)S"),]],
        #     num_bits=num_bits,
        #     counted=counted,
        # )
        fp_vector += add_fp_vector
        print(fp_vector.shape)
        samples = []
        n_seqs = 500
        for model in models:
            with torch.inference_mode():
                with torch.no_grad():
                    model.eval()
                    descriptors = torch.tensor(fp_vector, device=device, dtype=torch.float32).repeat(n_seqs, 1)
                    sampled = model.sample(
                        descriptors=descriptors,
                        n_sequences=n_seqs,
                        max_len=250,
                        return_smiles=True,
                        return_losses=False,
                    )
                    for sample in sampled:
                        try:
                            mol = smiles_to_mol(sample)
                            smiles = mol_to_smiles(mol)
                            # print(smiles)
                            samples.append(smiles)
                        except:
                            continue
        
        total = len(models) * n_seqs
        valid = len(samples)
        print(f"Generated {valid}/{total} valid samples")

        # chlorination_count = sum(1 for s in samples if 'Cl' in s)
        # print(f"Chlorination occurrences: {chlorination_count} out of {valid} valid samples")
        # sulfur_count = sum(1 for s in samples if 'S' in s)
        # print(f"Sulfur occurrences: {sulfur_count} out of {valid} valid samples")

        # Create 2D UMAP of samples
        mols = [smiles_to_mol(s) for s in samples]
        fps = [mol_to_fpr(m, rad=2, nbs=2048) for m in mols]
        fps_array = np.array(fps)
        print(fps_array.shape)
        return fps_array

    smi1 = r"CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O"
    fp_smi1 = mol_to_fpr(smiles_to_mol(smi1), rad=2, nbs=2048)
    fps_sampled1 = get_samples(smi1)

    smi2 = r"CCCCCCCCCC(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@H](CC(=O)N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H]3[C@H](OC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](NC(=O)CNC(=O)[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)CNC3=O)CCCN)CC(=O)O)C)CC(=O)O)CO)[C@H](C)CC(=O)O)CC(=O)C4=CC=CC=C4N)C"
    fp_smi2 = mol_to_fpr(smiles_to_mol(smi2), rad=2, nbs=2048)
    fps_sampled2 = get_samples(smi2)

    # add 1 dim to single fps
    fp_smi1 = fp_smi1.reshape(1, -1)
    fp_smi2 = fp_smi2.reshape(1, -1)

    print(fp_smi1.shape)
    print(fps_sampled1.shape)
    print(fp_smi2.shape)
    print(fps_sampled2.shape)

    # concat all fps
    fps_all = np.concatenate([fps_sampled1, fps_sampled2, fp_smi1, fp_smi2], axis=0)
    lbs_all = np.array([0]*fps_sampled1.shape[0] + [2]*fps_sampled2.shape[0] + [1]*fp_smi1.shape[0] + [3]*fp_smi2.shape[0])

    model = PCA(n_components=2)
    embedding = model.fit_transform(fps_all)
    ev = model.explained_variance_ratio_

    lbl_to_name = {0: "Samples A", 1: "Original A", 2: "Samples B", 3: "Original B"}
    plt.figure(figsize=(8, 6))

    for lbl, color in zip([0, 2], ["#1f77b4", "#2ca02c"]):  # blue, green
        idxs = np.where(lbs_all == lbl)
        x, y = embedding[idxs, 0], embedding[idxs, 1]
        sns.kdeplot(
            x=x.flatten(),
            y=y.flatten(),
            fill=True,
            alpha=0.5,
            levels=40,
            color=color,
            label=lbl_to_name[lbl],
        )

    for lbl, color, marker in zip([1, 3], ["#1f77b4", "#2ca02c"], ["X", "D"]):
        idxs = np.where(lbs_all == lbl)
        x, y = embedding[idxs, 0], embedding[idxs, 1]
        plt.scatter(
            x, y,
            color=color,
            s=150,
            edgecolor="black",
            marker=marker,
            label=lbl_to_name[lbl],
            zorder=5,
        )
    
    def _update_legend_marker_size(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([40])  # smaller size in legend

    plt.legend(
        handler_map={plt.Line2D: HandlerPathCollection(update_func=_update_legend_marker_size)},
        scatterpoints=1,
        loc="best"
    )

    plt.xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "conditional_model_sampling_pca.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Generate compounds with a trained CLM model."""

import argparse
import os
import re
from typing import List

import torch
from tqdm import tqdm

from clm.datasets import Vocabulary
from clm.models import RNN


def cli() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--work-dir", type=str, required=True, help="Working directory for output files")
    
    parser.add_argument("--sample-size", type=int, default=1_000_000, help="Number of compounds to sample per model")

    parser.add_argument("--train-dataset-name", type=str, default="coconut_smiles", help="Name of the training dataset")
    parser.add_argument("--enum-factor", type=int, default=100, help="Enumeration factor for model selection")
    parser.add_argument("--rnn-type", type=str, default="LSTM", choices=["GRU", "LSTM"], help="Type of RNN to use")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of RNN layers")
    parser.add_argument("--embedding-size", type=int, default=128, help="Size of the embedding layer")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Size of the hidden layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
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
) -> List[RNN]:
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

    models: List[RNN] = []

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
        model = RNN(
            vocabulary=vocab,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()

        models.append(model)
    
    return models


def sample_model(model: RNN, n_samples: int) -> List[str]:
    """
    Sample SMILES strings from the given model.

    :param model: RNN model to sample from
    :param n_samples: Number of samples to generate
    :return: List of sampled SMILES strings
    """
    model.eval()

    with torch.no_grad():
        samples = model.sample(
            n_sequences=n_samples,
            max_len=250,
            return_smiles=True,
            return_losses=False,
            descriptors=None,
        )

    return samples


def main() -> None:
    """
    Main function to generate compounds.
    """
    args = cli()
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    models = load_models(
        args.model_dir,
        train_dataset_name=args.train_dataset_name,
        enum_factor=args.enum_factor,
        rnn_type=args.rnn_type,
        n_layers=args.n_layers,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        device=device,
    )
    print(f"Gathered {len(models)} models for sampling")

    # Define output path for dereplicated SMILES
    output_smiles_path = os.path.join(args.work_dir, "sampled_unique_compounds.smi")

    batch_size = 10_000
    flush_every = 10

    # Sample SMILES; make sure to dereplicate with InChiKey
    with open(output_smiles_path, "a", buffering=1024 * 1024) as f:

        for model in tqdm(models, desc="sampling models", leave=False):
            
            remaining = args.sample_size
            batch_idx = 0
            
            pbar = tqdm(total=args.sample_size, desc="sampling compounds", leave=False)
            while remaining > 0:
                n = batch_size if remaining >= batch_size else remaining
                batch_idx += 1

                with torch.inference_mode():
                    samples: List[str] = sample_model(model, n_samples=n)

                if samples:
                    f.write("\n".join(samples))
                    f.write("\n")
                    wrote = len(samples)
                else:
                    wrote = 0

                pbar.update(wrote)
                remaining -= wrote

                if (batch_idx % flush_every) == 0:
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass  # Not all file systems support fsync

            pbar.close()

            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # Not all file systems support fsync


if __name__ == "__main__":
    main()

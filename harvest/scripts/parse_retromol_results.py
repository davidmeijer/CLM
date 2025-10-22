#!/usr/bin/env python3

"""Parse Retromol results and extract fingerprints for CLM training."""

import argparse
import os

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from retromol.fingerprint import (
    FingerprintGenerator,
    NameSimilarityConfig,
    polyketide_family_of
)
from retromol.helpers import iter_json
from retromol.io import Result
from retromol.rules import get_path_default_matching_rules


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: parsed arguments namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to RetroMol Results JSONL file")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")

    # Fingerprinting settings
    parser.add_argument("--cov-thresh", type=float, default=1.0, help="Coverage threshold for fingerprints")
    parser.add_argument("--num-bits", type=int, default=512, help="Number of bits in the fingerprint")
    parser.add_argument("--counted", action="store_true", help="Use counted fingerprints instead of binary")

    return parser.parse_args()


def get_fingerprint_generator() -> FingerprintGenerator:
    """
    Create and configure a FingerprintGenerator for polyketides.

    :return: configured FingerprintGenerator instance
    """
    path_default_matching_rules = get_path_default_matching_rules()
    collapse_by_name = ["glycosylation", "methylation"]
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
    Main function to parse Retromol results and extract fingerprints.
    """
    args = cli()
    os.makedirs(args.outdir, exist_ok=True)

    # Define paths
    outfile = os.path.join(args.outdir, "retromol_fingerprints.txt")
    logfile = os.path.join(args.outdir, "retromol_fingerprints.log")

    # Initialize fingerprint generator
    fingerprint_generator = get_fingerprint_generator()

    eligible_results = total_fingerprints = 0
    with (open(outfile, "w") as outfile, open(logfile, "w") as logfile):

        # Write out command line arguments
        logfile.write(f"Command line arguments: {args}\n")

        # Write header
        header = ",".join([f"bit_{i}" for i in range(args.num_bits)]) + "\n"
        outfile.write(f"smiles,{header}")

        for json_data in tqdm(iter_json(args.results, jsonl=True), desc="Fingerprinting results"):
            result = Result.from_serialized(json_data["result"])
            cov = result.best_total_coverage()

            # Calculate fingerprint if coverage meets threshold
            if cov >= args.cov_thresh:
                eligible_results += 1
            else:
                continue

            smiles = result.get_input_smiles(remove_tags=True)
            fingerprint = calc_fingerprint(fingerprint_generator, result, args.num_bits, args.counted)
            
            for fp_vector in fingerprint:
                total_fingerprints += 1
                fp_str = ",".join(map(str, fp_vector.tolist()))
                outfile.write(f"{smiles},{fp_str}\n")
        
        msg = f"Found {eligible_results} eligible results with coverage >= {args.cov_thresh}"
        logfile.write(msg + "\n")
        msg = f"Total fingerprints generated: {total_fingerprints}"
        logfile.write(msg + "\n")

        # Flush every 1000 fingerprints
        if total_fingerprints % 1000 == 0:
            outfile.flush()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Dereplicate SMILES strings from an input file."""

import argparse, os, sys
from typing import Tuple

from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .smi file (no header)")
    parser.add_argument("--work-dir", type=str, required=True, help="Path to working directory")
    return parser.parse_args()


def process_compounds(input_path: str, output_path: str) -> Tuple[int, int, int, int, int]:
    """
    Process and dereplicate SMILES strings from the input file.

    :param input_path: Path to the input .smi file
    :param output_path: Path to the output .smi file
    :return: Tuple containing counts of total lines, blank lines, valid SMILES, invalid SMILES, and duplicate valid SMILES
    """
    total_lines = blank_lines = valid_smiles = unique_smiles = duplicate_valid = 0
    seen = set()

    with (
        open(input_path, "r", encoding="utf-8", errors="replace") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        # Write header
        fout.write("identifier,inchikey,canonical_smiles\n")

        for line_idx, line in tqdm(enumerate(fin), desc="Processing SMILES", unit="lines"):
            total_lines += 1
            s = line.strip()
            if not s:
                blank_lines += 1
                continue
            
            mol = Chem.MolFromSmiles(s, sanitize=True)
            if mol is None:
                continue  # invalid SMILES

            valid_smiles += 1
            can = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            mol_no_stereo = Chem.MolFromSmiles(can, sanitize=True)
            if mol_no_stereo is None:
                continue  # happens rarely, but just in case
            inchikey_no_stereo = Chem.MolToInchiKey(mol_no_stereo)

            if inchikey_no_stereo in seen:
                duplicate_valid += 1
                continue
            seen.add(inchikey_no_stereo)
            unique_smiles += 1
            fout.write(f"{unique_smiles},{inchikey_no_stereo},{can}\n")

            # Flush output every 1000 lines
            if line_idx % 1000 == 0:
                fout.flush()

    return total_lines, blank_lines, valid_smiles, unique_smiles, duplicate_valid


def main() -> None:
    """
    Main function to dereplicate SMILES strings.
    """
    args = cli()
    os.makedirs(args.work_dir, exist_ok=True)

    # Force all outputs into work-dir
    output_path = os.path.join(args.work_dir, "valid_dereplicated.smi")
    log_path = os.path.join(args.work_dir, "dereplicate_smiles.log")

    try:
        total, blanks, valid, unique, dups = process_compounds(args.input, output_path)
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"ERROR: {str(e)}\n")
        sys.exit(1)

    invalid = total - blanks - valid

    with open(log_path, "a", encoding="utf-8") as log:
        log.write("Dereplication summary\n")
        log.write(f"Input file           : {os.path.abspath(args.input)}\n")
        log.write(f"Output file          : {os.path.abspath(output_path)}\n")
        log.write(f"Total lines          : {total}\n")
        log.write(f"Blank lines          : {blanks}\n")
        log.write(f"Valid SMILES         : {valid}\n")
        log.write(f"Invalid SMILES       : {invalid}\n")
        log.write(f"Unique valid SMILES  : {unique}\n")
        log.write(f"Duplicate valid      : {dups}\n")

    print(f"[DONE] valid={valid}, unique={unique}, invalid={invalid}")
    print(f"Outputs written to: {os.path.abspath(args.work_dir)}")


if __name__ == "__main__":
    main()

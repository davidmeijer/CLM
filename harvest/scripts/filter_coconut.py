#!/usr/bin/env python3

"""Retrieve microbial and fungal SMILES from the COCONUT database, using a local NCBI taxonomy database stored in --work-dir."""

import argparse
import os, sys, zipfile
from collections import Counter
from urllib.request import urlretrieve
from urllib.parse import urlparse, unquote
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# --- Local taxonomy import ---
try:
    from ete3 import NCBITaxa
except ImportError as e:
    raise SystemExit("This script now uses ete3. Please install it first: pip install ete3") from e

COCONUT_URL = r"https://coconut.s3.uni-jena.de/prod/downloads/2025-10/coconut_csv-10-2025.zip"

# Global handle to avoid re-init
_NCBI: Optional[NCBITaxa] = None


def cli() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True, help="Working directory for output files and taxonomy cache")
    parser.add_argument("--update-taxdump", action="store_true", help="Update the local NCBI taxonomy database (slow)")
    return parser.parse_args()


def _get_ncbi(work_dir: str, update: bool = False) -> NCBITaxa:
    """
    Initialize a local NCBITaxa instance that keeps its SQLite DB inside work_dir.
    """
    global _NCBI
    if _NCBI is not None:
        return _NCBI

    # Place taxonomy database inside the given work_dir
    taxdb_dir = os.path.join(work_dir, "ncbi_taxonomy")
    os.makedirs(taxdb_dir, exist_ok=True)
    taxdb_path = os.path.join(taxdb_dir, "taxdump.sqlite")

    # Override ete3's default DB location
    _NCBI = NCBITaxa(dbfile=taxdb_path)

    if update or not os.path.exists(taxdb_path):
        print(f"Building or updating local NCBI taxonomy database in {taxdb_dir} (this may take several minutes)...", flush=True)
        _NCBI.update_taxonomy_database()
    else:
        print(f"Using local taxonomy DB: {taxdb_path}")

    return _NCBI


def fetch(url: str, work_dir: str = ".") -> str:
    """
    Download `url` into `work_dir` with progress; unzip if needed.
    """
    os.makedirs(work_dir, exist_ok=True)
    name = unquote(os.path.basename(urlparse(url).path)) or "downloaded.file"
    fpath = os.path.join(work_dir, name)
    if not os.path.exists(fpath):
        def _hook(blocks, block_size, total_size):
            if total_size <= 0:
                sys.stdout.write(f"\rDownloading {name}: {blocks*block_size/1_000_000:.1f} MB"); sys.stdout.flush(); return
            done = min(blocks * block_size, total_size)
            pct = done * 100 // total_size
            bar = "█" * int(30 * done / total_size) + "-" * (30 - int(30 * done / total_size))
            sys.stdout.write(f"\rDownloading {name}: |{bar}| {pct}% ({done/1_000_000:.1f}/{total_size/1_000_000:.1f} MB)")
            sys.stdout.flush()
        urlretrieve(url, fpath, _hook)
        sys.stdout.write("\n")
    else:
        print(f"Already downloaded: {fpath}")

    if zipfile.is_zipfile(fpath):
        extract_dir = os.path.join(work_dir, os.path.splitext(name)[0])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(fpath) as zf:
            zf.extractall(extract_dir)
        print(f"Unpacked into: {os.path.abspath(extract_dir)}")
        return extract_dir

    return fpath


def get_lineage_text(genus: str, ncbi: NCBITaxa) -> List[str]:
    """
    Retrieve the taxonomic lineage (names) for a given genus using local NCBI taxonomy DB.
    Returns [] if not found.
    """
    try:
        mapping = ncbi.get_name_translator([genus])
        if not mapping or genus not in mapping or not mapping[genus]:
            return []
        taxid = mapping[genus][0]

        ranks = ncbi.get_rank([taxid])
        rank = ranks.get(taxid)
        if rank != "genus":
            lineage_taxids = ncbi.get_lineage(taxid)
            lineage_ranks = ncbi.get_rank(lineage_taxids)
            genus_taxids = [tid for tid in lineage_taxids if lineage_ranks.get(tid) == "genus"]
            if genus_taxids:
                taxid = genus_taxids[-1]

        lineage_taxids = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage_taxids)
        lineage_names = [names[tid] for tid in lineage_taxids if tid in names]
        return lineage_names
    except Exception:
        return []


def determine_organism_type(genus: str, ncbi: NCBITaxa) -> str:
    """
    Determine if the given genus corresponds to a bacterium or fungus using local taxonomy.
    """
    lineage = get_lineage_text(genus, ncbi)
    s = set(lineage)
    if "Bacteria" in s:
        return "bacterium"
    elif "Fungi" in s:
        return "fungal"
    else:
        return "other"


def main() -> None:
    args = cli()
    os.makedirs(args.work_dir, exist_ok=True)

    # Initialize local taxonomy DB in work_dir
    ncbi = _get_ncbi(args.work_dir, update=args.update_taxdump)

    # Fetch and unzip COCONUT dataset
    dirpath = fetch(COCONUT_URL, args.work_dir)
    fpath = os.path.join(dirpath, "coconut_csv-10-2025.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Expected file not found: {fpath}")

    genus_to_type_cache: Dict[str, str] = {}
    processed_rows = 0
    chunk_size = 1000
    genus_counter = Counter()

    outpath = os.path.join(args.work_dir, "coconut_microbial_fungal_smiles.csv")
    with open(outpath, "w") as out_f:
        out_f.write("identifier,canonical_smiles\n")

        for chunk in tqdm(pd.read_csv(fpath, chunksize=chunk_size), desc="Processing COCONUT dataset"):
            chunk = chunk[["identifier", "canonical_smiles", "organisms"]]
            chunk = chunk.dropna(subset=["canonical_smiles", "organisms"])
            if chunk.empty:
                continue

            for _, row in chunk.iterrows():
                identifier = row["identifier"]
                smiles = row["canonical_smiles"]
                orgs = str(row["organisms"]).split("|")
                orgs = [o.split(" ")[0] for o in orgs if o.strip()]
                orgs = list(set(orgs))
                if not orgs:
                    continue

                bacterial_or_fungal_found = False
                for genus in orgs:
                    genus_counter[genus] += 1
                    organism_type = genus_to_type_cache.get(genus)
                    if not organism_type:
                        try:
                            organism_type = determine_organism_type(genus, ncbi)
                            genus_to_type_cache[genus] = organism_type
                        except Exception:
                            continue
                    if organism_type in ("bacterium", "fungal"):
                        bacterial_or_fungal_found = True

                if bacterial_or_fungal_found:
                    out_f.write(f"{identifier},{smiles}\n")

            processed_rows += len(chunk)
            out_f.flush()

    most_common_genera = genus_counter.most_common(10)
    print("Most common genera in COCONUT dataset:")
    for genus, count in most_common_genera:
        organism_type = genus_to_type_cache.get(genus, "unknown")
        print(f"{genus}: {count} occurrences ({organism_type})")

    print(f"Total unique genera processed: {len(genus_counter)}")
    print(f"Processed {processed_rows} rows from COCONUT dataset that had both 'canonical_smiles' and 'organisms'")
    print(f"Output written to: {outpath}")


if __name__ == "__main__":
    main()

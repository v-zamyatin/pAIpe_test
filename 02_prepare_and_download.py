#!/usr/bin/env python3

import os
import time
import h5py
import pandas as pd
import subprocess
from collections import defaultdict


#############################################
### Variables
#############################################

INPUT_FILE = "/workspace/input_variants.txt" 
DATA_DIR = "/workspace/data"
PREPROC_DATA_TRAIN = "/workspace/preprocessed_data/train"
PREPROC_DATA_TEST = "/workspace/preprocessed_data/test"


TG_DIR = f"{DATA_DIR}/TraitGym"
TG_DATA = f"{DATA_DIR}/TraitGym/TG.txt"
CADD_DATA = f"{DATA_DIR}/CADD/whole_genome_SNVs.tsv.gz"
CADD_INDEX_DATA = f"{DATA_DIR}/CADD/whole_genome_SNVs.tsv.gz.tbi"
GPN_MSA_DATA = f"{DATA_DIR}/GPN-MSA/scores.tsv.bgz"
GPN_MSA_INDEX_DATA = f"{DATA_DIR}/GPN-MSA/scores.tsv.bgz.tbi"


CADD_URL = "https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz"
CADD_INDEX_URL = "https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz.tbi"
GPN_MSA_URL = "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz"
GPN_MSA_INDEX_URL = "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz.tbi"
TG_URL = "https://huggingface.co/datasets/songlab/TraitGym/resolve/main/complex_traits_matched_9/test.parquet"

#############################################
#### CHECK INPUT FILE #######################
#############################################
import os
import sys
import pandas as pd

INPUT_FILE = "/workspace/input_variants.txt"

# Длины хромосом GRCh38/hg38 (primary assembly, NCBI/Ensembl) :contentReference[oaicite:2]{index=2}
CHROM_SIZES = {
    "chr1": 248_956_422,
    "chr2": 242_193_529,
    "chr3": 198_295_559,
    "chr4": 190_214_555,
    "chr5": 181_538_259,
    "chr6": 170_805_979,
    "chr7": 159_345_973,
    "chr8": 145_138_636,
    "chr9": 138_394_717,
    "chr10": 133_797_422,
    "chr11": 135_086_622,
    "chr12": 133_275_309,
    "chr13": 114_364_328,
    "chr14": 107_043_718,
    "chr15": 101_991_189,
    "chr16": 90_338_345,
    "chr17": 83_257_441,
    "chr18": 80_373_285,
    "chr19": 58_617_616,
    "chr20": 64_444_167,
    "chr21": 46_709_983,
    "chr22": 50_818_468,
    "chrX": 156_040_895,
    "chrY": 57_227_415,
    # митохондрия hg38, chrM (MT) ≈ 16 569 bp :contentReference[oaicite:3]{index=3}
    "chrM": 16_569,
}

EXPECTED_HEADER = ["CHROM", "POS", "REF", "ALT"]
VALID_BASES = {"A", "C", "G", "T"}


def validate_input_variants(path: str) -> pd.DataFrame:
    errors: list[str] = []

    # --- 0. Базовые проверки файла ---
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Input file not found: {path}")

    if os.path.getsize(path) == 0:
        sys.exit(f"[ERROR] Input file is empty: {path}")

    # --- 1. Проверка хедера и разделителя ---
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n\r")

    header_cols_tab = header_line.split("\t")

    if header_cols_tab != EXPECTED_HEADER:
        errors.append(
            "Header must be exactly (tab-separated): "
            f"'CHROM\\tPOS\\tREF\\tALT', got: {header_line!r}"
        )

    # --- 2. Чтение таблицы как TSV (строки как строки) ---
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            comment="#",
        )
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read TSV file '{path}': {e}")

    # --- 3. Проверка количества и названий колонок ---
    if df.shape[1] != 4:
        errors.append(
            f"File must contain exactly 4 columns, got {df.shape[1]}: {list(df.columns)!r}"
        )

    missing_cols = [c for c in EXPECTED_HEADER if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in EXPECTED_HEADER]
    if extra_cols:
        errors.append(f"Unexpected extra columns present: {extra_cols}")

    # Приводим к правильному порядку, если все колонки есть
    if not missing_cols:
        df = df[EXPECTED_HEADER]

    # --- 4. Проверка пропусков ---
    if df.isnull().any().any():
        n_rows_with_na = df.isnull().any(axis=1).sum()
        errors.append(f"Found {n_rows_with_na} rows with missing values (NaN / empty).")

    # --- 5. Проверка названий хромосом ---
    allowed_chroms = set(CHROM_SIZES.keys())
    invalid_chr_mask = ~df["CHROM"].isin(allowed_chroms)
    if invalid_chr_mask.any():
        bad_chroms = sorted(df.loc[invalid_chr_mask, "CHROM"].dropna().unique())
        errors.append(
            "Invalid chromosome names detected: "
            f"{', '.join(bad_chroms)}. "
            f"Allowed: {', '.join(sorted(allowed_chroms))}."
        )

    # --- 6. Проверка POS: целые, >=1 и <= длины хромосомы ---
    # сначала: все ли это целые числа
    pos_str = df["POS"].astype(str)
    non_int_mask = ~pos_str.str.fullmatch(r"[0-9]+")
    if non_int_mask.any():
        bad_examples = df.loc[non_int_mask, ["CHROM", "POS"]].head(5).to_dict(
            orient="records"
        )
        errors.append(
            f"POS column must contain positive integers only. "
            f"Invalid POS in {non_int_mask.sum()} rows, examples: {bad_examples}"
        )
    else:
        # конвертируем в int
        df["POS"] = pos_str.astype(int)

        # POS >= 1
        below_one = df["POS"] < 1
        if below_one.any():
            errors.append(
                f"{below_one.sum()} rows have POS < 1. "
                "Positions are 1-based coordinates."
            )

        # POS <= длина хромосомы
        for chrom, max_pos in CHROM_SIZES.items():
            mask_chr = df["CHROM"] == chrom
            if not mask_chr.any():
                continue
            over_mask = mask_chr & (df["POS"] > max_pos)
            if over_mask.any():
                max_seen = int(df.loc[over_mask, "POS"].max())
                n_over = int(over_mask.sum())
                examples = df.loc[over_mask, ["CHROM", "POS"]].head(5).to_dict(
                    orient="records"
                )
                errors.append(
                    f"{n_over} rows have POS > chromosome length for {chrom} "
                    f"(max allowed {max_pos}, max seen {max_seen}). Examples: {examples}"
                )

    # --- 7. Проверка REF / ALT ---
    for col in ["REF", "ALT"]:
        # длина 1 символ
        lengths = df[col].astype(str).str.len()
        wrong_len_mask = lengths != 1
        if wrong_len_mask.any():
            examples = df.loc[wrong_len_mask, ["CHROM", "POS", col]].head(
                5
            ).to_dict(orient="records")
            errors.append(
                f"{wrong_len_mask.sum()} rows have {col} length != 1. "
                f"Expected single nucleotide. Examples: {examples}"
            )

        # только A/C/G/T
        upper_vals = df[col].astype(str).str.upper()
        invalid_nt_mask = ~upper_vals.isin(VALID_BASES)
        if invalid_nt_mask.any():
            bad_vals = sorted(upper_vals[invalid_nt_mask].dropna().unique())
            errors.append(
                f"Invalid bases in column {col}: {', '.join(bad_vals)}. "
                f"Allowed bases: {', '.join(sorted(VALID_BASES))}."
            )

    # REF != ALT
    same_ra_mask = (
        df["REF"].astype(str).str.upper() == df["ALT"].astype(str).str.upper()
    )
    if same_ra_mask.any():
        examples = df.loc[same_ra_mask, ["CHROM", "POS", "REF", "ALT"]].head(
            5
        ).to_dict(orient="records")
        errors.append(
            f"{same_ra_mask.sum()} rows have REF == ALT (no variation). "
            f"Examples: {examples}"
        )

    # --- 8. Дубликаты ---
    dup_mask = df.duplicated(subset=["CHROM", "POS", "REF", "ALT"], keep=False)
    if dup_mask.any():
        examples = (
            df.loc[dup_mask, ["CHROM", "POS", "REF", "ALT"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        errors.append(
            f"Found {dup_mask.sum()} duplicate rows with identical (CHROM, POS, REF, ALT). "
            f"Examples: {examples}"
        )

    # --- 9. Итог ---
    if errors:
        print("[ERROR] Input variants file failed validation:\n", file=sys.stderr)
        for i, msg in enumerate(errors, 1):
            print(f"  {i}. {msg}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[OK] Input variants file '{path}' passed validation: "
        f"{len(df)} variants, {df.shape[1]} columns."
    )
    return df


validate_input_variants(INPUT_FILE)


################################################################################
### 1. Download data
################################################################################

# print("=== Step 1: Download data ===")

# # CADD
print(f"Downloading CADD data...")
subprocess.run(["wget","-O", CADD_DATA, CADD_URL])
subprocess.run(["wget", "-O", CADD_INDEX_DATA, CADD_INDEX_URL])
print("✅ All CADD data downloaded successfully.\n")

# GPN-MSA
print(f"Downloading GPN-MSA data...")
subprocess.run(["wget", "-O", GPN_MSA_DATA, GPN_MSA_URL])
subprocess.run(["wget", "-O", GPN_MSA_INDEX_DATA, GPN_MSA_INDEX_URL])
print("✅ All GPN-MSA data downloaded successfully.\n")

# Trait-Gym
print("Downloading TraitGym data...")
subprocess.run(["wget", "-O", f"{TG_DIR}/test.parquet", TG_URL])
TG = pd.read_parquet(f"{TG_DIR}/test.parquet")
TG.iloc[:, 0] = "chr" + TG.iloc[:, 0].astype(str)
TG.to_csv(TG_DATA, sep='\t', index=False)
os.remove(f"{TG_DIR}/test.parquet")
print("✅ TraitGym data downloaded successfully.\n")

print("=== All datasets are ready! ===")


################################################################################
### Function to prepare data for classification
################################################################################

def prepare_for_classification(INPUT_FILE, PREPROC_DATA):
    print(f"=== Processing {INPUT_FILE} → {PREPROC_DATA} ===")

   
    BORZOI_VCF = f"{PREPROC_DATA}/borzoi_input.vcf"
    BORZOI_TRACK_H5 = f"{PREPROC_DATA}/sad.h5"
    BORZOI_TRACK_TXT = f"{PREPROC_DATA}/borzoi_tracks.txt"
    BORZOI_CADD_GPN_MSA = f"{PREPROC_DATA}/borzoi_with_cadd_gpn_msa.txt"
    GPN_MSA_SUBSET = f"{PREPROC_DATA}/gpn_msa_subset.txt"
    CADD_SUBSET = f"{PREPROC_DATA}/cadd_subset.txt"


    ################################################################################
    ### 2. Prepare VCF file for Borzoi run
    ################################################################################

    print("=== Step 2: Preparing VCF file for Borzoi ===")

    df = pd.read_csv(INPUT_FILE, sep="\t")
    df.columns = df.columns.str.upper()
    
    vcf_to_borzoi = pd.DataFrame()
    vcf_to_borzoi['chrom'] =  df['CHROM'].astype(str)
    vcf_to_borzoi['pos'] = df['POS']
    vcf_to_borzoi['hash'] = df.iloc[:, :4].astype(str).agg("_".join, axis=1)
    vcf_to_borzoi['ref'] = df['REF']
    vcf_to_borzoi['alt'] = df['ALT']
    vcf_to_borzoi['some1'] = '.'
    vcf_to_borzoi['some2'] = '.'


    with open(BORZOI_VCF, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        vcf_to_borzoi.to_csv(f, index=False, sep="\t", header=False)

    ###############################################################################
    ## 3. Borzoi tracks
    ###############################################################################

    print("=== Step 3: Borzoi tracks ===")

    os.system(
        f"borzoi_sad.py -o {PREPROC_DATA} --rc --stats logD2 -u "
        f"-t /opt/borzoi/examples/targets_human.txt "
        f"/opt/borzoi/examples/params_pred.json "
        f"/opt/borzoi/examples/saved_models/f3c0/train/model0_best.h5 "
        f"{BORZOI_VCF}"
    )

    ################################################################################
    ### 4. Convert Borzoi tracks to txt
    ################################################################################

    print("=== Step 4: Convert Borzoi tracks to txt ===")

    sad_h5 = h5py.File(BORZOI_TRACK_H5, 'r')
    sad_df = pd.DataFrame(sad_h5['logD2'])
    sad_df.index = list(map(lambda x: str(x).strip("'b"), sad_h5['snp']))
    sad_df.columns = list(map(lambda x: str(x).strip("'b"), sad_h5['target_labels']))
    sad_df['chrom'] = sad_df.index.str.split('_').str[0]
    sad_df['hash'] = sad_df.index

    seen = defaultdict(int)
    new_cols = []
    for col in sad_df.columns:
        if seen[col] == 0:
            new_cols.append(col)
        else:
            new_cols.append(f"{col}.{seen[col]}")
        seen[col] += 1
    sad_df.columns = new_cols

    sad_df.to_csv(BORZOI_TRACK_TXT, sep='\t', index=False)

    ###############################################################################
    ## 5. Create CADD and GPN-MSA subsets
    ###############################################################################

    print("=== Step 5: Create CADD and GPN-MSA subsets ===")

    cmd = (
        f"set -euo pipefail; "
        f"awk 'BEGIN{{OFS=\"\\t\"}} NR>1 {{gsub(/^chr/,\"\",$1); print $1, $2-1, $2}}' '{INPUT_FILE}' "
        f"| sort -k1,1V -k2,2n "
        f"| tee >(tabix '{GPN_MSA_DATA}' -R - > '{GPN_MSA_SUBSET}') "
        f"      >(tabix '{CADD_DATA}' -R - > '{CADD_SUBSET}') "
        f"> /dev/null"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

    for path in [GPN_MSA_SUBSET, CADD_SUBSET]:
        while not os.path.exists(path) or os.path.getsize(path) == 0:
            time.sleep(0.5)
    time.sleep(60)
    print("=== Step 5 done ===")

    ################################################################################
    ### 6. Add CADD and GPN-MSA scores
    ################################################################################

    print("=== Step 6: Add CADD and GPN-MSA scores ===")

    gpn_dict = {
        f"chr{c}_{p}_{r}_{a}": s
        for c, p, r, a, s, *_ in (
            line.rstrip().split("\t") for line in open(GPN_MSA_SUBSET)
        )
    }

    cadd_dict = {
        f"chr{c}_{p}_{r}_{a}": s
        for c, p, r, a, s, *_ in (
            line.rstrip().split("\t") for line in open(CADD_SUBSET)
        )
    }

    with open(BORZOI_TRACK_TXT) as fin, open(BORZOI_CADD_GPN_MSA, "w") as fout:
        header = fin.readline().strip().split("\t")
        h_idx = header.index("hash")

        insert_index = len(header) - 2
        header[insert_index:insert_index] = ["gpn-msa_score", "RawScore"]
        fout.write("\t".join(header) + "\n")

        for line in fin:
            row = line.strip().split("\t")
            h = row[h_idx]
            row[insert_index:insert_index] = [gpn_dict.get(h, ""), cadd_dict.get(h, "")]
            fout.write("\t".join(row) + "\n")

    print(f"✅ Finished processing {INPUT_FILE}\n")


################################################################################
### Run for both input files
################################################################################


prepare_for_classification(INPUT_FILE, PREPROC_DATA_TEST)
prepare_for_classification(TG_DATA, PREPROC_DATA_TRAIN) 

print("🎯 All datasets processed successfully.")



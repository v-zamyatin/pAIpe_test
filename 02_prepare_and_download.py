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



################################################################################
### 1. Download data
################################################################################

# print("=== Step 1: Download data ===")

# # CADD
# print(f"Downloading CADD data...")
# subprocess.run(["wget","-O", CADD_DATA, CADD_URL])
# subprocess.run(["wget", "-O", CADD_INDEX_DATA, CADD_INDEX_URL])
# print("✅ All CADD data downloaded successfully.\n")

# # GPN-MSA
# print(f"Downloading GPN-MSA data...")
# subprocess.run(["wget", "-O", GPN_MSA_DATA, GPN_MSA_URL])
# subprocess.run(["wget", "-O", GPN_MSA_INDEX_DATA, GPN_MSA_INDEX_URL])
# print("✅ All GPN-MSA data downloaded successfully.\n")

# # Trait-Gym
# print("Downloading TraitGym data...")
# subprocess.run(["wget", "-O", f"{TG_DIR}/test.parquet", TG_URL])
# TG = pd.read_parquet(f"{TG_DIR}/test.parquet")
# TG.iloc[:, 0] = "chr" + TG.iloc[:, 0].astype(str)
# TG.to_csv(TG_DATA, sep='\t', index=False)
# os.remove(f"{TG_DIR}/test.parquet")
# print("✅ TraitGym data downloaded successfully.\n")

# print("=== All datasets are ready! ===")


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

    # print("=== Step 2: Preparing VCF file for Borzoi ===")

    # df = pd.read_csv(INPUT_FILE, sep="\t")
    # df.columns = df.columns.str.upper()
    
    # vcf_to_borzoi = pd.DataFrame()
    # vcf_to_borzoi['chrom'] =  df['CHROM'].astype(str)
    # vcf_to_borzoi['pos'] = df['POS']
    # vcf_to_borzoi['hash'] = df.iloc[:, :4].astype(str).agg("_".join, axis=1)
    # vcf_to_borzoi['ref'] = df['REF']
    # vcf_to_borzoi['alt'] = df['ALT']
    # vcf_to_borzoi['some1'] = '.'
    # vcf_to_borzoi['some2'] = '.'


    # with open(BORZOI_VCF, "w") as f:
    #     f.write("##fileformat=VCFv4.2\n")
    #     vcf_to_borzoi.to_csv(f, index=False, sep="\t", header=False)

    # ###############################################################################
    # ## 3. Borzoi tracks
    # ###############################################################################

    # print("=== Step 3: Borzoi tracks ===")

    # os.system(
    #     f"borzoi_sad.py -o {PREPROC_DATA} --rc --stats logD2 -u "
    #     f"-t /opt/borzoi/examples/targets_human.txt "
    #     f"/opt/borzoi/examples/params_pred.json "
    #     f"/opt/borzoi/examples/saved_models/f3c0/train/model0_best.h5 "
    #     f"{BORZOI_VCF}"
    # )

    ################################################################################
    ### 4. Convert Borzoi tracks to txt
    ################################################################################

    print("=== Step 4: Convert Borzoi tracks to txt ===")

    # sad_h5 = h5py.File(BORZOI_TRACK_H5, 'r')
    # sad_df = pd.DataFrame(sad_h5['logD2'])
    # sad_df.index = list(map(lambda x: str(x).strip("'b"), sad_h5['snp']))
    # sad_df.columns = list(map(lambda x: str(x).strip("'b"), sad_h5['target_labels']))
    # sad_df['chrom'] = sad_df.index.str.split('_').str[0]
    # sad_df['hash'] = sad_df.index

    # seen = defaultdict(int)
    # new_cols = []
    # for col in sad_df.columns:
    #     if seen[col] == 0:
    #         new_cols.append(col)
    #     else:
    #         new_cols.append(f"{col}.{seen[col]}")
    #     seen[col] += 1
    # sad_df.columns = new_cols

    # sad_df.to_csv(BORZOI_TRACK_TXT, sep='\t', index=False)

    ################################################################################
    ### 5. Create CADD and GPN-MSA subsets
    ################################################################################

    print("=== Step 5: Create CADD and GPN-MSA subsets ===")

    # cmd = (
    #     f"set -euo pipefail; "
    #     f"awk 'BEGIN{{OFS=\"\\t\"}} NR>1 {{gsub(/^chr/,\"\",$1); print $1, $2-1, $2}}' '{INPUT_FILE}' "
    #     f"| sort -k1,1V -k2,2n "
    #     f"| tee >(tabix '{GPN_MSA_DATA}' -R - > '{GPN_MSA_SUBSET}') "
    #     f"      >(tabix '{CADD_DATA}' -R - > '{CADD_SUBSET}') "
    #     f"> /dev/null"
    # )
    # subprocess.run(["bash", "-c", cmd], check=True)

    # for path in [GPN_MSA_SUBSET, CADD_SUBSET]:
    #     while not os.path.exists(path) or os.path.getsize(path) == 0:
    #         time.sleep(0.5)
    # time.sleep(120)
    print("=== Step 5 done ===")

    ################################################################################
    ### 6. Add CADD and GPN-MSA scores
    ################################################################################

    print("=== Step 6: Add CADD and GPN-MSA scores ===")

    # gpn_dict = {
    #     f"chr{c}_{p}_{r}_{a}": s
    #     for c, p, r, a, s, *_ in (
    #         line.rstrip().split("\t") for line in open(GPN_MSA_SUBSET)
    #     )
    # }

    # cadd_dict = {
    #     f"chr{c}_{p}_{r}_{a}": s
    #     for c, p, r, a, s, *_ in (
    #         line.rstrip().split("\t") for line in open(CADD_SUBSET)
    #     )
    # }

    # with open(BORZOI_TRACK_TXT) as fin, open(BORZOI_CADD_GPN_MSA, "w") as fout:
    #     header = fin.readline().strip().split("\t")
    #     h_idx = header.index("hash")

    #     insert_index = len(header) - 2
    #     header[insert_index:insert_index] = ["gpn-msa_score", "RawScore"]
    #     fout.write("\t".join(header) + "\n")

    #     for line in fin:
    #         row = line.strip().split("\t")
    #         h = row[h_idx]
    #         row[insert_index:insert_index] = [gpn_dict.get(h, ""), cadd_dict.get(h, "")]
    #         fout.write("\t".join(row) + "\n")

    print(f"✅ Finished processing {INPUT_FILE}\n")


################################################################################
### Run for both input files
################################################################################


# prepare_for_classification(INPUT_FILE, PREPROC_DATA_TEST)
prepare_for_classification(TG_DATA, PREPROC_DATA_TRAIN) 

print("🎯 All datasets processed successfully.")



#!/usr/bin/env python3


import os
import pandas as pd
import h5py
import subprocess
import numpy as np
import joblib
from io import StringIO
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from tqdm.auto import tqdm



#############################################
### Variables
#############################################
RESULTS_DIR = "/workspace/results"
PREPROC_TEST_DATA = "/workspace/preprocessed_data/test"
BORZOI_VCF = f"{PREPROC_TEST_DATA}/borzoi_input.vcf"
BORZOI_GENE_H5 = f"{PREPROC_TEST_DATA}/sed.h5"
BORZOI_GENE_TXT = f"{PREPROC_TEST_DATA}/borzoi_genes.txt"

INPUT_CLF_DATA = f"{PREPROC_TEST_DATA}/borzoi_with_cadd_gpn_msa.txt"
MODEL = "/workspace/results/final_model.joblib"
SCORE = "/workspace/results/predictions_new.txt"

INPUT_FILE = "/workspace/input_variants.txt" 
INPUT_TRACKS = "/workspace/data/tracks_ontology_IDs_RNA-seq_with_groups.xlsx"
BORZOI_SCORE = f"{PREPROC_TEST_DATA}/borzoi_score.txt"
ALPHAGENOME_SCORES = f"{RESULTS_DIR}/alphagenome_score.txt"
ALPHAGENOME_RAW_OUT = f"{RESULTS_DIR}/alphagenome_raw_scores.csv"
BORZOI_SCORE_AG = f"{RESULTS_DIR}/borz_score_ag.txt"

##############################################################################
## 1. Borzoi gene
###############################################################################

print("=== Step 1:  Borzoi gene ===")

os.system(f"borzoi_sed.py -o {PREPROC_TEST_DATA} --rc --stats logSED,logD2 -u -t /opt/borzoi/examples/targets_gtex.txt /opt/borzoi/examples/params_pred.json /opt/borzoi/examples/saved_models/f3c0/train/model0_best.h5 {BORZOI_VCF}")

################################################################################
### 2. Convert Borzoi gene to txt
################################################################################

print("=== Step 2: Convert Borzoi gene to txt ===")

#######  BORZOI TO DF #####
with h5py.File(BORZOI_GENE_H5, "r") as sed_h5:
    df_logSED = pd.DataFrame(sed_h5["logSED"])
    tissue = [t.decode() for t in sed_h5["target_labels"][:]]
    si = sed_h5["si"][:]
    genes = [g.decode() for g in sed_h5["gene"][:]]

    df_logSED.columns = tissue
    snp = sed_h5["snp"][:]
    snp_gene = [f"{snp[si[i]].decode()}-{genes[i]}" for i in range(len(si))]
    df_logSED.index = snp_gene
df_logSED = df_logSED.drop_duplicates()
df_logSED = df_logSED.astype('float64')

######

df_prct= (2 ** df_logSED - 1) * 100
df_prct = df_prct.add_suffix("_prct_change")
df_prct_mean = df_prct.T.groupby(level=0).mean().T
df_prct_mean.columns = df_prct_mean.columns.str.replace("^RNA:", "", regex=True)
df_prct_mean['hash'] = df_prct_mean.index.str.split('-').str[0]
df_prct_mean['gene_id'] = df_prct_mean.index.str.split('-').str[1]
df_prct_mean_cardio = df_prct_mean[[ 'hash', 'gene_id', 'blood_prct_change', 'blood_vessel_prct_change', 'heart_prct_change', 'lung_prct_change']]


prct_cols = [c for c in df_prct_mean_cardio.columns if c.endswith("_prct_change") or c.endswith("_prcnt")]
results = []
for h, g in df_prct_mean_cardio.groupby("hash"):
    row = {"hash": h}
    for col in prct_cols:
        idx = g[col].abs().idxmax()        
        row[col] = g.loc[idx, col]         
        clean_name = col.replace("_prct_change", "").replace("_prcnt", "")
        row[f"{clean_name}_gene"] = g.loc[idx, "gene_id"]
    results.append(row)

result = pd.DataFrame(results)
result.to_csv(BORZOI_GENE_TXT, sep='\t', index=False)


#############################################
###  Step 3: Apply classification model to input dataset 
#############################################

print("=== Step 2: Apply classification model to input dataset  ===")

clf = joblib.load(MODEL)
new_df = pd.read_csv(INPUT_CLF_DATA, sep='\t')
features = [col for col in new_df.columns if col not in ["hash", "chrom"]]
new_df["probability"] = clf.predict_proba(new_df[features])[:, 1]

new_df.to_csv(SCORE, sep='\t', index=False)
print("Предсказания сохранены в predictions_new.txt")


#############################################
###  Step 4: Merge Borzoi and score 
#############################################

print("=== Step 4: Merge Borzoi and score ===")

score = pd.read_csv(SCORE, sep="\t")
score = score.drop_duplicates().reset_index(drop=True)
borz = pd.read_csv(BORZOI_GENE_TXT,  sep="\t")

borz_score = borz.merge(score[['hash', 'probability']], how = 'left', on = 'hash')
borz_score.to_csv(BORZOI_SCORE, sep="\t", index=False)



##############################################################################
## 5. Running AlphaGenome
###############################################################################

print("=== Step 5: Running AlphaGenome ===")

dna_model = dna_client.create('AIzaSyDXa3d7U0zGaqfvH3aWtHd1LhQkq7zITDg')

variants = pd.read_table(INPUT_FILE)
variants['variant_id'] = variants[['CHROM', 'POS', 'REF', 'ALT']].astype(str).agg('_'.join, axis=1)
variants['variant_id'] = variants['variant_id'] + '_b38'

tracks = pd.read_excel(INPUT_TRACKS)
ontology_id_cardio = list(tracks[~(tracks['tissue_group']=='EXCLUDE')].ID)

vcf = variants.copy()
required_columns = ['variant_id', 'CHROM', 'POS', 'REF', 'ALT']
for column in required_columns:
  if column not in vcf.columns:
    raise ValueError(f'VCF file is missing required column: {column}.')

organism = 'human'  # @param ["human", "mouse"] {type:"string"}

# @markdown Specify length of sequence around variants to predict:
sequence_length = '1MB'  # @param ["2KB", "16KB", "100KB", "500KB", "1MB"] { type:"string" }
sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
    f'SEQUENCE_LENGTH_{sequence_length}'
]

# @markdown Specify which scorers to use to score your variants:
score_rna_seq = True  # @param { type: "boolean"}
score_cage = False  # @param { type: "boolean" }
score_procap = False  # @param { type: "boolean" }
score_atac = False  # @param { type: "boolean" }
score_dnase = False  # @param { type: "boolean" }
score_chip_histone = False  # @param { type: "boolean" }
score_chip_tf = False  # @param { type: "boolean" }
score_polyadenylation = True  # @param { type: "boolean" }
score_splice_sites = False  # @param { type: "boolean" }
score_splice_site_usage = False  # @param { type: "boolean" }
score_splice_junctions = False  # @param { type: "boolean" }

# @markdown Other settings:
save_raw = True  # @param { type: "boolean" }

# Parse organism specification.
organism_map = {
    'human': dna_client.Organism.HOMO_SAPIENS,
    'mouse': dna_client.Organism.MUS_MUSCULUS,
}
organism = organism_map[organism]

# Parse scorer specification.
scorer_selections = {
    'rna_seq': score_rna_seq,
    'cage': score_cage,
    'procap': score_procap,
    'atac': score_atac,
    'dnase': score_dnase,
    'chip_histone': score_chip_histone,
    'chip_tf': score_chip_tf,
    'polyadenylation': score_polyadenylation,
    'splice_sites': score_splice_sites,
    'splice_site_usage': score_splice_site_usage,
    'splice_junctions': score_splice_junctions,
}

all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
selected_scorers = [
    all_scorers[key]
    for key in all_scorers
    if scorer_selections.get(key.lower(), False)
]

# Remove any scorers or output types that are not supported for the chosen organism.
unsupported_scorers = [
    scorer
    for scorer in selected_scorers
    if (
        organism.value
        not in variant_scorers.SUPPORTED_ORGANISMS[scorer.base_variant_scorer]
    )
    | (
        (scorer.requested_output == dna_client.OutputType.PROCAP)
        & (organism == dna_client.Organism.MUS_MUSCULUS)
    )
]
if len(unsupported_scorers) > 0:
  print(
      f'Excluding {unsupported_scorers} scorers as they are not supported for'
      f' {organism}.'
  )
  for unsupported_scorer in unsupported_scorers:
    selected_scorers.remove(unsupported_scorer)

# Score variants in the VCF file.
results = []

for i, (_, vcf_row) in enumerate(
    tqdm(vcf.iterrows(),
         total=len(vcf),
         desc='current chunk',     # название внутреннего бара
         leave=False,              # не «оставлять» бар после завершения
         unit='rows',
         position=1),              # строка №1 (под внешним баром)
    1):
  variant = genome.Variant(
      chromosome=str(vcf_row.CHROM),
      position=int(vcf_row.POS),
      reference_bases=vcf_row.REF,
      alternate_bases=vcf_row.ALT,
      name=vcf_row.variant_id,
  )
  interval = variant.reference_interval.resize(sequence_length)

  variant_scores = dna_model.score_variant(
      interval=interval,
      variant=variant,
      variant_scorers=selected_scorers,
      organism=organism,
      
  )
  results.append(variant_scores)

df_scores = variant_scorers.tidy_scores(results)    
if save_raw:
  df_scores.to_csv(ALPHAGENOME_RAW_OUT, index=False)     



# какие ткани оставляем и в каком порядке показываем в итоговой таблице
TG_ORDER = ['blood', 'endothelial', 'blood vessel', 'lung']

# --- 0) join df_all ↔ tracks по ontology_curie → tissue_group ---
#    (left-join: все строки из df_all сохраняются)
df = df_scores.merge(
    tracks[['ID', 'tissue_group']],
    left_on='ontology_curie', right_on='ID',
    how='left'
)
# --- ЖЁСТКАЯ НОРМАЛИЗАЦИЯ ТИПОВ (исправляет "unhashable type: 'Variant'") ---
#   всё, что будет ключом/группировкой — в строки; score — в float
for col in ['variant_id', 'tissue_group', 'gene_name', 'gene_id']:
    if col in df.columns:
        df[col] = df[col].astype('string')  # безопасная строка, NaN → <NA>

# --- 1) фильтрация: убрать EXCLUDE и оставить только нужные группы ---
df = df[df['tissue_group'].isin(TG_ORDER)]

# --- 2) по каждому (variant_id, tissue_group) взять ген с max |raw_score| ---
#     raw_score — это log2FC; берём максимум по модулю.
df_sorted = (
    df.assign(_abs_raw=np.abs(df['raw_score'].astype(float)))
      .sort_values(['variant_id', 'tissue_group', '_abs_raw'],
                   ascending=[True, True, False])
)
best_per_group = (
    df_sorted
      .drop_duplicates(['variant_id', 'tissue_group'], keep='first')
      .drop(columns=['_abs_raw'])
      .copy()
)

# если gene_name пуст, подставим gene_id
best_per_group['gene_display'] = best_per_group['gene_name'].where(
    best_per_group['gene_name'].notna(), best_per_group['gene_id']
)

# --- 3) перевести log2FC → % изменения, сохранив знак ---
# Формула: FC = 2**log2FC; %Δ = (FC - 1) * 100
best_per_group['prct_change'] = (
    (2.0 ** best_per_group['raw_score'].astype(float) - 1.0) * 100.0
)

# --- 4) сформировать широкую таблицу ---
# по каждой ткани: <tg>_gene и <tg>_prct_change
# нормализуем имена колонок: 'blood vessel' → 'blood_vessel'
norm = {tg: tg.replace(' ', '_') for tg in TG_ORDER}

gene_wide = (
    best_per_group
      .pivot(index='variant_id', columns='tissue_group', values='gene_display')
      .rename(columns={tg: f"{norm[tg]}_gene" for tg in TG_ORDER})
)

pct_wide = (
    best_per_group
      .pivot(index='variant_id', columns='tissue_group', values='prct_change')
      .rename(columns={tg: f"{norm[tg]}_prct_change" for tg in TG_ORDER})
)

out = pd.concat([gene_wide, pct_wide], axis=1)

# упорядочим столбцы строго по TG_ORDER
cols = []
for tg in TG_ORDER:
    k = norm[tg]
    cols += [f"{k}_gene", f"{k}_prct_change"]
out = out.reindex(columns=cols)

# вернуть variant_id в таблицу
out = out.reset_index()

# out — это итоговая таблица:
# ['variant_id',
#  'blood_gene','blood_prct_change',
#  'endothelial_gene','endothelial_prct_change',
#  'blood_vessel_gene','blood_vessel_prct_change',
#  'lung_gene','lung_prct_change']
out.variant_id = out.variant_id.str.replace(':','_').str.replace(">", '_')
out.to_csv(ALPHAGENOME_SCORES, index=False, sep='\t')

###############################################################################
### Step 6: Merge Borzoi, score and AlphaGenome
################################################################################

print("=== Step 6: Merge Borzoi, score and AlphaGenome ===")


ag = pd.read_csv(ALPHAGENOME_SCORES, sep="\t")
ag = ag.rename(columns={"variant_id": "hash"})
ag['heart_prct_change'] = np.nan
ag['heart_gene'] = np.nan
ag['source'] = 'alpagenome'
ag = ag[['hash', 'source','blood_prct_change', 'blood_gene', 'blood_vessel_prct_change','blood_vessel_gene', 'lung_prct_change', 'lung_gene', 'endothelial_prct_change', 'endothelial_gene', 'heart_prct_change', 'heart_gene']]

borz_score = pd.read_csv(BORZOI_SCORE, sep="\t")
borz_score['endothelial_prct_change'] = np.nan
borz_score['endothelial_gene'] = np.nan
borz_score['source'] = 'borzoi'
borz_score = borz_score[['hash', 'source', 'blood_prct_change', 'blood_gene', 'blood_vessel_prct_change','blood_vessel_gene', 'lung_prct_change', 'lung_gene', 'endothelial_prct_change', 'endothelial_gene', 'heart_prct_change', 'heart_gene', 'probability']]
ag_new = ag.merge(borz_score[['hash', 'probability']], on ='hash', how='left')

result = pd.concat([ag_new, borz_score], ignore_index=True)
result = result.sort_values(["hash", "source"]).reset_index(drop=True)
cols = [c for c in result.columns if c.endswith("prct_change")]
result[cols] = result[cols].round(2)
result['probability'] = result['probability'].round(2)
result.to_csv(BORZOI_SCORE_AG, sep='\t', index=False)

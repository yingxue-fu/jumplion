# -*- coding: utf-8 -*-

import re 
import logging
import pandas as pd
from pathlib import Path
from jumplion.utils import extract_unique_parts

# generate a small id report for testing
is_tims = True
id_file = "report.parquet"
required_columns = ['Run', 'Precursor.Id', 'RT', 'Q.Value']
if is_tims:
    required_columns = required_columns + ['IM']
id_df = pd.read_parquet(id_file, columns=required_columns, engine='fastparquet')
id_df['Run'] = extract_unique_parts(id_df['Run'].to_list())
rt_mat = id_df.pivot(index='Precursor.Id', columns='Run', values='RT')
rt_mat = rt_mat[(rt_mat.shape[1] - rt_mat.isna().sum(axis=1)) >= 2]
rt_mat_toy = rt_mat.sample(n=int(len(rt_mat)/5), random_state=42)
id_df_toy = id_df[id_df['Precursor.Id'].isin(rt_mat_toy.index.values)]
id_df_toy.to_parquet('id_report_toy.parquet', index=False)

# generate a small spectral library for testing
lib_file = "report-lib.parquet"
lib_df = pd.read_parquet(lib_file, engine='fastparquet')
# Remove decoys
for col_index, header in enumerate(lib_df.columns):
    if re.sub(r"[\s\._-]", "", header.lower()) in [h.lower() for h in ("Decoy", "isDecoy")]:
        decoy_col_name = header
        decoy_values = ['1', 't', 'T']
        decoy_mask = lib_df[decoy_col_name].astype(str).isin(decoy_values)
        n_decoys = decoy_mask.sum()
        logging.warning(f"WARNING: the library contains {n_decoys} decoy precursors, which will be removed")
        lib_df = lib_df[~decoy_mask].copy()
        break
lib_df.drop(columns=['Decoy',
                     'Q.Value', 
                     'Peptidoform.Q.Value', 
                     'PTM.Site.Confidence', 
                     'PG.Q.Value',
                     'Flags'], inplace=True)
lib_df_toy = lib_df[lib_df['Precursor.Id'].isin(id_df_toy['Precursor.Id'].unique())]
lib_df_toy.to_parquet('spectral_library_toy.parquet', index=False)

# generate a small XIC data file
xic_data = "H7_3_S1-E5_1_667.xic.parquet"
xic_df = pd.read_parquet(xic_data, engine='fastparquet')
kept_precs = xic_df['Precursor.Id'].drop_duplicates().sample(n=11, random_state=42).tolist()
kept_precs = kept_precs + ["MGELASESQGSK2"]
xic_df_toy = xic_df[xic_df['Precursor.Id'].isin(kept_precs)]
xic_df_toy.to_parquet('H7_3_S1-E5_1_667.xic_toy.parquet', index=False)

# generate small precursor-XIC mapping files
prec_xic_data = "Precursor.Id_MS2.XIC.Quant.Raw_mapping.parquet"
prec_xic_df = pd.read_parquet(prec_xic_data, engine='fastparquet')
kept_precs = prec_xic_df['target_id'].drop_duplicates().sample(n=11, random_state=42).tolist()
kept_precs = kept_precs + ["INEKPTVVNDYEAAR2"]
file_list = list(Path('.').glob("Precursor.Id_*_mapping.parquet"))
for file in file_list:
    df = pd.read_parquet(file, filters=[('target_id', 'in', kept_precs)])
    df.to_parquet('toy_' + str(file), index=False)

# generate small precursor quantification file
prec_data = "precursor_quantity_summ_table_norm.parquet"
prec_df = pd.read_parquet(prec_data, engine='fastparquet')
kept_prots = prec_df['Protein.Group'].drop_duplicates().sample(n=11, random_state=42).tolist()
kept_prots = kept_prots + ["O14467"]
file_list = list(Path('.').glob("precursor_*_summ_table*"))
for file in file_list:
    prec_df_toy = pd.read_parquet(file, filters=[('Protein.Group', 'in', kept_prots)])
    prec_df_toy.to_parquet('toy_' + str(file), index=False)




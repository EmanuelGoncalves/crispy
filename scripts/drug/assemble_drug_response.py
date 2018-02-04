#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2, venn2_circles

DEPRECATED_DRUG_IDS = 'data/gdsc/drug_single/deprecated_drugids.csv'

DRUG_V17 = 'data/gdsc/drug_single/fully_annotated_384_pipeline_v17_20170825.csv'
DRUG_RS = 'data/gdsc/drug_single/fitted_data_rapid_screen_1536_GDSC_23Jun17_1351.csv'

QC_DRUG_CORR = 'data/gdsc/drug_single/overlaping_drugs_correlation.csv'


def import_v17(file, use_in_publications=True, rmse_thres=0.3):
    df = pd.read_csv(file)

    # Filter by RMSE
    df = df[df['RMSE'] < rmse_thres]

    # Filter in USE_IN_PUBLICATIONS
    if use_in_publications:
        df = df[df['WEBRELEASE'] == 'Y']

    # Version
    df['VERSION'] = 'v17'

    return df


def import_rs(file, rmse_thres=0.3):
    df = pd.read_csv(file)

    # RS data-set is already filtered by use_in_publications
    df['USE_IN_PUBLICATIONS'] = 'Y'

    # Filter by RMSE (threshold = 0.3)
    df = df[df['RMSE'] < rmse_thres]

    # Rename DRUG_ID_lib to DRUG_ID
    df = df.rename(columns={'DRUG_ID_lib': 'DRUG_ID'})

    # Version
    df['VERSION'] = 'RS'

    return df


def merge_drug_dfs(d_17, d_rs, drug_id_remove, deprecated_id):
    index, columns, value = ('DRUG_ID', 'DRUG_NAME', 'VERSION'), ('COSMIC_ID', 'CELL_LINE_NAME'), 'IC50_nat_log'

    # Replace deprecated DRUG_IDs
    d_17 = d_17.replace({'DRUG_ID': deprecated_ids.to_dict()})
    d_rs = d_rs.replace({'DRUG_ID': deprecated_ids.to_dict()})

    # Build tables and exclude mismatching drugs
    drug_v17_matrix_m = pd.pivot_table(d_17, index=index, columns=columns, values=value)
    drug_v17_matrix_m = drug_v17_matrix_m[[idx not in drug_id_remove for idx, dname, dversion in drug_v17_matrix_m.index]]

    drug_rs_matrix_m = pd.pivot_table(d_rs, index=index, columns=columns, values=value)

    drug_matrix_m = pd.concat([drug_v17_matrix_m, drug_rs_matrix_m], axis=0)

    return drug_matrix_m


if __name__ == '__main__':
    # - Deprecated drug ids match list
    deprecated_ids = pd.read_csv(DEPRECATED_DRUG_IDS, index_col=0)['new_id']

    # - Drugs to be excluded for not correlating well between screens
    drug_id_remove = set(pd.read_csv(QC_DRUG_CORR, index_col=0).query('remove_in_v17 == 1').index)
    print('Compounds to remove: ', len(drug_id_remove))

    # - Import drug screens
    d_17 = import_v17(DRUG_V17)
    d_rs = import_rs(DRUG_RS)

    # - Merge screens
    drug_m = merge_drug_dfs(d_17, d_rs, drug_id_remove, deprecated_ids)
    drug_m.to_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv')


#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import scripts as mp

if __name__ == '__main__':
    # - Import ENSG
    ensg = pd.read_csv(mp.BIOMART_HUMAN_ID_TABLE)
    ensg = ensg.groupby('Gene name')['Gene stable ID'].agg(lambda x: set(x)).to_dict()

    # - Import cell lines samplesheet
    s_samplesheet = pd.read_csv(mp.GDSC_RNASEQ_SAMPLESHEET)
    s_samplesheet = pd.Series({
        k: v.split('.')[0] for k, v in s_samplesheet.groupby('expression_matrix_name')['sample_name'].first().to_dict().items()
    })

    # - RNA-seq RPKMs
    rpkm = pd.read_csv(mp.GDSC_RNASEQ_RPKM, index_col=0)

    # - Low-expressed genes: take the highest abundance across the different transcripts
    nexp = pd.DataFrame(
        {g: (rpkm.reindex(ensg[g]) < mp.RPKM_THRES).astype(int).min() for g in ensg if len(ensg[g].intersection(rpkm.index)) > 0}
    ).T

    # - Merged cell lines replicates: take the highest abundance across the different replicates
    nexp = nexp.groupby(s_samplesheet.loc[nexp.columns], axis=1).min()

    # - Export
    nexp.to_csv(mp.NON_EXP)

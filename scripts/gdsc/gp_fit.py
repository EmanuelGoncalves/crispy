#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


if __name__ == '__main__':
    # - Example cell line
    sample = 'AU565'

    bed = pd.read_csv(f'{gdsc.DIR}/bed/{sample}.crispy.bed', sep='\t')

    # - Aggregate by segment
    agg = dict(ratio=np.mean, fold_change=np.mean, sgrna='count')
    bed_seg = bed.groupby(['chr', 'start', 'end']).agg(agg)

    # - Fit
    x, y = bed_seg.query('sgrna >= 10')['ratio'], bed_seg.query('sgrna >= 10')['fold_change']

    kernel = ConstantKernel() * RBF() + WhiteKernel()

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
    gpr = gpr.fit(x.values.reshape(-1, 1), y)

    x_ = np.arange(0, 4.5, .1)
    y_, y_std_ = gpr.predict(x_.reshape(-1, 1), return_std=True)

    #
    bed_ = bed.assign(gp_mean=gpr.predict(bed[['ratio']]))
    bed_ = bed_.assign(corrected=bed_.eval('fold_change - gp_mean').values)
    cy.QCplot.recall_curve_discretise(
        bed_.groupby('gene')['corrected'].mean(), bed_.groupby('gene')['ratio'].mean(), 10
    )

    # - Plot
    # Segments used for fitting
    plt.scatter(
        x, y, c=cy.QCplot.PAL_DBGD[0], alpha=.7, edgecolors='white', lw=.3, label='#(sgRNA) >= 10'
    )

    # Segments non used for fitting
    x_low_sgrna, y_low_sgrna = bed_seg.query('sgrna < 10')['ratio'], bed_seg.query('sgrna < 10')['fold_change']
    plt.scatter(
        x_low_sgrna, y_low_sgrna, c=cy.QCplot.PAL_DBGD[0], marker='X', alpha=.3, edgecolors='white', lw=.3, label='#(sgRNA) < 10'
    )

    # GP fit
    plt.plot(x_, y_, ls='-', lw=1., c=cy.QCplot.PAL_DBGD[1], label='GPR mean')
    plt.fill_between(x_, y_ - y_std_, y_ + y_std_, alpha=0.2, color=cy.QCplot.PAL_DBGD[1], lw=0)

    # Misc
    plt.axhline(0, ls=':', color=cy.QCplot.PAL_DBGD[2], lw=.3)

    plt.xlabel('Segment\ncopy-number ratio')
    plt.ylabel('Segment\nmean CRISPR-Cas9 fold-change')

    plt.title(f'{gpr.kernel_}', fontsize=6)
    plt.suptitle(sample)

    plt.legend(frameon=False)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig(f'reports/gdsc/gpfit.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

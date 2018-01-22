import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from protein_attenuation.enrichment.plot_gsea import plot_gsea


def plot_gsea(e_score, pvalue, dataset, hits, running_hit, filename, title, y1_label='Enrichment score', y2_label='Data value'):
    x, y = np.array(range(len(dataset[0]))), np.array(running_hit)

    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)

    sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])

    # GSEA running hit
    ax1.plot(x, y, '-', c='#3498db')
    ax1.fill_between(x, 0, y, alpha=.2, color='#808080')
    [ax1.axvline(i, c='#e74c3c', lw=.3) for i in x[np.array(hits, dtype='bool')]]
    ax1.axhline(0, c='black', lw=.3, ls='-')
    ax1.set_ylabel(y1_label)
    ax1.set_title('%s\n(score: %.2f, p-val: %.1e)' % (title, e_score, pvalue))
    ax1.get_xaxis().set_visible(False)

    # Data
    ax2.scatter(x, dataset[1], c='#3498db', linewidths=0, s=2)
    ax2.fill_between(x, 0, dataset[1], alpha=.2, color='#808080')
    ax2.axhline(0, c='black', lw=.3, ls='-')
    [ax2.axvline(i, c='#e74c3c', lw=.3) for i in x[np.array(hits, dtype='bool')]]
    ax2.set_ylabel(y2_label)
    ax2.get_xaxis().set_visible(False)

    # General configurations
    sns.despine(bottom=True, ax=ax1)
    sns.despine(bottom=True, ax=ax2)

    # plt.tight_layout()
    ax1.set_xlim([0, len(x)])
    ax2.set_xlim([0, len(x)])

    # Save figure
    plt.gcf().set_size_inches(5, 3)

    if filename.endswith('.png'):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename, bbox_inches='tight')

    plt.close('all')

    print('[INFO] GSEA plot saved: ', filename)


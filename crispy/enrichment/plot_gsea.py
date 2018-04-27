import operator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.gridspec import GridSpec


def plot_gsea(e_score, pvalue, dataset, hits, running_hit, filename, title, y1_label='Enrichment score', y2_label='Data value'):
    dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))

    x, y = np.array(range(len(dataset[0]))), np.array(running_hit)

    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)

    sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])

    # GSEA running hit
    ax1.plot(x, y, '-', c=bipal_dbgd[0])
    ax1.fill_between(x, 0, y, alpha=.2, color=bipal_dbgd[0])
    [ax1.axvline(i, c=bipal_dbgd[1], lw=.1, zorder=0, alpha=.5) for i in x[np.array(hits, dtype='bool')]]
    ax1.axhline(0, c='black', lw=.3, ls='-')
    ax1.set_ylabel(y1_label)
    ax1.set_title('{}\n(e-score={:.2f}, FDR={:.1e})'.format(title, e_score, pvalue))
    ax1.get_xaxis().set_visible(False)

    # Data
    ax2.scatter(x, dataset[1], c=bipal_dbgd[0], linewidths=0, s=2)
    ax2.fill_between(x, 0, dataset[1], alpha=.2, color=bipal_dbgd[0])
    ax2.axhline(0, c='black', lw=.3, ls='-')
    [ax2.axvline(i, c=bipal_dbgd[1], lw=.1, zorder=0, alpha=.5) for i in x[np.array(hits, dtype='bool')]]
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


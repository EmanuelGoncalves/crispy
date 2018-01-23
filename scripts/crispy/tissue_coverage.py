#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# - Tissue distribution
plot_df = ss.loc[list(fc), 'Cancer Type'].value_counts(ascending=True).reset_index().rename(columns={'index': 'Cancer Type', 'Cancer Type': 'Counts'})
plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

plt.barh(plot_df['ypos'], plot_df['Counts'], .8, color=bipal_dbgd[0], align='center')

for x, y in plot_df[['ypos', 'Counts']].values:
    plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=6)

plt.yticks(plot_df['ypos'])
plt.yticks(plot_df['ypos'], plot_df['Cancer Type'])

plt.xlabel('# cell lines')
plt.title('CRISPR/Cas9 screen\n(%d cell lines)' % plot_df['Counts'].sum())

# plt.show()
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/crispy/screen_samples.png', bbox_inches='tight', dpi=600)
plt.close('all')

#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pkg_resources
import seaborn as sns

__version__ = pkg_resources.require('crispy')[0].version

sns.set(style='ticks', context='paper',
        rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in',
            'ytick.direction': 'in'})

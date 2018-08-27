#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import seaborn as sns
from crispy.qc_plot import QCplot
from crispy.utils import Utils, SSGSEA
from crispy.crispy import Crispy, CrispyGaussian


__version__ = '0.2.0'

# - SET STYLE
sns.set(style='ticks', context='paper', rc=QCplot.SNS_RC)


# - HANDLES
__all__ = [
    'Crispy',
    'CrispyGaussian',
    'Utils',
    'QCplot',
    'SSGSEA'
]

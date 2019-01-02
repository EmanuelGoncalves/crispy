#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import seaborn as sns
from crispy.Utils import Utils, SSGSEA
from crispy.CrispyPlot import CrispyPlot
from crispy.QCPlot import QCplot, GSEAplot
from crispy.CopyNumberCorrection import Crispy, CrispyGaussian


__version__ = '0.2.0'

# - SET STYLE
sns.set(style='ticks', context='paper', rc=CrispyPlot.SNS_RC)


# - HANDLES
__all__ = [
    'Crispy',
    'CrispyGaussian',

    'QCplot',
    'CrispyPlot',

    'SSGSEA',
    'GSEAplot',

    'Utils'
]

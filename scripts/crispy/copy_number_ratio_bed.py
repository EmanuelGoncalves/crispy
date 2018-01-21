#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
from crispy.ratio import map_cn

bed_file = 'data/gdsc/wgs/ascat_bed/BB30-HNC.wgs.ascat.bed'

bed_df = map_cn(bed_file).dropna()

gene = 'SCAI'
print(bed_df.loc[gene])

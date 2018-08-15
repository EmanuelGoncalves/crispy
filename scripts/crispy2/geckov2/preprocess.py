#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scripts.crispy2.geckov2 as gecko
from crispy2.crispy import Crispy


if __name__ == '__main__':
    # - Imports
    samplesheet = gecko.get_samplesheet()
    raw_counts = gecko.get_raw_counts()
    copy_number = gecko.get_copy_number_segments()

    lib = gecko.get_crispr_library()

    # -
    s = 'A375-311cas9'
    s_name = samplesheet.loc[s, 'name']

    s_crispy = Crispy(
        raw_counts=raw_counts[[i for i in raw_counts if i.split(' ')[0] == s] + [gecko.PLASMID]],
        copy_number=copy_number.query("Sample == @s_name"),
        library=lib.groupby(['chr', 'start', 'end', 'sgrna'])['PAM'].count().rename('ngenes').reset_index(),
        plasmid=gecko.PLASMID
    )

    fc = s_crispy.correct()

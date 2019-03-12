#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt


class Library:
    def __init__(self):
        self.ddir = pkg_resources.resource_filename("crispy", "data/crispr_libs/")



#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
from scipy.interpolate import interpn


DPATH = pkg_resources.resource_filename("crispy", "data/")


def project_score_sample_map(lib_version="V1.1"):
    ky_v11_manifest = pd.read_csv(
        f"{DPATH}/crispr_manifests/project_score_manifest.csv.gz"
    )

    s_map = []
    for i in ky_v11_manifest.index:
        s_map.append(
            pd.DataFrame(
                dict(
                    model_id=ky_v11_manifest.iloc[i]["model_id"],
                    s_ids=ky_v11_manifest.iloc[i]["library"].split(", "),
                    s_lib=ky_v11_manifest.iloc[i]["experiment_identifier"].split(", "),
                )
            )
        )
    s_map = pd.concat(s_map).set_index("s_lib")
    s_map = s_map.query(f"s_ids == '{lib_version}'")

    return s_map


def density_interpolate(xx, yy):
    data, x_e, y_e = np.histogram2d(xx, yy, bins=20)

    zz = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([xx, yy]).T,
        method="splinef2d",
        bounds_error=False,
    )

    return zz

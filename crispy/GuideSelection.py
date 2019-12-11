#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import ast
import numpy as np
import pandas as pd
from crispy.CRISPRData import Library


class GuideSelection:
    GREEN = dict(offtarget=[1, 0], jacks_thres=1, ruleset2_thres=0.4)
    AMBER = dict(offtarget=[1], jacks_thres=None, ruleset2_thres=None)
    RED = dict(offtarget=[3])

    def __init__(self, masterlib="MasterLib_v1.csv.gz"):
        self.masterlibfile = masterlib
        self.masterlib = self.import_master_library()

    def import_master_library(self):
        df = Library.load_library(self.masterlibfile, set_index=False)

        # Drop sgRNAs that do not align to GRCh38
        df = df.query("Assembly == 'Human (GRCh38)'")

        # Remove sgRNAs that match to multiple genes
        sg_count = df.groupby("WGE_Sequence")["Approved_Symbol"].agg(
            lambda v: len(set(v))
        )
        sg_count = sg_count[sg_count != 1]
        df = df[~df["WGE_Sequence"].isin(sg_count.index)]

        # Remove sgRNAs with no alignment to GRCh38
        df = df.dropna(subset=["Approved_Symbol", "Off_Target"])

        # Remove duplicates (sgRNAs shared across multiple libraries)
        df["Library"] = pd.Categorical(
            df["Library"], ["KosukeYusa", "Avana", "Brunello", "TKOv3"]
        )
        df = df.sort_values("Library")
        df = df.groupby("WGE_Sequence").first().reset_index()

        # Parse off target summaries
        df["Off_Target"] = df["Off_Target"].apply(ast.literal_eval).values

        # Calculate absolute distance of JACKS scores to 1 (similar to gene sgRNAs mean)
        df["JACKS_min"] = abs(df["JACKS"] - 1).values

        # Sort guides according to KS scores
        df = df.sort_values(
            ["KS", "JACKS_min", "RuleSet2"], ascending=[False, True, False]
        )

        return df

    def get_sgrnas(
        self,
        gene,
        offtarget,
        n_guides,
        library=None,
        sortby=None,
        ascending=None,
        query=None,
        dropna=None,
        sgrnas_exclude=None,
    ):
        # Select gene sgRNAs gene
        gguides = self.masterlib.query(f"Approved_Symbol == '{gene}'")

        # Subset to library
        if library is not None:
            gguides = gguides.query(f"Library == '{library}'")

        # List of sgRNAs to exclude
        if sgrnas_exclude is not None:
            gguides = gguides[~gguides["sgRNA_ID"].isin(sgrnas_exclude)]

        # Drop NaNs
        if dropna is not None:
            gguides = gguides.dropna(subset=dropna)

        # Query
        if query is not None:
            gguides = gguides.query(query)

        # Sort guides
        if (sortby is not None) and (ascending is not None):
            gguides = gguides.sort_values(sortby, ascending=ascending)

        # Off-target filter
        gguides = gguides.loc[
            [
                np.all([ot[i] <= v for i, v in enumerate(offtarget)])
                for ot in gguides["Off_Target"]
            ]
        ]

        # Pick top n sgRNAs
        gguides = gguides.head(n_guides)

        return gguides

    def select_sgrnas(
        self,
        gene,
        n_guides=5,
        jacks_thres=1,
        ruleset2_thres=0.4,
        offtarget=None,
        sgrnas_exclude=None,
    ):
        offtarget = [1, 0] if offtarget is None else offtarget

        # KosukeYusa v1.1 sgRNAs
        g_guides = self.get_sgrnas(
            gene=gene,
            library="KosukeYusa",
            offtarget=offtarget,
            n_guides=n_guides,
            query=f"JACKS_min < {jacks_thres}" if jacks_thres is not None else None,
            dropna=["KS"],
            sgrnas_exclude=sgrnas_exclude,
        )

        # Top-up with Avana guides
        if len(g_guides) < n_guides:
            g_guides_avana = self.get_sgrnas(
                gene=gene,
                library="Avana",
                offtarget=offtarget,
                n_guides=n_guides - g_guides.shape[0],
                query=f"JACKS_min < {jacks_thres}" if jacks_thres is not None else None,
                dropna=["KS"],
                sgrnas_exclude=sgrnas_exclude,
            )
            g_guides = pd.concat([g_guides, g_guides_avana], sort=False)

        # Top-up with Brunello guides
        if len(g_guides) < n_guides:
            g_guides_brunello = self.get_sgrnas(
                gene=gene,
                library="Brunello",
                offtarget=offtarget,
                n_guides=n_guides - g_guides.shape[0],
                query=f"RuleSet2 > {ruleset2_thres}" if ruleset2_thres is not None else None,
                sortby="RuleSet2",
                ascending=False,
                sgrnas_exclude=sgrnas_exclude,
            )
            g_guides = pd.concat([g_guides, g_guides_brunello], sort=False)

        # Top-up with TKOv3 guides
        if len(g_guides) < n_guides:
            g_guides_tkov3 = self.get_sgrnas(
                gene=gene,
                library="TKOv3",
                offtarget=offtarget,
                n_guides=n_guides - g_guides.shape[0],
                sgrnas_exclude=sgrnas_exclude,
            )
            g_guides = pd.concat([g_guides, g_guides_tkov3], sort=False)

        return g_guides

    def selection_rounds(self, gene, n_guides=5, do_amber_round=True, do_red_round=True):
        # - Stringest sgRNA selection round
        gguides = self.select_sgrnas(gene=gene, n_guides=n_guides, **self.GREEN)
        gguides = gguides.assign(Confidence="Green")

        # - Middle sgRNA selection round
        if do_amber_round and (len(gguides) < n_guides):
            gguides_amber = self.select_sgrnas(
                gene=gene,
                n_guides=n_guides - gguides.shape[0],
                sgrnas_exclude=set(gguides["sgRNA_ID"]),
                **self.AMBER,
            ).assign(Confidence="Amber")
            gguides = pd.concat([gguides, gguides_amber], sort=False)

        # - Lose sgRNA selection round
        if do_red_round and (len(gguides) < n_guides):
            gguides_red = self.get_sgrnas(
                gene=gene,
                n_guides=n_guides - gguides.shape[0],
                sgrnas_exclude=set(gguides["sgRNA_ID"]),
                **self.RED,
            ).assign(Confidence="Red")
            gguides = pd.concat([gguides, gguides_red], sort=False)

        return gguides

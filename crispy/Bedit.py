#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from crispy import CrispyPlot


class Modifications:
    AA = {
        "A": dict(abbreviation="Ala", name="Alanine"),
        "R": dict(abbreviation="Arg", name="Arginine"),
        "N": dict(abbreviation="Asn", name="Asparagine"),
        "D": dict(abbreviation="Asp", name="Aspartic acid"),
        "C": dict(abbreviation="Cys", name="Cysteine"),
        "Q": dict(abbreviation="Gln", name="Glutamine"),
        "E": dict(abbreviation="Glu", name="Glutamic acid"),
        "G": dict(abbreviation="Gly", name="Glycine"),
        "H": dict(abbreviation="His", name="Histidine"),
        "I": dict(abbreviation="Ile", name="Isoleucine"),
        "M": dict(abbreviation="Met", name="Methionine"),
        "L": dict(abbreviation="Leu", name="Leucine"),
        "K": dict(abbreviation="Lys", name="Lysine"),
        "F": dict(abbreviation="Phe", name="Phenylalanine"),
        "P": dict(abbreviation="Pro", name="Proline"),
        "S": dict(abbreviation="Ser", name="Serine"),
        "T": dict(abbreviation="Thr", name="Threonine"),
        "W": dict(abbreviation="Trp", name="Tryptophan"),
        "Y": dict(abbreviation="Tyr", name="Tyrosine"),
        "V": dict(abbreviation="Val", name="Valine"),
        "*": dict(abbreviation="STOP", name="STOP"),
    }

    @classmethod
    def get_order(cls, field="abbreviation"):
        order = cls.AA.keys()

        if field is not None:
            order = [cls.AA[i][field] for i in order]

        return order

    @classmethod
    def get_names(cls, aas=None, field="abbreviation"):
        if aas is None:
            aas = cls.AA.keys()
        return {aa: cls.AA[aa][field] for aa in aas if aa in cls.AA}


class BaseEditor:
    PAM_WT_CAS9 = "NGG"

    def __init__(
        self, edit_window=None, pam=None, base_edit=None, len_guide=None, name=None
    ):
        self.edit_window = edit_window
        self.pam = pam
        self.base_edit = base_edit
        self.len_guide = len_guide
        self.name = name

    @staticmethod
    def base_complement(nucleotide):
        if nucleotide == "C":
            return "G"
        elif nucleotide == "G":
            return "C"
        elif nucleotide == "A":
            return "T"
        elif nucleotide == "T":
            return "A"

    @staticmethod
    def parse_coordinates(genomic_coordinates):
        chrm, pos = genomic_coordinates.split(":")
        chrm, start, end = chrm, int(pos.split("-")[0]), int(pos.split("-")[1])
        return chrm, start, end

    def assert_guide(self, guide):
        assert (
            len(guide) == self.len_guide
        ), f"Guide size: expected {self.len_guide} got {len(guide)}: {guide}"

        pam_regex = self.pam.replace("N", ".")
        assert re.search(
            f"{pam_regex}$", guide
        ), f"PAM mismatch: expected {self.pam} got {guide[-len(self.pam):]}"

    def split_guide(self, guide):
        start = guide[: self.edit_window[0]]
        edit = guide[self.edit_window[0] : self.edit_window[1]]
        end = guide[self.edit_window[1] : -len(self.pam)]

        pam = guide[-len(self.pam) :]

        return start, edit, end, pam

    def print_guide(self, guide):
        self.assert_guide(guide)

        start, edit, end, pam = self.split_guide(guide)

        print(f"{start} {edit} {end} [{pam}]")

    def edit_guide(self, guide, guide_strand, target_strand):
        start, edit, end, pam = self.split_guide(guide)

        if guide_strand == target_strand:
            edit = edit.replace(
                self.base_complement(self.base_edit[0]),
                self.base_complement(self.base_edit[1]),
            )
        else:
            edit = edit.replace(self.base_edit[0], self.base_edit[1])

        return start + edit + end + pam

    def to_vep(self, guide_original, guide_edited):
        start_win, original, end_win, pam = self.split_guide(guide_original)
        _, edited, _, _ = self.split_guide(guide_edited)

        if original == edited:
            return None

        else:
            idx_start = min([i for i, bp in enumerate(original) if bp != edited[i]])
            idx_end = len(original) - min(
                [i for i, bp in enumerate(original[::-1]) if bp != edited[::-1][i]]
            )

            edit = f"{original[idx_start:idx_end]}/{edited[idx_start:idx_end]}"

            return edit, len(start_win) + idx_start, len(start_win) + (idx_end - 1)

    def list_base_edits(self, guide_original, guide_edited):
        start_win, original, end_win, pam = self.split_guide(guide_original)
        _, edited, _, _ = self.split_guide(guide_edited)

        if original == edited:
            return None

        else:
            return [
                (f"{original[i]}/{edited[i]}", len(start_win) + i)
                for i, bp in enumerate(original)
                if original[i] != edited[i]
            ]


class CytidineBaseEditor(BaseEditor):
    def __init__(
        self,
        edit_window=(3, 8),
        pam=BaseEditor.PAM_WT_CAS9,
        len_guide=23,
        name="Cytidine base editor",
    ):
        super().__init__(
            edit_window=edit_window,
            pam=pam,
            base_edit=("C", "T"),
            len_guide=len_guide,
            name=name,
        )


class AdenineBaseEditor(BaseEditor):
    def __init__(
        self,
        edit_window=(3, 8),
        pam=BaseEditor.PAM_WT_CAS9,
        len_guide=23,
        name="Adenine base editor",
    ):
        super().__init__(
            edit_window=edit_window,
            pam=pam,
            base_edit=("A", "G"),
            len_guide=len_guide,
            name=name,
        )


class BeditPlot(CrispyPlot):
    # TODO: Test situtation where not all 20 aa and stop codon are present
    @classmethod
    def aa_grid(
        cls, dataframe=None, aggfunc=len, index_var="mutant", col_var="wildtype"
    ):
        order = Modifications.get_order(field="abbreviation")

        plot_df = pd.pivot_table(
            dataframe[[index_var, col_var]],
            index=index_var,
            columns=col_var,
            aggfunc=aggfunc,
            fill_value=np.nan,
        ).loc[order, order]

        #
        f, axs = plt.subplots(
            2,
            2,
            sharex="col",
            sharey="row",
            gridspec_kw=dict(height_ratios=[1, 4], width_ratios=[4, 1]),
        )

        g = sns.heatmap(
            plot_df, cmap="Greys", annot=True, cbar=False, linewidths=0.5, ax=axs[1, 0]
        )

        axs[1, 0].set_xlabel("Wild-type")
        axs[1, 0].set_ylabel("Mutant")
        axs[1, 0].grid(lw=0.5, ls="--", color=cls.PAL_DBGD[2])
        sns.despine(right=False, top=False, ax=axs[1, 0])

        for i in [0, 1]:
            i_sums = plot_df.sum(i)

            norm = mpl.colors.Normalize(vmin=0, vmax=i_sums.max())
            colors = [plt.cm.Greys(norm(v)) for v in i_sums]

            barfun = axs[i, i].barh if i else axs[i, i].bar
            barfun(np.arange(len(order)) + 0.5, i_sums, color=colors)

            axs[i, i].set_xlabel("Count" if i else "")
            axs[i, i].set_ylabel("" if i else "Count")
            sns.despine(ax=axs[i, i])

        axs[0, 1].axis("off")

        plt.subplots_adjust(hspace=0, wspace=0)

        return g

    @classmethod
    def aa_countplot(cls, dataframe, index_var="mutant", col_var="wildtype"):
        plot_df = pd.concat(
            [
                dataframe[index_var].value_counts().rename("Mutant"),
                dataframe[col_var].value_counts().rename("Wild-type"),
            ],
            axis=1,
            sort=False,
        )

        plot_df = plot_df.unstack().reset_index()
        plot_df.columns = ["type", "aa", "count"]

        order = Modifications.get_order()

        pal = {"Mutant": cls.PAL_DBGD[0], "Wild-type": cls.PAL_DBGD[2]}

        sns.barplot(
            "count", "aa", "type", data=plot_df, palette=pal, order=order, orient="h"
        )

        plt.grid(lw=0.3, ls="-", color=cls.PAL_DBGD[2], axis="x", zorder=0)

        plt.legend(frameon=False, prop=dict(size=4))

        plt.xlabel("AA count")
        plt.ylabel("")

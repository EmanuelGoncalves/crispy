#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import itertools as it
from crispy.Utils import Utils
from scipy.stats import shapiro, iqr
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import quantile_transform
from statsmodels.stats.multitest import multipletests


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class Sample:
    """
    Import module that handles the sample list (i.e. list of cell lines) and their descriptive information.

    """

    def __init__(
        self,
        index="model_id",
        samplesheet_file="model_list_20200204.csv",
        growth_file="growth_rates_rapid_screen_1536_v1.6.3_02Jun20.csv",
        medium_file="SIDMvsMedia.xlsx",
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ):
        self.index = index

        # Import samplesheet
        self.samplesheet = (
            pd.read_csv(f"{DPATH}/{samplesheet_file}")
            .dropna(subset=[self.index])
            .set_index(self.index)
        )

        # Growth rates
        self.growth = pd.read_csv(f"{DPATH}/{growth_file}")
        self.samplesheet["growth"] = (
            self.growth.groupby(self.index)["GROWTH_RATE"]
            .mean()
            .reindex(self.samplesheet.index)
            .values
        )

        # CRISPR institute
        self.samplesheet["institute"] = (
            pd.read_csv(f"{DPATH}/{institute_file}", index_col=0, header=None)
            .iloc[:, 0]
            .reindex(self.samplesheet.index)
            .values
        )

        # Breakdown tissue type
        self.samplesheet["model_type"] = [
            c if t in ["Lung", "Haematopoietic and Lymphoid"] else t
            for t, c in self.samplesheet[["tissue", "cancer_type"]].values
        ]

        # Screen medium
        self.media = pd.read_excel(f"{DPATH}/{medium_file}")
        self.media = self.media.groupby("SIDM")["Screen Media"].first()
        self.samplesheet["media"] = self.media.reindex(self.samplesheet.index)

    def get_covariates(self, culture_conditions=True, cancer_type=True):
        covariates = []

        # Cell lines culture conditions
        if culture_conditions:
            culture = pd.get_dummies(self.samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cell lines culture conditions
        if cancer_type:
            ctype = pd.get_dummies(self.samplesheet["cancer_type"])
            covariates.append(ctype)

        # Merge covariates
        covariates = pd.concat(covariates, axis=1, sort=False)

        return covariates


class WES:
    def __init__(self, wes_file="WES_variants.csv.gz"):
        self.wes = pd.read_csv(f"{DPATH}/wes/{wes_file}")

    def get_data(self, as_matrix=True, mutation_class=None, recurrence=False):
        df = self.wes.copy()

        # Filter by mutation types
        if mutation_class is not None:
            df = df[df["Classification"].isin(mutation_class)]

        if recurrence:
            df = df[df["Recurrence Filter"] == "Yes"]

        if as_matrix:
            df["value"] = 1

            df = pd.pivot_table(
                df,
                index="Gene",
                columns="model_id",
                values="value",
                aggfunc="first",
                fill_value=0,
            )

        return df

    def filter(
        self,
        subset=None,
        min_events=5,
        as_matrix=True,
        mutation_class=None,
        recurrence=False,
    ):
        df = self.get_data(
            as_matrix=as_matrix, mutation_class=mutation_class, recurrence=recurrence
        )

        # Subset samples
        if subset is not None:
            if as_matrix:
                df = df.loc[:, df.columns.isin(subset)]

            else:
                df = df[df["model_id"].isin(subset)]

            assert df.shape[1] != 0, "No columns after filter by subset"

        # Minimum number of events
        if min_events is not None:
            df = df[df.sum(1) >= min_events]

        return df


class GeneExpression:
    """
    Import module of gene-expression data-set.

    """

    def __init__(
        self,
        voom_file="gexp/rnaseq_voom.csv.gz",
        read_count="gexp/rnaseq_20191101/rnaseq_read_count_20191101.csv",
    ):
        self.voom = pd.read_csv(f"{DPATH}/{voom_file}", index_col=0)
        self.readcount = pd.read_csv(f"{DPATH}/{read_count}", index_col=1).drop(
            columns=["model_id"]
        )
        self.discrete = pd.read_csv(
            f"{DPATH}/GDSC_discretised_table.csv.gz", index_col=0
        )

    def get_data(self, dtype="voom"):
        if dtype == "voom":
            return self.voom.copy()

        elif dtype == "readcount":
            return self.readcount.copy()

        else:
            assert False, f"Dtype {dtype} not supported"

    def filter(self, dtype="voom", subset=None, iqr_range=None, normality=False, lift_gene_ids=True):
        df = self.get_data(dtype=dtype)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by IQR
        if iqr_range is not None:
            iqr_ranges = (
                df.apply(lambda v: iqr(v, rng=iqr_range), axis=1)
                .rename("iqr")
                .to_frame()
            )

            gm_iqr = GaussianMixture(n_components=2).fit(iqr_ranges[["iqr"]])
            iqr_ranges["gm"] = gm_iqr.predict(iqr_ranges[["iqr"]])

            df = df.reindex(iqr_ranges["gm"] != gm_iqr.means_.argmin())

            LOG.info(f"IQR {iqr_range}")
            LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.reindex(normality[0])

        if lift_gene_ids:
            gmap = pd.read_csv(f"{DPATH}/gexp/hgnc-symbol-check.csv").groupby("Input")["Approved symbol"].first()
            df.index = gmap.loc[df.index]

        return df

    def is_not_expressed(self, rpkm_threshold=1, subset=None):
        rpkm = self.filter(dtype="rpkm", subset=subset)
        rpkm = (rpkm < rpkm_threshold).astype(int)
        return rpkm


class DrugResponse:
    """
    Importer module for drug-response measurements acquired at Sanger Institute GDSC (https://cancerrxgene.org).

    """

    SAMPLE_COLUMNS = ["model_id"]
    DRUG_COLUMNS = ["drug_id", "drug_name", "dataset"]

    def __init__(
        self,
        drugresponse_file="drugresponse/DrugResponse_PANCANCER_GDSC1_GDSC2_20200602.csv.gz",
    ):
        # Import and Merge drug response matrix (IC50)
        self.drugresponse = pd.read_csv(f"{DPATH}/{drugresponse_file}")
        self.drugresponse = self.drugresponse[
            ~self.drugresponse["cell_line_name"].isin(["LS-1034"])
        ]

        # Drug max concentration
        self.maxconcentration = self.drugresponse.groupby(self.DRUG_COLUMNS)[
            "max_screening_conc"
        ].first()

    @staticmethod
    def assemble():
        gdsc1 = pd.read_csv(
            f"{DPATH}/drugresponse/fitted_data_screen_96_384_v1.6.0_02Jun20.csv"
        )
        gdsc1 = gdsc1.assign(dataset="GDSC1").query("(RMSE < 0.3)")
        gdsc1 = gdsc1.query("use_in_publications == 'Y'")

        gdsc2 = pd.read_csv(
            f"{DPATH}/drugresponse/fitted_data_rapid_screen_1536_v1.6.3_02Jun20.csv"
        )
        gdsc2 = gdsc2.assign(dataset="GDSC2").query("(RMSE < 0.3)")
        gdsc2 = gdsc2.query("use_in_publications == 'Y'")

        columns = set(gdsc1).intersection(gdsc2)

        drespo = pd.concat([gdsc1[columns], gdsc2[columns]], axis=0, ignore_index=True)

        drespo.to_csv(
            f"{DPATH}/drugresponse/DrugResponse_PANCANCER_GDSC1_GDSC2_20200602.csv.gz",
            compression="gzip",
            index=False,
        )

    def get_data(self, dtype="ln_IC50"):
        data = pd.pivot_table(
            self.drugresponse,
            index=self.DRUG_COLUMNS,
            columns=self.SAMPLE_COLUMNS,
            values=dtype,
            fill_value=np.nan,
        )

        return data

    def filter(
        self,
        dtype="ln_IC50",
        subset=None,
        min_events=3,
        min_meas=0.75,
        max_c=0.5,
        filter_min_observations=False,
        filter_max_concentration=False,
        filter_combinations=False,
    ):
        # Drug max screened concentration
        df = self.get_data(dtype="ln_IC50")
        d_maxc = np.log(self.maxconcentration * max_c)

        # - Filters
        # Subset samples
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by mininum number of observations
        if filter_min_observations:
            df = df[df.count(1) > (df.shape[1] * min_meas)]

        # Filter by max screened concentration
        if filter_max_concentration:
            df = df[[sum(df.loc[i] < d_maxc.loc[i]) >= min_events for i in df.index]]

        # Filter combinations
        if filter_combinations:
            df = df[[" + " not in i[1] for i in df.index]]

        return self.get_data(dtype=dtype).loc[df.index, df.columns]


class BioGRID:
    def __init__(
        self,
        biogrid_file="BIOGRID-ALL-3.5.180.tab2.zip",
        organism=9606,
        etype="physical",
        stypes_exclude=None,
        homodymers_exclude=True,
    ):
        self.etype = etype
        self.organism = organism
        self.homodymers_exclude = homodymers_exclude
        self.stypes_exclude = (
            {"Affinity Capture-RNA", "Protein-RNA"}
            if stypes_exclude is None
            else stypes_exclude
        )

        # Import
        self.biogrid = pd.read_csv(f"{DPATH}/{biogrid_file}", sep="\t")

        # Filter by organism
        self.biogrid = self.biogrid[
            self.biogrid["Organism Interactor A"] == self.organism
        ]
        self.biogrid = self.biogrid[
            self.biogrid["Organism Interactor B"] == self.organism
        ]

        # Filter by type of interaction
        if self.etype is not None:
            self.biogrid = self.biogrid[
                self.biogrid["Experimental System Type"] == self.etype
            ]

        # Exlude experimental systems
        self.biogrid = self.biogrid[
            ~self.biogrid["Experimental System"].isin(self.stypes_exclude)
        ]

        # Exclude homodymers
        if self.homodymers_exclude:
            self.biogrid = self.biogrid[
                self.biogrid["Official Symbol Interactor A"]
                != self.biogrid["Official Symbol Interactor B"]
            ]

        # Build set of interactions (both directions, i.e. p1-p2, p2-p1)
        self.biogrid = {
            (p1, p2)
            for p1, p2 in self.biogrid[
                ["Official Symbol Interactor A", "Official Symbol Interactor B"]
            ].values
        }
        self.biogrid = {
            (p1, p2) for p in self.biogrid for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
        }


class PPI:
    """
    Module used to import protein-protein interaction network

    """

    def __init__(
        self,
        string_file="ppi/9606.protein.links.full.v11.0.txt.gz",
        string_alias_file="ppi/9606.protein.aliases.v11.0.txt.gz",
    ):
        self.string_file = string_file
        self.string_alias_file = string_alias_file

    @classmethod
    def ppi_annotation(
        cls, df, ppi, target_thres=5, y_var="y_id", x_var="x_id", ppi_var="x_ppi"
    ):
        genes_source = set({g for v in df[y_var].dropna() for g in v.split(";")}).intersection(
            set(ppi.vs["name"])
        )
        genes_target = set({g for v in df[x_var].dropna() for g in v.split(";")}).intersection(
            set(ppi.vs["name"])
        )

        # Calculate distance between drugs and genes in PPI
        dist_g_g = {
            g: pd.Series(
                ppi.shortest_paths(source=g, target=genes_target)[0], index=genes_target
            ).to_dict()
            for g in genes_source
        }

        def gene_gene_annot(g_source, g_target):
            if str(g_source) == "nan" or str(g_target) == "nan":
                res = np.nan

            elif len(set(g_source.split(";")).intersection(g_target.split(";"))) > 0:
                res = "T"

            elif g_source not in genes_source:
                res = "-"

            elif g_target not in genes_target:
                res = "-"

            else:
                g_st_min = np.min([dist_g_g[gs][gt] for gs in g_source.split(";") for gt in g_target.split(";")])
                res = cls.ppi_dist_to_string(g_st_min, target_thres)

            return res

        # Annotate drug regressions
        df = df.assign(
            x_ppi=[
                gene_gene_annot(g_source, g_target)
                for g_source, g_target in df[[y_var, x_var]].values
            ]
        )
        df = df.rename(columns=dict(x_ppi=ppi_var))

        return df

    @staticmethod
    def ppi_dist_to_string(d, target_thres):
        if d == 0:
            res = "T"

        elif d == np.inf:
            res = "-"

        elif d < target_thres:
            res = f"{int(d)}"

        else:
            res = f"{int(target_thres)}+"

        return res

    def build_string_ppi(self, score_thres=900, export_pickle=None):
        import igraph

        # ENSP map to gene symbol
        gmap = pd.read_csv(f"{DPATH}/{self.string_alias_file}", sep="\t")
        gmap = gmap[["BioMart_HUGO" in i.split(" ") for i in gmap["source"]]]
        gmap = (
            gmap.groupby("string_protein_id")["alias"].agg(lambda x: set(x)).to_dict()
        )
        gmap = {k: list(gmap[k])[0] for k in gmap if len(gmap[k]) == 1}
        logging.getLogger("DTrace").info(f"ENSP gene map: {len(gmap)}")

        # Load String network
        net = pd.read_csv(f"{DPATH}/{self.string_file}", sep=" ")

        # Filter by moderate confidence
        net = net[net["combined_score"] > score_thres]

        # Filter and map to gene symbol
        net = net[
            [
                p1 in gmap and p2 in gmap
                for p1, p2 in net[["protein1", "protein2"]].values
            ]
        ]
        net["protein1"] = [gmap[p1] for p1 in net["protein1"]]
        net["protein2"] = [gmap[p2] for p2 in net["protein2"]]
        LOG.info(f"String: {len(net)}")

        #  String network
        net_i = igraph.Graph(directed=False)

        # Initialise network lists
        edges = [(px, py) for px, py in net[["protein1", "protein2"]].values]
        vertices = list(set(net["protein1"]).union(net["protein2"]))

        # Add nodes
        net_i.add_vertices(vertices)

        # Add edges
        net_i.add_edges(edges)

        # Add edge attribute score
        net_i.es["score"] = list(net["combined_score"])

        # Simplify
        net_i = net_i.simplify(combine_edges="max")
        LOG.info(net_i.summary())

        # Export
        if export_pickle is not None:
            net_i.write_pickle(export_pickle)

        return net_i

    @staticmethod
    def ppi_corr(ppi, m_corr, m_corr_thres=None):
        """
        Annotate PPI network based on Pearson correlation between the vertices of each edge using
        m_corr data-frame and m_corr_thres (Pearson > m_corr_thress).

        :param ppi:
        :param m_corr:
        :param m_corr_thres:
        :return:
        """
        # Subset PPI network
        ppi = ppi.subgraph([i.index for i in ppi.vs if i["name"] in m_corr.index])

        # Edge correlation
        crispr_pcc = np.corrcoef(m_corr.loc[ppi.vs["name"]].values)
        ppi.es["corr"] = [crispr_pcc[i.source, i.target] for i in ppi.es]

        # Sub-set by correlation between vertices of each edge
        if m_corr_thres is not None:
            ppi = ppi.subgraph_edges(
                [i.index for i in ppi.es if abs(i["corr"]) > m_corr_thres]
            )

        LOG.info(ppi.summary())

        return ppi

    @classmethod
    def get_edges(cls, ppi, nodes, corr_thres, norder):
        # Subset network
        ppi_sub = ppi.copy().subgraph_edges(
            [e for e in ppi.es if abs(e["corr"]) >= corr_thres]
        )

        # Nodes that are contained in the network
        nodes = {v for v in nodes if v in ppi_sub.vs["name"]}
        assert len(nodes) > 0, "None of the nodes is contained in the PPI"

        # Nodes neighborhood
        neighbor_nodes = {
            v for n in nodes for v in ppi_sub.neighborhood(n, order=norder)
        }

        # Build subgraph
        subgraph = ppi_sub.subgraph(neighbor_nodes)

        # Build data-frame
        nodes_df = pd.DataFrame(
            [
                {
                    "source": subgraph.vs[e.source]["name"],
                    "target": subgraph.vs[e.target]["name"],
                    "r": e["corr"],
                }
                for e in subgraph.es
            ]
        ).sort_values("r")

        return nodes_df


class CORUM:
    def __init__(
        self,
        corum_file="coreComplexes.txt",
        organism="Human",
        homodymers_exclude=True,
        protein_subset=None,
    ):
        self.organism = organism
        self.homodymers_exclude = homodymers_exclude
        self.protein_subset = protein_subset

        # Load CORUM DB
        self.db = pd.read_csv(f"{DPATH}/{corum_file}", sep="\t")
        self.db = self.db.query(f"Organism == '{organism}'")
        self.db_name = self.db.groupby("ComplexID")["ComplexName"].first()

        # Melt into list of protein pairs (both directions, i.e. p1-p2, p2-p1)
        self.db_melt = self.melt_ppi()

        # Map to gene symbols
        self.gmap = self.map_gene_name()
        self.db_melt_symbol = {
            (self.gmap.loc[p1, "GeneSymbol"], self.gmap.loc[p2, "GeneSymbol"]): i
            for (p1, p2), i in self.db_melt.items()
            if p1 in self.gmap.index and p2 in self.gmap.index
        }

        # Exclude homodymers
        if self.homodymers_exclude:
            self.db_melt_symbol = {
                (p1, p2): i for (p1, p2), i in self.db_melt_symbol.items() if p1 != p2
            }

        # Subset interactions
        if self.protein_subset is not None:
            self.db_melt_symbol = {
                (p1, p2): i
                for (p1, p2), i in self.db_melt_symbol.items()
                if p1 in self.protein_subset and p2 in self.protein_subset
            }

    def melt_ppi(self, idx_id="ComplexID", idx_sub="subunits(UniProt IDs)"):
        db_melt = self.db[[idx_id, idx_sub]].copy()
        db_melt[idx_sub] = db_melt[idx_sub].apply(
            lambda v: list(it.permutations(v.split(";"), 2))
        )
        db_melt = {p: i for i, c in db_melt[[idx_id, idx_sub]].values for p in c}
        return db_melt

    @staticmethod
    def map_gene_name(index_col="Entry"):
        idmap = pd.read_csv(f"{DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap["Gene names  (primary )"].apply(
            lambda v: v.split("; ")[0] if str(v).lower() != "nan" else v
        )

        return idmap


class HuRI:
    def __init__(self, ppi_file="HuRI.tsv", idmap_file="HuRI_biomart_idmap.tsv"):
        self.huri = pd.read_csv(f"{DPATH}/{ppi_file}", sep="\t", header=None)

        # Convert to a set of pairs {(p1, p2), ...}
        self.huri = {(p1, p2) for p1, p2 in self.huri.values}

        # Map ids
        idmap = pd.read_csv(f"{DPATH}/{idmap_file}", sep="\t", index_col=0)[
            "Gene name"
        ].to_dict()
        self.huri = {
            (idmap[p1], idmap[p2])
            for p1, p2 in self.huri
            if p1 in idmap and p2 in idmap
        }

        # Remove self interactions
        self.huri = {(p1, p2) for p1, p2 in self.huri if p1 != p2}

        # Build set of interactions (both directions, i.e. p1-p2, p2-p1)
        self.huri = {
            (p1, p2) for p in self.huri for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
        }


class Metabolomics:
    def __init__(self, metab_file="metabolomics/CCLE_metabolomics_20190502.csv"):
        m_ss = Sample().samplesheet
        m_ss = m_ss.reset_index().dropna(subset=["BROAD_ID"]).set_index("BROAD_ID")

        # Import
        self.metab = pd.read_csv(f"{DPATH}/{metab_file}")
        self.metab["model_id"] = self.metab["DepMap_ID"].replace(m_ss["model_id"])
        self.metab = self.metab.groupby("model_id").mean().T

    def get_data(self):
        return self.metab.copy()

    def filter(
        self,
        dtype="protein",
        subset=None,
        normality=False,
        iqr_range=None,
        perc_measures=None,
        quantile_normalise=False,
    ):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df


class Proteomics:
    def __init__(
        self,
        protein_matrix="proteomics/E0022_P06_Protein_Matrix_ProNorM.tsv.gz",
        protein_raw_matrix="proteomics/E0022_P06_Protein_Matrix_Raw_Mean_Intensities.tsv.gz",
        protein_mean_raw="proteomics/E0022_P06_Protein_Mean_Raw_Intensities.tsv",
        protein_rep_corr="proteomics/E0022_P06_final_reps_correlation.csv",
        manifest="proteomics/E0022_P06_final_sample_map.txt",
        samplesheet="proteomics/E0022_P06_samplehseet.csv",
        broad_tmt="proteomics/broad_tmt.csv.gz",
        coread_tmt="proteomics/proteomics_coread_processed_parsed.csv",
        hgsc_prot="proteomics/hgsc_cell_lines_proteomics.csv",
        brca_prot="proteomics/brca_cell_lines_proteomics_preprocessed.csv",
    ):
        self.ss = pd.read_csv(f"{DPATH}/{samplesheet}", index_col=0)

        deprecated_ids = self.map_deprecated()

        # Import manifest
        self.manifest = pd.read_csv(f"{DPATH}/{manifest}", index_col=0, sep="\t")

        # Remove excluded samples
        self.exclude_man = self.manifest[~self.manifest["SIDM"].isin(self.ss.index)]
        self.manifest = self.manifest[~self.manifest.index.isin(self.exclude_man.index)]

        # Replicate correlation
        self.reps = pd.read_csv(f"{DPATH}/{protein_rep_corr}", index_col=0).iloc[:, 0]

        # Import mean protein abundance
        self.protein_raw = pd.read_csv(
            f"{DPATH}/{protein_raw_matrix}", sep="\t", index_col=0
        )
        self.peptide_raw_mean = pd.read_csv(
            f"{DPATH}/{protein_mean_raw}", sep="\t", index_col=0
        ).iloc[:, 0]

        # Import imputed protein levels
        self.protein = pd.read_csv(f"{DPATH}/{protein_matrix}", sep="\t", index_col=0).T

        self.protein["Protein"] = (
            self.protein.reset_index()["index"]
            .replace(deprecated_ids["Entry name"])
            .values
        )
        self.protein = self.protein.set_index("Protein")
        self.protein = self.protein.rename(
            columns=self.manifest.groupby("Cell_line")["SIDM"].first()
        )

        exclude_controls = [
            "Control_HEK293T_lys",
            "Control_HEK293T_std_H002",
            "Control_HEK293T_std_H003",
        ]
        self.protein = self.protein.drop(columns=exclude_controls)

        # Import Broad TMT data-set
        self.broad = pd.read_csv(f"{DPATH}/{broad_tmt}", compression="gzip")
        self.broad = (
            self.broad.dropna(subset=["Gene_Symbol"])
            .groupby("Gene_Symbol")
            .agg(np.nanmean)
        )

        # Import CRC COREAD TMT
        self.coread = pd.read_csv(f"{DPATH}/{coread_tmt}", index_col=0)
        self.coread = self.coread.loc[
            :, self.coread.columns.isin(self.ss["model_name"])
        ]

        coread_ss = self.ss[self.ss["model_name"].isin(self.coread.columns)]
        coread_ss = coread_ss.reset_index().set_index("model_name")
        self.coread = self.coread.rename(columns=coread_ss["model_id"])

        # Import HGSC proteomics
        self.hgsc = (
            pd.read_csv(f"{DPATH}/{hgsc_prot}")
            .dropna(subset=["Gene names"])
            .drop(columns=["Majority protein IDs"])
        )
        self.hgsc = self.hgsc.groupby("Gene names").mean()
        self.hgsc = self.hgsc.loc[:, self.hgsc.columns.isin(self.ss["model_name"])]

        hgsc_ss = self.ss[self.ss["model_name"].isin(self.hgsc.columns)]
        hgsc_ss = hgsc_ss.reset_index().set_index("model_name")
        self.hgsc = self.hgsc.rename(columns=hgsc_ss["model_id"])

        # Import BRCA proteomics
        self.brca = pd.read_csv(f"{DPATH}/{brca_prot}", index_col=0)
        self.brca = self.brca.loc[:, self.brca.columns.isin(self.ss["model_name"])]

        brca_ss = self.ss[self.ss["model_name"].isin(self.brca.columns)]
        brca_ss = brca_ss.reset_index().set_index("model_name")
        self.brca = self.brca.rename(columns=brca_ss["model_id"])

    def get_data(self, dtype="protein", map_ids=True, quantile_normalise=False):
        if dtype.lower() == "protein":
            data = self.protein.copy()

        else:
            assert False, f"{dtype} not supported"

        if quantile_normalise:
            data = pd.DataFrame(
                quantile_transform(data, ignore_implicit_zeros=True),
                index=data.index,
                columns=data.columns,
            )

        if map_ids:
            pmap = self.map_gene_name().reindex(data.index)["GeneSymbol"].dropna()

            data = data[data.index.isin(pmap.index)]
            data = data.groupby(pmap.reindex(data.index)).mean()

        return data

    def filter(
        self,
        dtype="protein",
        subset=None,
        normality=False,
        iqr_range=None,
        perc_measures=None,
        quantile_normalise=False,
    ):
        df = self.get_data(dtype=dtype, quantile_normalise=quantile_normalise)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by IQR
        if iqr_range is not None:
            iqr_ranges = (
                df.apply(lambda v: iqr(v, rng=iqr_range), axis=1)
                .rename("iqr")
                .to_frame()
            )

            gm_iqr = GaussianMixture(n_components=2).fit(iqr_ranges[["iqr"]])
            iqr_ranges["gm"] = gm_iqr.predict(iqr_ranges[["iqr"]])

            df = df.reindex(iqr_ranges["gm"] != gm_iqr.means_.argmin())

            LOG.info(f"IQR {iqr_range}")
            LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.reindex(normality[0])

        # Filter by number of obvservations
        if perc_measures is not None:
            df = df[df.count(1) > (perc_measures * df.shape[1])]

        return df

    @staticmethod
    def map_deprecated():
        return pd.read_csv(
            f"{DPATH}/uniprot_human_idmap_deprecated.tab", sep="\t", index_col=0
        )

    @staticmethod
    def map_gene_name(index_col="Entry name"):
        idmap = pd.read_csv(f"{DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap["Gene names  (primary )"].apply(
            lambda v: v.split("; ")[0] if str(v).lower() != "nan" else v
        )

        return idmap

    def calculate_mean_protein_intensities(
        self, peptide_matrix_raw="proteomics/E0022_P06_Peptide_Matrix_Raw.tsv.gz"
    ):
        peptide_raw = pd.read_csv(
            f"{DPATH}/{peptide_matrix_raw}", sep="\t", index_col=0
        ).T

        peptide_raw_mean = (
            peptide_raw.pipe(np.log2).groupby(self.manifest["SIDM"], axis=1).mean()
        )
        peptide_raw_mean = peptide_raw_mean.groupby(
            [p.split("=")[0] for p in peptide_raw_mean.index]
        ).mean()
        peptide_raw_mean = peptide_raw_mean.mean(1).sort_values()

        pmap = (
            self.map_gene_name().reindex(peptide_raw_mean.index)["GeneSymbol"].dropna()
        )

        peptide_raw_mean = peptide_raw_mean[peptide_raw_mean.index.isin(pmap.index)]
        peptide_raw_mean = peptide_raw_mean.groupby(
            pmap.reindex(peptide_raw_mean.index)
        ).mean()

        return peptide_raw_mean

    def replicates_correlation(
        self, reps_file="proteomics/E0022_P06_Protein_Matrix_Replicate_ProNorM.tsv.gz"
    ):
        reps = pd.read_csv(f"{DPATH}/{reps_file}", sep="\t", index_col=0).T

        reps_corr = {}

        for n, df in reps.groupby(self.manifest["SIDM"], axis=1):
            df_corr = df.corr()
            df_corr = pd.DataFrame(df_corr.pipe(np.triu, k=1)).replace(0, np.nan)
            reps_corr[n] = df_corr.unstack().dropna().mean()

        reps_corr = pd.Series(reps_corr, name="RepsCorrelation").sort_values(
            ascending=False
        )

        return reps_corr


class CRISPR:
    """
    Importer module for CRISPR-Cas9 screens acquired at Sanger and Broad Institutes.

    """

    def __init__(
        self,
        fc_file="crispr/CRISPR_corrected_qnorm_20191108.csv.gz",
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
        merged_file="crispr/CRISPRcleanR_FC.txt.gz",
    ):
        self.crispr = pd.read_csv(f"{DPATH}/{fc_file}", index_col=0)
        self.institute = pd.read_csv(
            f"{DPATH}/{institute_file}", index_col=0, header=None
        ).iloc[:, 0]

        sid = (
            Sample()
            .samplesheet.reset_index()
            .dropna(subset=["BROAD_ID"])
            .groupby("BROAD_ID")["model_id"]
            .first()
        )

        self.merged = pd.read_csv(f"{DPATH}/{merged_file}", index_col=0, sep="\t")
        self.merged_institute = pd.Series(
            {c: "Broad" if c.startswith("ACH-") else "Sanger" for c in self.merged}
        )

        self.merged = self.merged.rename(columns=sid)
        self.merged_institute = self.merged_institute.rename(index=sid)

    def get_data(self, scale=True, dtype="merged"):
        if dtype == "merged":
            df = self.merged.copy()
        else:
            df = self.crispr.copy()

        if scale:
            df = self.scale(df)

        return df

    def filter(
        self,
        dtype="merged",
        subset=None,
        scale=True,
        std_filter=False,
        abs_thres=None,
        drop_core_essential=False,
        min_events=5,
        drop_core_essential_broad=False,
        binarise_thres=None,
    ):
        df = self.get_data(scale=True, dtype=dtype)

        # - Filters
        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by scaled scores
        if abs_thres is not None:
            df = df[(df.abs() > abs_thres).sum(1) >= min_events]

        # Filter out core essential genes
        if drop_core_essential:
            df = df[~df.index.isin(Utils.get_adam_core_essential())]

        if drop_core_essential_broad:
            df = df[~df.index.isin(Utils.get_broad_core_essential())]

        # - Subset matrices
        x = self.get_data(scale=scale, dtype=dtype).reindex(
            index=df.index, columns=df.columns
        )

        if binarise_thres is not None:
            x = (x < binarise_thres).astype(int)

        if std_filter:
            x = x.reindex(x.std(1) > 0)

        return x

    @staticmethod
    def scale(df, essential=None, non_essential=None, metric=np.median):
        if essential is None:
            essential = Utils.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = Utils.get_non_essential_genes(return_series=False)

        assert (
            len(essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with essential list"

        assert (
            len(non_essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with non essential list"

        essential_metric = metric(df.reindex(essential).dropna(), axis=0)
        non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

        df = df.subtract(non_essential_metric).divide(
            non_essential_metric - essential_metric
        )

        return df


class Mobem:
    """
    Import module for Genomic binary feature table (containing mutations and copy-number calls)
    Iorio et al., Cell, 2016.

    """

    def __init__(
        self, mobem_file="mobem/PANCAN_mobem.csv.gz", drop_factors=True, add_msi=True
    ):
        self.sample = Sample()

        idmap = (
            self.sample.samplesheet.reset_index()
            .dropna(subset=["COSMIC_ID", "model_id"])
            .set_index("COSMIC_ID")["model_id"]
        )

        mobem = pd.read_csv(f"{DPATH}/{mobem_file}", index_col=0)
        mobem = mobem[mobem.index.astype(str).isin(idmap.index)]
        mobem = mobem.set_index(idmap[mobem.index.astype(str)].values)

        if drop_factors is not None:
            mobem = mobem.drop(columns={"TISSUE_FACTOR", "MSI_FACTOR", "MEDIA_FACTOR"})

        if add_msi:
            self.msi = self.sample.samplesheet.reindex(mobem.index)["msi_status"]
            mobem["msi_status"] = (self.msi == "MSI-H").astype(int)[mobem.index].values

        self.mobem = mobem.astype(int).T

    def get_data(self):
        return self.mobem.copy()

    def filter(self, subset=None, min_events=5):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Minimum number of events
        df = df[df.sum(1) >= min_events]

        return df

    @staticmethod
    def mobem_feature_to_gene(f):
        if f.endswith("_mut"):
            genes = {f.split("_")[0]}

        elif f.startswith("gain.") or f.startswith("loss."):
            genes = {
                g
                for fs in f.split("..")
                if not (fs.startswith("gain.") or fs.startswith("loss."))
                for g in fs.split(".")
                if g != ""
            }

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")

        return genes

    @staticmethod
    def mobem_feature_type(f):
        if f.endswith("_mut"):
            return "Mutation"

        elif f.startswith("gain."):
            return "CN gain"

        elif f.startswith("loss."):
            return "CN loss"

        elif f == "msi_status":
            return f

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")


class CopyNumber:
    def __init__(
        self,
        cnv_file="copy_number/cnv_abs_copy_number_picnic_20191101.csv.gz",
        gistic_file="copy_number/cnv_gistic_20191101.csv.gz",
        calculate_deletions=False,
        calculate_amplifications=False,
    ):
        self.ss_obj = Sample()

        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}", index_col=0)

        self.ploidy = self.ss_obj.samplesheet["ploidy"]

        self.gistic = pd.read_csv(
            f"{DPATH}/{gistic_file}", index_col="gene_symbol"
        ).drop(columns=["gene_id"])

        if calculate_deletions:
            self.copynumber_del = pd.DataFrame(
                {
                    s: self.copynumber[s].apply(
                        lambda v: CopyNumber.is_deleted(v, self.ploidy[s])
                    )
                    for s in self.copynumber
                    if s in self.ploidy
                }
            )

        if calculate_amplifications:
            self.copynumber_amp = pd.DataFrame(
                {
                    s: self.copynumber[s].apply(
                        lambda v: CopyNumber.is_amplified(v, self.ploidy[s])
                    )
                    for s in self.copynumber
                    if s in self.ploidy
                }
            )

    def get_data(self, dtype="matrix"):
        if dtype == "del":
            res = self.copynumber_del.copy()

        elif dtype == "amp":
            res = self.copynumber_amp.copy()

        elif dtype == "gistic":
            res = self.gistic.copy()

        else:
            res = self.copynumber.copy()

        return res

    def filter(self, subset=None, dtype="cn"):
        df = self.get_data(dtype=dtype)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    def ploidy_from_segments(
        self, seg_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"
    ):
        copynumber_seg = pd.read_csv(f"{DPATH}/{seg_file}")

        return pd.Series(
            {
                s: self.calculate_ploidy(df)[1]
                for s, df in copynumber_seg.groupby("model_id")
            }
        )

    @classmethod
    def genomic_instability(
        cls, seg_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"
    ):
        # Import segments
        cn_seg = pd.read_csv(f"{DPATH}/{seg_file}")

        # Use only autosomal chromosomes
        cn_seg = cn_seg[~cn_seg["chr"].isin(["chrX", "chrY"])]
        cn_seg = cn_seg[~cn_seg["chr"].isin([23, 24])]

        # Segment length
        cn_seg = cn_seg.assign(length=cn_seg.eval("end - start"))

        # Calculate genomic instability
        s_instability = {}
        for s, df in cn_seg.groupby("model_id"):
            s_ploidy = np.round(cls.calculate_ploidy(df), 0)

            s_chr = []
            # c, c_df = list(df.groupby("chr"))[0]
            for c, c_df in df.groupby("chr"):
                c_gain = (
                    c_df[c_df["copy_number"] > s_ploidy]["length"].sum()
                    / c_df["length"].sum()
                )
                c_loss = (
                    c_df[c_df["copy_number"] < s_ploidy]["length"].sum()
                    / c_df["length"].sum()
                )
                s_chr.append(c_gain + c_loss)

            s_instability[s] = np.mean(s_chr)

        s_instability = pd.Series(s_instability)
        return s_instability

    @classmethod
    def calculate_ploidy(cls, cn_seg):
        # Use only autosomal chromosomes
        cn_seg = cn_seg[~cn_seg["chr"].isin(["chrX", "chrY"])]
        cn_seg = cn_seg[~cn_seg["chr"].isin([23, 24])]

        ploidy = cls.weighted_median(cn_seg["copy_number"], cn_seg["length"])

        return ploidy

        # cn_seg = cn_seg.assign(length=cn_seg["end"] - cn_seg["start"])
        # cn_seg = cn_seg.assign(
        #     cn_by_length=cn_seg["length"] * (cn_seg["copy_number"] + 1)
        # )
        #
        # chrm = (
        #     cn_seg.groupby("chr")["cn_by_length"]
        #     .sum()
        #     .divide(cn_seg.groupby("chr")["length"].sum())
        #     - 1
        # )
        #
        # ploidy = (cn_seg["cn_by_length"].sum() / cn_seg["length"].sum()) - 1
        #
        # return chrm, ploidy

    @staticmethod
    def weighted_median(data, weights):
        # Origingal code: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
        data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
        s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(s_weights)

        if any(weights > midpoint):
            w_median = (data[weights == np.max(weights)])[0]

        else:
            cs_weights = np.cumsum(s_weights)
            idx = np.where(cs_weights <= midpoint)[0][-1]
            if cs_weights[idx] == midpoint:
                w_median = np.mean(s_data[idx : idx + 2])
            else:
                w_median = s_data[idx + 1]

        return w_median

    @staticmethod
    def is_amplified(
        cn, ploidy, cn_threshold_low=5, cn_thresholds_high=9, ploidy_threshold=2.7
    ):
        if (ploidy <= ploidy_threshold) and (cn >= cn_threshold_low):
            return 1

        elif (ploidy > ploidy_threshold) and (cn >= cn_thresholds_high):
            return 1

        else:
            return 0

    @staticmethod
    def is_deleted(cn, ploidy, ploidy_threshold=2.7):
        if (ploidy <= ploidy_threshold) and (cn == 0):
            return 1

        elif (ploidy > ploidy_threshold) and (cn < (ploidy - ploidy_threshold)):
            return 1

        else:
            return 0


class Methylation:
    """
    Import module for Illumina Methylation 450k arrays
    """

    def __init__(
        self, methy_gene_promoter="methylation/methy_beta_gene_promoter.csv.gz"
    ):
        self.methy_promoter = pd.read_csv(f"{DPATH}/{methy_gene_promoter}", index_col=0)

    def get_data(self):
        return self.methy_promoter.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    @staticmethod
    def discretise(beta):
        if beta < 0.33:
            return "umethylated"
        elif beta > 0.66:
            return "methylated"
        else:
            return "hemimethylated"


class CopyNumberSegmentation:
    def __init__(
        self, cnv_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"
    ):
        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}")

    def get_data(self):
        return self.copynumber.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df[df["model_id"].isin(subset)]

        return df


class PipelineResults:
    def __init__(
        self,
        results_dir,
        import_fc=False,
        import_bagel=False,
        import_mageck=False,
        mageck_fdr_thres=0.1,
        fc_thres=-0.5,
    ):
        self.results_dir = results_dir

        if import_fc:
            self.fc_thres = fc_thres
            self.fc, self.fc_c, self.fc_cq, self.fc_s, self.fc_b = (
                self.import_fc_results()
            )
            LOG.info("Fold-changes imported")

        if import_bagel:
            self.bf, self.bf_q, self.bf_b = self.import_bagel_results()
            LOG.info("BAGEL imported")

        if import_mageck:
            self.mageck_fdr_thres = mageck_fdr_thres
            self.mdep_fdr, self.mdep_bin = self.import_mageck_results()
            LOG.info("MAGeCK imported")

    def import_bagel_results(self):
        # Bayesian factors
        bf = pd.read_csv(
            f"{self.results_dir}/_BayesianFactors.tsv", sep="\t", index_col=0
        )

        # Quantile normalised bayesian factors
        bf_q = pd.read_csv(
            f"{self.results_dir}/_scaledBayesianFactors.tsv", sep="\t", index_col=0
        )

        # Binarised bayesian factors
        bf_b = pd.read_csv(
            f"{self.results_dir}/_binaryDepScores.tsv", sep="\t", index_col=0
        )

        return bf, bf_q, bf_b

    def import_fc_results(self):
        # Fold-changes
        fc = pd.read_csv(f"{self.results_dir}/_logFCs.tsv", sep="\t", index_col=0)

        # Copy-number corrected fold-changes
        fc_c = pd.read_csv(
            f"{self.results_dir}/_corrected_logFCs.tsv", sep="\t", index_col=0
        )

        # Quantile normalised copy-number corrected fold-changes
        fc_cq = pd.read_csv(
            f"{self.results_dir}/_qnorm_corrected_logFCs.tsv", sep="\t", index_col=0
        )

        # Scale fold-changes
        fc_s = CRISPR.scale(fc)

        # Fold-change binary
        fc_b = (fc_s < self.fc_thres).astype(int)

        return fc, fc_c, fc_cq, fc_s, fc_b

    def import_mageck_results(self):
        # Dependencies FDR
        mdep_fdr = pd.read_csv(
            f"{self.results_dir}/_MageckFDRs.tsv", sep="\t", index_col=0
        )
        mdep_bin = (mdep_fdr < self.mageck_fdr_thres).astype(int)

        return mdep_fdr, mdep_bin

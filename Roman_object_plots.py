ROMAN = r"""
   ___                         _____  __  ___  __________
  / _ \___  __ _  ___ ____    / __/ |/ / / _ \/  _/_  __/
 / , _/ _ \/  ' \/ _ `/ _ \  _\ \/    / / ___// /  / /
/_/|_|\___/_/_/_/\_,_/_//_/ /___/_/|_/ /_/  /___/ /_/
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context="talk",
    style="ticks",
    font="serif",
    palette="colorblind",
    color_codes=True,
)

FIGURE_PREFIX = "figures/"
LINESTYLE_CYCLE = itertools.cycle(
    [
        "solid",
        "dashed",
        "dotted",
        # "dashdot",
        # (0, (3, 5, 1, 5, 1, 5)), #dot-dot-dashed
    ]
)
# S_N_MAX_CUT = 2
# PLASTICC_FILE = "data_release/PIP_PIPPIN_ROMAN_TRANS_ROMAN_PLAsTiCC/PIP_PIPPIN_ROMAN_TRANS_ROMAN_PLAsTiCC.DUMP"
# CC_FILE = "data_release/PIP_PIPPIN_ROMAN_TRANS_SUPERNOVA/PIP_PIPPIN_ROMAN_TRANS_SUPERNOVA.DUMP"
# SNIA_FILE = "data_release/PIP_PIPPIN_ROMAN_TRANS_SUPERNOVA/PIP_PIPPIN_ROMAN_TRANS_SUPERNOVA.DUMP"
# # https://github.com/LSSTDESC/elasticc/blob/main/alert_schema/elasticc_origmap.txt
# GENTYPES_PLASTICC = {
#     40: "SLSN",  # NONIaMODEL0
#     42: "TDE",  # NONIaMODEL1
#     45: "ILOT",  # NONIaMODEL2
#     # 46: "CART",  NONIaMODEL3
#     50: "KN",  # NONIaMODEL4
#     59: "PISN",  # NONIaMODEL5
# }
# GENTYPES_SN = {
#     30: "CCSN",
#     11: "91bg-like",
#     12: "SN Iax",
#     10: "SN Ia",
# }
# # GENTYPES_IA = {10: "SNIa"}
#
# results = pd.DataFrame(columns=["Transient", "GENTYPE", "Total Detected", "Mean S/N", "Minimum LC Points", "Median LC Points", "Maximum LC Points", "Median Redshift"])
#
# transients = pd.read_csv(
#     PLASTICC_FILE,
#     sep="\s+",
#     comment="#",
#     na_values=[-9, -99, -99999, 99, 99999],
#     index_col=1,
# )
# transients.drop(columns="VARNAMES:", inplace=True)
# # transients.dropna(subset=['PEAKMAG_Y', 'peak_mag_J'], inplace=True)
# transients["MAGERR_MAX_Y"] = np.nan  # add a column, to be used later.
#
# sn = pd.read_csv(
#     SNIA_FILE,
#     sep="\s+",
#     comment="#",
#     na_values=[-9, -99, -99999, 99, 99999],
#     index_col=1,
# )
# sn.drop(columns="VARNAMES:", inplace=True)
# # sn.dropna(subset=['PEAKMAG_Y', 'peak_mag_J'], inplace=True)
# sn["MAGERR_MAX_Y"] = np.nan  # add a column, to be used later.
#
# # Do a S/N at max cut.
# # transients = transients[transients["snr_max_Y"] >= S_N_MAX_CUT]
# # transients = transients[transients["snr_max_J"] >= S_N_MAX_CUT]
# # # ccsn = ccsn[ccsn["snr_max_Y"] >= S_N_MAX_CUT]
# # # ccsn = ccsn[ccsn["snr_max_J"] >= S_N_MAX_CUT]
# # sn = sn[sn["snr_max_Y"] >= S_N_MAX_CUT]
# # sn = sn[sn["snr_max_J"] >= S_N_MAX_CUT]
# # transients = transients[transients["SNRMAX"] >= S_N_MAX_CUT]
# # sn = sn[sn["SNRMAX"] >= S_N_MAX_CUT]
#
# #################
# # Find some CIDs
# ################
# # sort low to high
# transients.sort_values("snr_max_Y", inplace=True)
# # ccsn.sort_values("snr_max_Y", inplace=True)
# sn.sort_values("snr_max_Y", inplace=True)
#
# for gentypes, data in ((GENTYPES_PLASTICC, transients), (GENTYPES_SN, sn)):
#     for key in gentypes.keys():
#         print(gentypes[key])
#         midpoint = len(data[data["GENTYPE"] == key]) // 2
#         try:
#           print("low", data[data["GENTYPE"] == key].iloc[0].name)
#           print("mid", data[data["GENTYPE"] == key].iloc[midpoint].name)
#           print("max", data[data["GENTYPE"] == key].iloc[-1].name)
#           print("Total objserved", data[data["GENTYPE"] == key].shape[0])
#           print("Median S/N", data[data["GENTYPE"] == key].iloc[midpoint].snr_max_Y)
#           print("Min points", data[data["GENTYPE"] == key].Nobjs.min())
#           print("Median points", data[data["GENTYPE"] == key].Nobjs.median())
#           print("Max points", data[data["GENTYPE"] == key].Nobjs.max())
#           print("Meadian Redshift", data[data["GENTYPE"] == key].z_cmb.median())
#           print("Max Redshift", data[data["GENTYPE"] == key].z_cmb.max())
#           print("")
#           results = pd.concat([results,
#             pd.DataFrame({"Transient": gentypes[key],
#               "GENTYPE": key,
#               "Total Detected": data[data["GENTYPE"] == key].shape[0],
#               "Median S/N": data[data["GENTYPE"] == key].iloc[midpoint].snr_max_Y,
#               "Minimum LC Points": data[data["GENTYPE"] == key].Nobjs.min(),
#               "Median LC Points": data[data["GENTYPE"] == key].Nobjs.median(),
#               "Maximum LC Points": data[data["GENTYPE"] == key].Nobjs.max(),
#               "Median Redshift": data[data["GENTYPE"] == key].z_cmb.median(),
#             }, index=[key])
#           ])
#         except IndexError:
#           print("too much nan")
#
# results.sort_values("GENTYPE", inplace=True)
# # results = results['Mean S/N'].apply(np.round)
# results['Minimum LC Points'] = results['Minimum LC Points'].apply(int)
# results['Median LC Points'] = results['Median LC Points'].apply(int)
# results['Maximum LC Points'] = results['Maximum LC Points'].apply(int)
# print(results)
# results.to_latex("results.tex", index=False, float_format="%.2f")


OBJECT_FILE = "data_release/hourglass_objects.parquet"
PHOTOMETRY_FILE = "data_release/hourglass_photometry.parquet"

objs = pd.read_parquet(OBJECT_FILE)
phots = pd.read_parquet(PHOTOMETRY_FILE)
print(objs.head())
print(phots.head())

results = pd.DataFrame(
    columns=[
        "Transient",
        "Total Detected",
        "Median S/N",
        # "Minimum LC Points",
        # "Median LC Points",
        # "Maximum LC Points",
        "Median Redshift",
    ]
)

for i, class_ in enumerate(objs["class"].unique()):
    sub_dataset = objs[objs["class"] == class_]
    print(class_, sub_dataset.snr_max_Y.median())
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                {
                    "Transient": class_.replace("_", " "),
                    "Total Detected": sub_dataset.shape[0],
                    "Median S/N": np.median(sub_dataset["snr_max_Y"].values),
                    # "Minimum LC Points": sub_dataset.n_obs.min(),
                    # "Median LC Points": sub_dataset.n_obs.median(),
                    # "Maximum LC Points": sub_dataset.n_obs.max(),
                    "Median Redshift": sub_dataset.z_cmb.median(),
                },
                index=[i],
            ),
        ]
    )

# results.sort_values("GENTYPE", inplace=True)
# results = results['Mean S/N'].apply(np.round)
# results["Minimum LC Points"] = results["Minimum LC Points"].apply(int)
# results["Median LC Points"] = results["Median LC Points"].apply(int)
# results["Maximum LC Points"] = results["Maximum LC Points"].apply(int)
print(results)
results.to_latex("results.tex", index=False, float_format="%.2f")


###################
# HISTOGRAM PLOTS
###################

# ALL transients
fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
for class_ in objs["class"].unique():
    # value = GENTYPES_PLASTICC[key]
    # n = transients[transients["GENTYPE"] == key].shape[0]
    objs[objs["class"] == class_].hist(
        "z_cmb",
        bins="auto",
        label=f"{class_}",  # , N={n:,}",
        ax=ax,
        histtype="step",
        linestyle=(next(LINESTYLE_CYCLE)),
    )
# for key in GENTYPES_SN.keys():
#     value = GENTYPES_SN[key]
#     n = sn[sn["GENTYPE"] == key].shape[0]
#     sn[sn["GENTYPE"] == key].hist(
#         "z_cmb",
#         bins="auto",
#         label=f"{value}",  # , N={n:,}",
#         ax=ax,
#         histtype="step",
#         linestyle=(next(LINESTYLE_CYCLE)),
#     )
# ax.set_xlim(0.09, 5)
ax.set_xlabel("Redshift")
# ax.set_xscale("log")
# ax.set_xticks([0.1, 1.0])
ax.set_title("")
ax.set_ylabel("Frequency")
ax.set_yscale("log")
# plt.legend(bbox_to_anchor=(0, 1.4), loc="upper left", ncol=2, fontsize="small")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
plt.savefig(FIGURE_PREFIX + "Roman_trans_z.pdf")

# rare transients
many_transients = ["CCSN", "SN_Ia", "SN_Iax", "SNIa-91bg"]
fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
for class_ in objs["class"].unique():
    if class_ not in many_transients + ["Fixed_mag", "AGN"]:
        objs[objs["class"] == class_].hist(
            "z_cmb",
            bins="auto",
            label=f"{class_}".replace("_", " "),  # , N={n:,}",
            ax=ax,
            histtype="step",
            linestyle=(next(LINESTYLE_CYCLE)),
            linewidth=3,
        )
# ax.set_xlim(0.09, 5)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title("")
plt.legend(bbox_to_anchor=(0, 1.1), loc="upper left", ncol=5, fontsize="small")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
plt.savefig(FIGURE_PREFIX + "Roman_trans_z_plasticc.pdf")

# many transients
fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
for class_ in objs["class"].unique():
    if class_ in many_transients:
        objs[objs["class"] == class_].hist(
            "z_cmb",
            bins="auto",
            label=f"{class_}".replace("_", " "),  # , N={n:,}",
            ax=ax,
            histtype="step",
            linestyle=(next(LINESTYLE_CYCLE)),
            linewidth=3,
        )
# ax.set_xlim(0.09, 5)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title("")
plt.legend(
    bbox_to_anchor=(0, 1.1),
    loc="upper left",
    ncol=len(many_transients),
    fontsize="small",
)
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
plt.savefig(FIGURE_PREFIX + "Roman_trans_z_sn.pdf")

# # ALL - fraction per z-bin
# fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
# for key in GENTYPES_PLASTICC.keys():
#     value = GENTYPES_PLASTICC[key]
#     n = transients[transients["GENTYPE"] == key].shape[0]
#     transients[transients["GENTYPE"] == key].hist(
#         "z_cmb",
#         bins="auto",
#         label=f"{value}, N={n:,}",
#         ax=ax,
#         histtype="step",
#         density=True,
#         linestyle=(next(LINESTYLE_CYCLE)),
#     )
# for key in GENTYPES_SN.keys():
#     value = GENTYPES_SN[key]
#     n = sn[sn["GENTYPE"] == key].shape[0]
#     sn[sn["GENTYPE"] == key].hist(
#         "z_cmb",
#         bins="auto",
#         label=f"{value}, N={n:,}",
#         ax=ax,
#         histtype="step",
#         density=True,
#         linestyle=(next(LINESTYLE_CYCLE)),
#     )
# ax.set_xlim(0.3, 3)
# ax.set_xlabel("Redshift")
# ax.set_xscale("log")
# ax.set_xticks([0.1, 1.0])
# ax.set_title("")
# ax.set_ylabel("Density")
# ax.set_yscale("log")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
# plt.savefig(FIGURE_PREFIX + "Roman_trans_z_frac.pdf")

# CCSN
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "CCSN"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title(f"CCSN, N={objs[objs['class'] == 'CCSN'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_ccsn_z.pdf")

# SNIa
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "SN_Ia"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title(f"SN Ia, N={objs[objs['class'] == 'SN_Ia'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_snia_z.pdf")

# KN
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "KN"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_title(f"KN, N={objs[objs['class']=='KN'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_kn_z.pdf")

# TDE
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "TDE"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title(f"TDE, N={objs[objs['class'] == 'TDE'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_tde_z.pdf")

# SLSN
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "SLSN-I"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title(f"SLSN-I, N={objs[objs['class'] == 'SNLS-I'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_slsn_z.pdf")

# PISN
fig, ax = plt.subplots(tight_layout=True)
objs[objs["class"] == "PISN"].hist(
    "z_cmb",
    bins="auto",
    ax=ax,
    histtype="step",
)
ax.set_xlim(0.3)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title(f"PISN, N={objs[objs['class'] == 'PISN'].shape[0]:,}")
plt.savefig(FIGURE_PREFIX + "Roman_pisn_z.pdf")


# TDE + KN
# fig, ax = plt.subplots(tight_layout=True)
# transients[transients["GENTYPE"] == 42].hist(
#     "z_cmb",
#     bins="auto",
#     ax=ax,
#     histtype="step",
#     label=f"TDE, N={transients[transients['GENTYPE'] == 42].shape[0]:,}",
# )
# transients[transients["GENTYPE"] == 50].hist(
#     "z_cmb",
#     bins="auto",
#     ax=ax,
#     histtype="step",
#     label=f"KN, N={transients[transients['GENTYPE'] == 50].shape[0]:,}",
# )
# ax.set_xlim(0.3)
# ax.set_xlabel("Redshift")
# ax.set_ylabel("Frequency")
# ax.set_title("")
# plt.legend()
# plt.savefig(FIGURE_PREFIX + "Roman_tde+kn_z.pdf")
#
#
# # PISN + SLSN
# fig, ax = plt.subplots(tight_layout=True)
# transients[transients["GENTYPE"] == 59].hist(
#     "z_cmb",
#     bins="auto",
#     ax=ax,
#     histtype="step",
#     label=f"PISN, N={transients[transients['GENTYPE'] == 59].shape[0]:,}",
# )
# transients[transients["GENTYPE"] == 40].hist(
#     "z_cmb",
#     bins="auto",
#     ax=ax,
#     histtype="step",
#     label=f"SLSN, N={transients[transients['GENTYPE'] == 40].shape[0]:,}",
# )
# ax.set_xlim(0.3)
# ax.set_xlabel("Redshift")
# ax.set_ylabel("Frequency")
# ax.set_title("")
# plt.legend()
# plt.savefig(FIGURE_PREFIX + "Roman_pisn+slsn_z.pdf")


###############
# S/N HIST PLOT
###############

# ALL - S/N
# fig, ax = plt.subplots(tight_layout=True)
# for key in GENTYPES_PLASTICC.keys():
#     value = GENTYPES_PLASTICC[key]
#     n = transients[transients["GENTYPE"] == key].shape[0]
#     transients.loc[transients["GENTYPE"] == key, "MAGERR_MAX_Y"] = (
#         1 / transients.loc[transients["GENTYPE"] == key, "snr_max_Y"]
#     )
#     transients[transients["GENTYPE"] == key].hist(
#         "MAGERR_MAX_Y",
#         # "snr_max_Y",
#         bins=np.arange(0, 0.6, 0.05),
#         label=f"{value}",
#         ax=ax,
#         histtype="step",
#         linestyle=(next(LINESTYLE_CYCLE)),
#     )
# for key in GENTYPES_SN.keys():
#     value = GENTYPES_SN[key]
#     n = sn[sn["GENTYPE"] == key].shape[0]
#     sn.loc[sn["GENTYPE"] == key, "MAGERR_MAX_Y"] = (
#         1 / sn.loc[sn["GENTYPE"] == key, "snr_max_Y"]
#     )
#     sn[sn["GENTYPE"] == key].hist(
#         "MAGERR_MAX_Y",
#         # "snr_max_Y",
#         bins=np.arange(0, 0.6, 0.05),
#         label=f"{value}",
#         ax=ax,
#         histtype="step",
#     )
# # ax.set_xlim(1, 550)
# ax.set_xlabel(r"$\sigma_{Y}$ at max (mag)")
# # ax.set_xscale("log")
# ax.set_xlim([0, 0.55])
# ax.set_title("")
# ax.set_ylabel("Frequency")
# ax.set_yscale("log")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
# # plt.show()
# plt.savefig(FIGURE_PREFIX + "Roman_trans_SNRY.pdf")


####################
# OTHER PLOTS
####################

# peakmag vs z (all)
# x = np.array([])
# y = np.array([])
# fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
# for key in GENTYPES_PLASTICC.keys():
#     value = GENTYPES_PLASTICC[key]
#     n = transients[transients["GENTYPE"] == key].shape[0]
#     # ax.plot(transients[transients["GENTYPE"] == key]["z_cmb"], transients[transients["GENTYPE"] == key]["peak_mag_Y"], ".", alpha=0.2, label=f"{value}")
#     x = np.append(x, transients[transients["GENTYPE"] == key]["z_cmb"].to_numpy())
#     y = np.append(y, transients[transients["GENTYPE"] == key]["peak_mag_Y"].to_numpy())
# for key in GENTYPES_SN.keys():
#     value = GENTYPES_SN[key]
#     n = sn[sn["GENTYPE"] == key].shape[0]
#     # ax.plot(sn[sn["GENTYPE"] == key]["z_cmb"], sn[sn["GENTYPE"] == key]["peak_mag_Y"], ".", alpha=0.2, label=f"{value}")
#     x = np.append(x, sn[sn["GENTYPE"] == key]["z_cmb"].to_numpy())
#     y = np.append(y, sn[sn["GENTYPE"] == key]["peak_mag_Y"].to_numpy())
# sns.kdeplot(x=x, y=y, levels=[0.0275, 0.1587, 0.5, 0.8413, 0.9725])
# ax.set_xlim([0, 3])
# ax.set_ylim([18, 31])
# ax.set_xlabel("Redshift")
# ax.set_ylabel("Peak Magnitude Y-band")
# plt.savefig(FIGURE_PREFIX + "peakmag-z-all.pdf")


# peakmag vs z (loop)
for value in objs["class"].unique():
    if value == "Fixed_mag":
        continue
    fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
    n = objs[objs["class"] == value].shape[0]
    # ax.plot(transients[transients["GENTYPE"] == key]["z_cmb"], transients[transients["GENTYPE"] == key]["peak_mag_Y"], ".", alpha=0.2, label=f"{value}")
    sns.kdeplot(
        x=objs[objs["class"] == value]["z_cmb"],
        y=objs[objs["class"] == value]["peak_mag_Y"],
        label=f"{value}",
        levels=[0.0275, 0.1587, 0.5, 0.8413, 0.9725],
    )
    ax.set_xlim([0, 3])
    ax.set_ylim([18, 31])
    plt.gca().invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Peak Magnitude Y-band")
    plt.savefig(FIGURE_PREFIX + f"peakmag-z-{value}.pdf")


# SNR vs Peakmag (SNIa)

value = "SN_Ia"
# FIELD contains WIDE or DEEP
n = objs[objs["class"] == value].shape[0]
wide_mask = [
    True if "WIDE" in x else False for x in objs[objs["class"] == value]["field"].values
]
deep_mask = [
    True if "DEEP" in x else False for x in objs[objs["class"] == value]["field"].values
]

fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
ax.plot(
    objs[objs["class"] == value]["peak_mag_Y"][wide_mask],
    objs[objs["class"] == value]["snr_max_Y"][wide_mask],
    ".",
    alpha=0.5,
    label="Y-band",
)
ax.plot(
    objs[objs["class"] == value]["peak_mag_J"][wide_mask],
    objs[objs["class"] == value]["snr_max_J"][wide_mask],
    "^",
    alpha=0.5,
    label="J-band",
)
plt.axhline(5, c="black")
plt.grid()
ax.set_xlim(17.5, 30.5)
ax.set_ylim(1, 700)
ax.set_xlabel("Peak Magnitude")
ax.set_ylabel("S/N at Max")
ax.set_yscale("log")
ax.set_title("100 s exposures")
plt.legend(framealpha=1)
plt.savefig(FIGURE_PREFIX + "SNR-peakmag-wide.pdf")

fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
ax.plot(
    objs[objs["class"] == value]["peak_mag_Y"][deep_mask],
    objs[objs["class"] == value]["snr_max_Y"][deep_mask],
    ".",
    alpha=0.5,
    label="Y-band",
)
ax.plot(
    objs[objs["class"] == value]["peak_mag_J"][deep_mask],
    objs[objs["class"] == value]["snr_max_J"][deep_mask],
    "^",
    alpha=0.5,
    label="J-band",
)
plt.axhline(5, c="black")
plt.grid()
ax.set_xlim(17.5, 30.5)
ax.set_ylim(1, 700)
ax.set_xlabel("Peak Magnitude")
ax.set_ylabel("S/N at Max")
ax.set_yscale("log")
ax.set_title("300 s exposures")
plt.legend(framealpha=1)
plt.savefig(FIGURE_PREFIX + "SNR-peakmag-deep.pdf")


# Find a prism candidate
# key = 10  # SNIa
# prism_mask = [
#     True if "PRISM" in x else False for x in sn[sn["GENTYPE"] == key]["field"].values
# ]
# prism_TDE = sn[sn["GENTYPE"] == key][prism_mask]
# prism_TDE.sort_values("snr_max_J", inplace=True)
# # for index, row in prism_TDE.iterrows():
# #   if 0.95 < row ["z_cmb"] and row["z_cmb"] < 1.05:
# #     print(f"z~1 SNIa {row.name}")
# #   if 1.45 < row ["z_cmb"] and row["z_cmb"] < 1.55:
# #     print(f"z~1.5 SNIa {row.name}")
# interst = int(len(prism_TDE) * 0.97)  # get x% percentile
# print("")
# print("SN Ia with prism", prism_TDE.iloc[interst].name)
# print(prism_TDE.iloc[interst])
# #
# key = 30  # CCSN
# prism_mask = [
#     True if "PRISM" in x else False for x in sn[sn["GENTYPE"] == key]["field"].values
# ]
# prism_CCSN = sn[sn["GENTYPE"] == key][prism_mask]
# prism_CCSN.sort_values("snr_max_J", inplace=True)
# interst = int(len(prism_CCSN) * 0.98)  # get x% percentile
# print("")
# print("CCSN with prism", prism_CCSN.iloc[interst].name)
# print(prism_CCSN.iloc[interst])
#
# key = 42  # TDE
# prism_mask = [
#     True if "PRISM" in x else False
#     for x in transients[transients["GENTYPE"] == key]["field"].values
# ]
# prism_TDE = transients[transients["GENTYPE"] == key][prism_mask]
# prism_TDE.sort_values("snr_max_J", inplace=True)
# # print("Highest S/N TDE with prism", prism_TDE.iloc[-1].name)
# interst = int(len(prism_TDE) * 0.98)  # get x% percentile
# print("")
# print("TDE with prism", prism_TDE.iloc[interst].name)
# print(prism_TDE.iloc[interst])

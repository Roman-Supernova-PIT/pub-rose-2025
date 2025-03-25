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

Y_SNR_VALUE = 10
FIGURE_PREFIX = "figures/"
OBJECT_FILE = "data_release/hourglass_objects.parquet"
DES_FILE = "DES-SN5YR_HD.csv"

bins = np.arange(0, 3.1, 0.05)

objs = pd.read_parquet(OBJECT_FILE)
des = pd.read_csv(DES_FILE)

fig, ax = plt.subplots(tight_layout=True, figsize=(9.4, 6.8))
for class_ in objs["class"].unique():
    if class_ == "SN_Ia":
        objs[
            np.logical_and(objs["class"] == class_, objs["snr_max_Y"] > Y_SNR_VALUE)
        ].hist(
            "z_cmb",
            bins=bins,
            label="Roman",  # , N={n:,}",
            ax=ax,
            histtype="step",
            linestyle="solid",
            linewidth=3,
        )
des.hist(
    "zCMB",
    bins=bins,
    label="DES",  # , N={n:,}",
    ax=ax,
    histtype="step",
    linestyle="dashed",
    linewidth=3,
)
# ax.set_xlim(0.09, 5)
ax.set_xlabel("Redshift")
ax.set_ylabel("Frequency")
ax.set_title("")
# plt.legend(bbox_to_anchor=(0, 1.4), loc="upper left", ncol=2, fontsize="small")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
plt.legend()
plt.savefig(FIGURE_PREFIX + "Roman_DES.pdf")

print(
    objs[
        np.logical_and(objs["class"] == "SN_Ia", objs["snr_max_Y"] > Y_SNR_VALUE)
    ].shape
)

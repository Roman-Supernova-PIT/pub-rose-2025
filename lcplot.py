from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from rich import print
import seaborn as sns

sns.set_theme(
    # context="talk",
    style="ticks",
    font="serif",
    palette="colorblind",
    color_codes=True,
)

FIGURE_PREFIX = "figures/"

OBJECT_FILE = "data_release/hourglass_objects.parquet"
PHOTOMETRY_FILE = "data_release/hourglass_photometry.parquet"

objs = pd.read_parquet(OBJECT_FILE)
phots = pd.read_parquet(PHOTOMETRY_FILE)
print(objs.head())
print(phots.head())


def mag(fluxcal):
    return -2.5 * np.log10(fluxcal) + 27.5


for class_ in objs["class"].unique():
    sub_dataset = objs[objs["class"] == class_]

    if class_ in ["ILOT"]:
        q = 0.54
    elif class_ in ["KN"]:
        q = 0.75
    elif class_ in ["SN_Iax"]:
        q = 0.50
    elif class_ in ["SNIa-91bg"]:
        q = 0.52
    elif class_ in ["CCSN"]:
        q = 0.52
    elif class_ in ["SLSN-I"]:
        q = 0.52
    else:
        q = 0.5
    median_snr = sub_dataset["snr_max_Y"].quantile(q, interpolation="nearest")
    median_objects = sub_dataset[sub_dataset["snr_max_Y"] == median_snr]

    median_lightcurve = phots[phots["cid"] == median_objects["cid"].values[0]]
    plt.figure()
    if "Z" in median_lightcurve["band"].unique():
        bands = ["R", "Z", "Y", "J"]
    if "H" in median_lightcurve["band"].unique():
        bands = ["Y", "J", "H", "F"]
    for band in bands:
        lc = median_lightcurve[median_lightcurve["band"] == band]
        lc.loc[:, "mag"] = mag(lc["fluxcal"].values)
        lc.loc[:, "mag_err"] = np.abs(lc["fluxcal_err"].values / lc["fluxcal"].values)
        lc.loc[lc["mag_err"] > 0.5, "mag"] = np.nan
        lc.loc[lc["mag_err"] > 0.5, "mag_err"] = np.nan
        lc.loc[lc["mag_err"] > 0.5, "mjd"] = np.nan
        lc.sort_values("mjd", inplace=True)
        plt.errorbar(
            lc["mjd"].values,
            lc["mag"].values,
            lc["mag_err"].values,
            fmt=".-",
            label=band,
        )
    # plt.title(f"{name}, cid = {snid},  $z = {redshift:.2f}$")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("MJD")
    plt.ylabel("Apparent Magnitude (mag)")  # arbitrary zero point of 27.5 mag
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_PREFIX
        + f"{class_}-{median_objects.index.values[0]}-{median_objects['z_cmb'].values[0]:.2f}.pdf",
        bbox_inches="tight",
    )

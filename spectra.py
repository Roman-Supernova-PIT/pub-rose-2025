from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
import seaborn as sns

sns.set_theme(
    context="talk",
    style="ticks",
    font="serif",
    palette="colorblind",
    color_codes=True,
)

SNR_PRECENTILE = 0.8
CLASS_ = "SN_Ia"
FIGURE_PREFIX = "figures/"

OBJECT_FILE = "data_release/hourglass_objects.parquet"
SPEC_FILE = "data_release/hourglass_spectra.parquet"

objs = pd.read_parquet(OBJECT_FILE)
spec = pd.read_parquet(SPEC_FILE)
objs = objs[objs["class"] == CLASS_]
prism_mask = []
for _, row in objs.iterrows():
    prism_mask.append("PRISM" in row["field"])
objs = objs[prism_mask]

objs = objs.sort_values("snr_max_Y")
snr = objs["snr_max_Y"].quantile(SNR_PRECENTILE, interpolation="nearest")
object = objs[objs["snr_max_Y"] == snr]
spec_timeseries = spec[spec["cid"] == object["cid"].values[0]]
print(object)
print(spec.columns)
print(spec_timeseries)
print(object.index[0])


offset = 0
fig, ax = plt.subplots(tight_layout=True)
fig.set_figheight(15)

for i, row in spec_timeseries.iterrows():
    phase = int(row["mjd"] - object["peak_mjd"])
    if phase < -10:
        continue
    if phase > 30:
        continue

    wavelength = (row["lam_min"] + row["lam_max"]) / 2
    flux = row["flam"]

    ax.plot(
        wavelength[10:-10] / 10_000,
        flux[10:-10] - offset,
        color="black",
        label=phase,
    )
    plt.annotate(str(phase) + " days", (1.5, -offset - 2e-19))

    offset += 1e-18

ax.set_xlabel("Wavelength ($\mu$m)")
ax.set_ylabel("F$_{\lambda}$ + offset")
ax.get_yaxis().set_ticks([])
# plt.legend()
plt.savefig(
    FIGURE_PREFIX
    + f"spec-{CLASS_}-cid{object.index[0]}-{object['field'].values[0]}-{object['z_cmb'].values[0]:.3f}-ymag{object['peak_mag_Y'].values[0]:.1f}-Trans.pdf"
)

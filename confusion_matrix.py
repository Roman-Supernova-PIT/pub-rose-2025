from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)

sns.set_theme(
    context="talk",
    style="ticks",
    font="serif",
    palette="colorblind",
    color_codes=True,
)


SCONE_CUT = 0.9
ZBINS = 20

DATA_RELASE = Path("data_release")
CLASSIFICATION_FILE = DATA_RELASE / "merged_0.csv"
FIGURE_PREFIX = Path("figures")
OBJECT_FILE = DATA_RELASE / "hourglass_objects.parquet"
objs = pd.read_parquet(OBJECT_FILE)
objs = objs[np.logical_or(objs["class"] == "SN_Ia", objs["class"] == "CCSN")]

objs["SCONE"] = [1 if x >= SCONE_CUT else 0 for x in objs["scone_prob_Ia"].values]
objs["TRUTH"] = [1 if x == "SN_Ia" else 0 for x in objs["class"].values]
print(objs[["SCONE", "TRUTH"]].describe())

# data = pd.read_csv(SN_FILE, sep="\s+", comment="#", na_values=[-9, -99], index_col=1)
# data.drop(columns="VARNAMES:", inplace=True)
# data = pd.concat([data, predictions], axis=1)

# mask = (data["SNRMAX2"] > S_N_MAX_CUT).to_numpy()

# cm = confusion_matrix(
#     data[mask].IA,
#     data[mask].SCONE,
#     # labels=["SN Ia", "CCSN"],
#     normalize="pred",  # shows purity, True_Positive/(True_Positive + False_Positive), in the off diagonal
# )
cm = confusion_matrix(
    objs.TRUTH,
    objs.SCONE,
    # normalize="pred",  # shows purity, True_Positive/(True_Positive + False_Positive), in the off diagonal
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["CCSN", "SN Ia"]
)  # labels for [0, 1]
# disp.plot(values_format=".1%")
disp.plot(values_format=",")
# disp.ax_.set_title("SCONE on Roman Simulations")
# plt.show()
plt.savefig(FIGURE_PREFIX / "SCONEonRoman_confusion.pdf", bbox_inches="tight")


# fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

objs["z_bin"] = pd.cut(objs["z_cmb"], ZBINS)
false_positive_rate = (
    objs[["SCONE", "TRUTH", "z_bin"]]
    .groupby("z_bin")
    .apply(
        lambda x: precision_score(x.TRUTH, x.SCONE)
        # lambda x: confusion_matrix(x.TRUTH, x.SCONE)[0, 1]
        # / (
        #     confusion_matrix(x.TRUTH, x.SCONE)[0, 1]
        #     + confusion_matrix(x.TRUTH, x.SCONE)[1, 1]
        # )
    )
)
print(false_positive_rate)

fig, ax = plt.subplots(tight_layout=True)
ax.plot(
    false_positive_rate.index.map(lambda x: x.mid).to_numpy(),
    false_positive_rate.values,
    ".-",
)
ax.grid()
# ax.set_yscale("log")
ax.set_xlabel("Redshift")
ax.set_xlim([0, 2.9])
# ax.set_ylabel("False Positive Rate")
ax.set_ylabel("Precision")
ax.set_ylim([0.5, 1])
plt.savefig(
    FIGURE_PREFIX / "SCONEonRoman_confusion_z.pdf", bbox_inches="tight"
)  # cm = conf

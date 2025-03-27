from pathlib import Path
from datetime import datetime

from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich import print
from tqdm import tqdm


###### SETUP ######

OUTPUT_WIDTH = 30


def add_general_metadata(df, data_set_name):
    df.attrs["name"] = "Hourglass Simulation"
    df.attrs["data set"] = data_set_name
    df.attrs["version"] = "1.0"
    df.attrs["date"] = datetime.today().strftime("%Y-%m-%d-%H:%M")
    df.attrs["author"] = "Roman SN Cosmology PIT"
    df.attrs["corresponding author"] = "Ben Rose <ben_rose@baylor.edu>"
    df.attrs["SNANA version"] = "v11_05t-74-g980fad9"


elasticc_map = {
    10: "SN_Ia",
    11: "SNIa-91bg",
    12: "SN_Iax",
    30: "CCSN",
    40: "SLSN-I",
    42: "TDE",
    45: "ILOT",
    50: "KN",
    59: "PISN",
    60: "AGN",
    90: "Fixed_mag",
}

SNANA_FOLDER = "snana_output"
RELEASE_FOLDER = "data_release"

dump_files = list(Path(".").glob(SNANA_FOLDER + "/PIP*/*DUMP.gz"))
object_files = list(Path(".").glob(SNANA_FOLDER + "/PIP*/*HEAD*"))
phot_files = list(Path(".").glob(SNANA_FOLDER + "/PIP*/*PHOT*"))
class_file = Path(SNANA_FOLDER + "/predictions.csv")
spec_files = list(Path(".").glob("snana_output/PIP*/*SPEC.FITS*"))

object_files.sort(key=lambda x: str(x))
phot_files.sort(key=lambda x: str(x))
spec_files.sort(key=lambda x: str(x))


###### OBJECT & PHOTOMETRY ######

print("\n" + "#" * OUTPUT_WIDTH)
print("Reading .DUMP files:".center(OUTPUT_WIDTH, " "))
print("")
dump = pd.DataFrame({})
for dump_file in dump_files:
    print(f"{dump_file}")
    temp_dump = pd.read_csv(
        dump_file,
        sep="\s+",
        comment="#",
        na_values=[-9, -99],
        usecols=[
            "CID",
            "SNTYPE",
            "NON1A_INDEX",
            "ZCMB",
            "PEAKMJD",
            "NOBS",
            "TRESTMIN",
            "TRESTMAX",
            "SNRMAX_Y",
            "SNRMAX_J",
            "PEAKMAG_Y",
            "PEAKMAG_J",
            "RA",
            "DEC",
            "FIELD",
            "MWEBV",
        ],
    )
    print(len(temp_dump), "total objects")
    dump = pd.concat([dump, temp_dump], ignore_index=True)
dump["SNTYPE"] = dump["SNTYPE"].map(elasticc_map)
del temp_dump

print("\n" + "#" * OUTPUT_WIDTH)
print(f"Reading {len(object_files)} *_HEAD.FITS and *_PHOT.FITS files:")
print("")
obj = pd.DataFrame()
phot = pd.DataFrame()
for object_file, phot_file in zip(object_files, phot_files):
    print(f"Opening {object_file}")
    temp_obj = Table.read(object_file, format="fits")
    # Could also get "RA","DEC","NOBS",
    # "MWEBV","MWEBV_ERR","REDSHIFT_FINAL","REDSHIFT_FINAL_ERR",
    temp_obj = temp_obj[
        "SNID", "SNTYPE", "PTROBS_MIN", "PTROBS_MAX", "NOBS", "SIM_TYPE_NAME"
    ].to_pandas()
    temp_obj.rename(columns={"SNID": "CID"}, inplace=True)
    temp_obj = temp_obj.astype({"CID": np.int32})
    temp_obj = temp_obj.astype({"SIM_TYPE_NAME": str})
    temp_obj["SIM_TYPE_NAME"] = temp_obj["SIM_TYPE_NAME"].str.strip()

    print(f"      & {phot_file}")
    temp_phot = Table.read(phot_file, format="fits")
    # ('MJD','BAND','CCDNUM','FIELD','PHOTFLAG','PHOTPROB','FLUXCAL','FLUXCALERR','PSF_SIG1','PSF_SIG2','PSF_RATIO','SKY_SIG','SKY_SIG_T','RDNOISE','ZEROPT','ZEROPT_ERR','GAIN','XPIX','YPIX','SIM_MAGOBS')
    temp_phot = temp_phot[
        "MJD",
        "BAND",
        # "FIELD",
        "PHOTFLAG",
        "FLUXCAL",
        "FLUXCALERR",
        "PSF_NEA",
        "SKY_SIG",
        "RDNOISE",
        "ZEROPT",
        "ZEROPT_ERR",
        "SIM_MAGOBS",
    ].to_pandas()
    temp_phot = temp_phot.astype({"BAND": str})
    temp_phot["BAND"] = temp_phot["BAND"].str.strip()
    # temp_phot = temp_phot.astype({"FIELD": str})
    # temp_phot["FIELD"] = temp_phot["FIELD"].str.strip()

    ## combine head and dump to make final photometry file
    single_lcs = []
    for row in tqdm(
        temp_obj.itertuples(),
        total=temp_obj.shape[0],
        desc="processing light-curves",
        unit="objects",
        dynamic_ncols=True,
        delay=1,
    ):
        single_lc = temp_phot[int(row.PTROBS_MIN) : int(row.PTROBS_MAX)]
        if len(single_lc) < 1:
            print(f'!!!!!!!!!!!!!issue with {row["CID"]}!!!!!!!!!!!!!!!!!!!!!!!')
            breakpoint()
        ## possible: ra, dec, z, peakmjd  or "SNTYPE"
        single_lc.insert(0, "CID", row.CID)
        single_lcs.append(single_lc)
    print("Concatinating object & photomerty data")
    phot = pd.concat([phot, *single_lcs], ignore_index=True)
    obj = pd.concat([obj, temp_obj], ignore_index=True)
del temp_obj
del temp_phot
del single_lc

dropped_objects = False
if not obj["CID"].is_unique:
    dropped_objects = True
    print("\n" + "#" * OUTPUT_WIDTH)
    print("DROPPING DUPLICATE CIDS!".center(OUTPUT_WIDTH, "#"))
    print("")
    original_size = obj.shape[0]
    print(f"Original size {original_size}")
    dump.drop_duplicates(subset=["CID"], inplace=True)
    obj.drop_duplicates(subset=["CID"], inplace=True)
    new_size = obj.shape[0]
    print(f"New size {new_size}")
dump.set_index("CID", inplace=True)
obj.set_index("CID", inplace=True)
obj = dump.join(obj, rsuffix="_HEAD")
obj = obj.dropna(subset=["PTROBS_MIN", "PTROBS_MAX"])
# NaN from join mismatch made these floats. Fix after .dropna().
obj.rename(columns={"SNTYPE": "CLASS"}, inplace=True)
obj = obj.astype(
    {"SNTYPE_HEAD": int, "PTROBS_MIN": int, "PTROBS_MAX": int, "NOBS_HEAD": int}
)

try:
    class_ = pd.read_csv(class_file, comment="#", na_values=[-9, -99], index_col="snid")
    class_.rename(columns={"pred_labels": "SCONE_PROB_IA"}, inplace=True)
    obj = obj.join(class_["SCONE_PROB_IA"], on="CID", rsuffix="_class")
    classification = True
except FileNotFoundError:
    classification = False

# fix band name to be just one letter. Somehow it became "Z087-Z"
phot["BAND"] = phot["BAND"].map(lambda x: x[-1])

print("\n" + "#" * OUTPUT_WIDTH)
with pd.option_context("display.max_columns", None):
    print("Checking final objects".center(OUTPUT_WIDTH, " "))
    print(obj.head())
    print(obj.describe())
    print("\nCheking final photometry file".center(OUTPUT_WIDTH, " "))
    print(phot.head(3))
    print(phot.tail(3))
    print(phot.describe())
print("")

add_general_metadata(obj, "objects")
add_general_metadata(phot, "photometry")


#### Convert to parquet ####

columns = [
    # remove columns that were used to verify join
    "FIELD",
    "CLASS",
    # "NON1A_INDEX",
    "SIM_TYPE_NAME",
    "ZCMB",
    "RA",
    "DEC",
    "MWEBV",
    "PEAKMJD",
    "SNRMAX_Y",
    "SNRMAX_J",
    "PEAKMAG_Y",
    "PEAKMAG_J",
    "NOBS",
    "TRESTMIN",
    "TRESTMAX",
]
if classification:
    columns.append("SCONE_PROB_IA")

new_columns = [
    "field",
    "class",
    "sub_class",
    "z_cmb",
    "ra",
    "dec",
    "mw_ebv",
    "peak_mjd",
    "snr_max_Y",
    "snr_max_J",
    "peak_mag_Y",
    "peak_mag_J",
    "n_obs",
    "t_rest_min",
    "t_rest_max",
]
if classification:
    new_columns.append("scone_prob_Ia")
new_columns.append("cid")

obj_table = pa.Table.from_pandas(obj, preserve_index=True, columns=columns)
# keeping the from_pandas created metadata allows for the usage of the `use_pandas_metadata` key in `parquet.read_table()`
obj_table = obj_table.rename_columns(new_columns)
obj_table = obj_table.replace_schema_metadata(
    {**obj.attrs, **(obj_table.schema.metadata or {})}
)
print("\n" + "#" * OUTPUT_WIDTH)
print("Checking parquet objects".center(OUTPUT_WIDTH, " "))
print(obj_table)
pq.write_table(obj_table, RELEASE_FOLDER + "/hourglass_objects.parquet")

new_columns = [
    "cid",
    "mjd",
    "band",
    "phot_flag",
    "fluxcal",
    "fluxcal_err",
    "psf_nea",
    "sky_sig",
    "read_noise",
    "zp",
    "zp_err",
    "sim_mag_obs",
]
phot_table = pa.Table.from_pandas(phot, preserve_index=False)
phot_table = phot_table.rename_columns(new_columns)
phot_table = phot_table.replace_schema_metadata(
    {**phot.attrs, **(phot_table.schema.metadata or {})}
)
print(phot_table)
pq.write_table(phot_table, RELEASE_FOLDER + "/hourglass_photometry.parquet")


###### SPECTROSCOPY ######


#### Spectra
print("\n" + "#" * OUTPUT_WIDTH)
print(f"Reading {len(spec_files)} *_SPEC.FITS:")
print("")
spec_temps = []
for spec_file in spec_files:
    print(f"Opening {spec_file}")
    with fits.open(spec_file) as hdul:
        # extension 1:'SNID','MJD','Texpose','SNR_COMPUTE','LAMMIN_SNR','LAMMAX_SNR','SCALE_HOST_CONTAM','NBIN_LAM','PTRSPEC_MIN','PTRSPEC_MAX','SIM_SYNMAG_Y','SIM_SYNMAG_J','SIM_SYNMAG_H','SIM_SYNMAG_F','SIM_SYNMAG_Z','SIM_SYNMAG_R'
        data = Table.read(hdul[1])
        spec_temp = data[
            [
                "SNID",
                "MJD",
                "Texpose",
                "NBIN_LAM",
                "PTRSPEC_MIN",
                "PTRSPEC_MAX",
            ]
        ].to_pandas()
        spec_temp.rename(columns={"SNID": "CID"}, inplace=True)
        # spec.set_index("CID", inplace=True)
        # extension 2:'LAMMIN','LAMMAX','FLAM','FLAMERR','SIM_FLAM'
        data = Table.read(hdul[2]).to_pandas()
    spec_dic = {key: spec_temp[key] for key in spec_temp.columns}
    for column in data.columns:
        spec_dic[column] = []

    for index, row in tqdm(
        spec_temp.iterrows(),
        total=spec_temp.shape[0],
        desc="processing spectra",
        unit="objects",
        dynamic_ncols=True,
        delay=1,
    ):
        single_spec = data[row["PTRSPEC_MIN"] : row["PTRSPEC_MAX"]]
        if len(single_spec) < 1:
            print(f'!!!!!!!!!!!!!issue with {row["CID"]}!!!!!!!!!!!!!!!!!!!!!!!')
            # breakpoint()
        for column in single_spec.columns.values:
            spec_dic[column].append(single_spec[column].to_list())

    spec_temp = pd.DataFrame(spec_dic)
    spec_temps.append(spec_temp)

spec = pd.concat([*spec_temps], ignore_index=True)
spec = spec.astype({"CID": np.int32})
print("Spectra data size:", spec.shape)
del spec_dic
del spec_temp
del single_spec

print("\n" + "#" * OUTPUT_WIDTH)
with pd.option_context("display.max_columns", None):
    print("Checking final spec".center(OUTPUT_WIDTH, " "))
    print(spec.head())
    print(spec.tail())
    print(spec.describe())
    print("")


################
add_general_metadata(spec, "spectroscopy")

spec_table = pa.Table.from_pandas(
    spec,
    preserve_index=False,
    columns=[
        "CID",
        "MJD",
        "Texpose",
        # "SNR_COMPUTE", # SNRs seem to all be nan
        # "LAMMIN_SNR",
        # "LAMMAX_SNR",
        "NBIN_LAM",
        "LAMMIN",
        "LAMMAX",
        "FLAM",
        "FLAMERR",
        "SIM_FLAM",
    ],
)
new_columns = [
    "cid",
    "mjd",
    "t_expose",
    "n_bin_lam",
    "lam_min",
    "lam_max",
    "flam",
    "flam_err",
    "sim_flam",
]
spec_table = spec_table.rename_columns(new_columns)
spec_table = spec_table.replace_schema_metadata(
    {**spec.attrs, **(spec_table.schema.metadata or {})}
)
print(spec_table)
pq.write_table(spec_table, RELEASE_FOLDER + "/hourglass_spectra.parquet")


# FINAL OUTPUT
print("\n" + "#" * OUTPUT_WIDTH)
print("Summary".center(OUTPUT_WIDTH, " "))
print("")
print(obj_table.schema.metadata)
print(obj.shape[0], "objects")
print(phot.shape[0], "photometry rows")
print(spec.shape[0], "spectral rows")
if dropped_objects:
    print(f"DROPPED {original_size-new_size} duplicate CIDs")
if not classification:
    print("Missing classification")
print("#" * OUTPUT_WIDTH)

###### CHECK ######


# Check output files (mostly metadata)

# temp = pd.read_parquet("data_release/hourglass_objects.parquet")
# print(temp.attrs)
# dataset = pq.ParquetDataset("data_release/hourglass_objects.parquet").read_pandas()
# print(dataset.schema)
# print(dataset.to_pandas().attrs)
#
# parquet_file = pq.ParquetFile("data_release/hourglass_objects.parquet")
# print(parquet_file.metadata.metadata)
# print(pq.read_metadata("data_release/hourglass_objects.parquet").metadata)

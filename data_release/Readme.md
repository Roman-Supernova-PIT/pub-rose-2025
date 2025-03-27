# The Hourglass Simulation

This simulation is a catalog of transients from the [Nancy Grace Roman][roman] [High-Latitude Time-Domain Core Community Survey][hltds] produced by the [Roman Supernova Cosmology Project Infrastructure Team][pit]. There are 71,716 objects, 12,473,457 photometric values, and 563,230 spectra.
The is the data associated with Rose et al. 2025. Details of the simulation process and associated input files can be found in that paper.

[hltds]:https://science.nasa.gov/mission/roman-space-telescope/high-latitude-time-domain-survey/
[pit]: https://github.com/roman-supernova-pit
[roman]: https://roman.gsfc.nasa.gov


## Files

The simulation is released in three files: `hourglass_objects.parquet`, `hourglass_photometry.parquet` and `hourglass_spectra.parquet`. These files are in the [parquet] format.

[parquet]: https://en.wikipedia.org/wiki/Apache_Parquet

### Reading the file metadata

Parquet files have a host of metadata. Such as file level metadata[1], and row group metadata (read_row_group). This information is mostly used internally but can also be used to speed up [data queries](https://arrow.apache.org/blog/2022/12/26/querying-parquet-with-millisecond-latency/).

[1]: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_metadata.html#pyarrow.parquet.read_metadata

Beyond the standard parquet metadata, we add addition Hourglass specific information. This can be seen via the following lines of python.

```python
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("hourglass_objects.parquet")
print(parquet_file.metadata.metadata)
```

This results in a dictionary that contains the keys "name", "data set", "version", "date", "author", and "corresponding author" along with others from PyArrow.

### Reading file data

Pandas has a simple [`read_parquet()`][read_parquet] method that works well with this data set. An example can be seen below.

[read_parquet]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html

```python
import pandas as pd

hourglass_objects = pd.read_parquet("hourglass_objects.parquet")
hourglass_objects.describe()
```

### Matching objects across files

The simulation is split into three files for data formatting reasons. For the objects file, each row is a transient. For the photometry and spectroscopy, each row corresponds to a photometric value or spectrum. Since each object has multiple photometric observations and/or spectra, these two tables are significantly longer.

The column "CID" (candidate ID) can be used to link information across tables. An example of this in python can

Note that not all objects have spectra.

An example of selecting the photometry of an object with a known S/N value can be seen below.

```python
target_snr_max = 15
object = hourglass_objects[hourglass_objects["snr_max_Y"] == target_snr_max]
#selects first CID/index of the list that matches the target S/N
lightcurve = photometry[photometry["cid"] == object.index[0]]
```

### Examples

Getting the peak magnitudes, in Y-band, for all the Type Ia supernovae.

```python
hourglass_objects[hourglass_objects["class"] == "SN_Ia"]["peak_mag_Y"]
```

Plotting the LC for an object with the CID of 20,000.

```python
lightcurve = photometry[photometry["cid"] == 20_000]
plt.figure()
if "Z" in lightcurve["band"].unique():
    bands = ["R", "Z", "Y", "J"]
if "H" in lightcurve["band"].unique():
    bands = ["Y", "J", "H", "F"]
for band in bands:
    lc = lightcurve[lightcurve["band"] == band]
    lc.loc[:, "mag"] = mag(lc["fluxcal"].values)
    lc.loc[:, "mag_err"] = np.abs(lc["fluxcal_err"].values / lc["fluxcal"].values)
    lc.sort_values("mjd", inplace=True)
    plt.errorbar(
        lc["mjd"].values,
        lc["mag"].values,
        lc["mag_err"].values,
        fmt=".-",
        label=band,
    )
```

##  Detail Column Descriptions

### `hourglass_objects.parquet`

| Column Name | Description Name | Units | Type |
|-------------|------------------|-------|------|
| cid | Candidate ID | ... | integer |
| field | Field name | ... | string|
| class | Object classification name | ... | string|
| sub_class | Name of subclass, such as "IIP" and "Ic" | ... | string|
| z_cmb | Redshift in CMB frame | ... | float|
| ra | Right Ascension | degrees | float |
| dec | Declination | degrees | float|
| mw_ebv | Milky-way E(B-V) along ling-of-sight | AB mag | float|
| peak_mjd | Modified Julian date of maximum | ... | float|
| snr_max_Y | Simulated peak S/N in Y-band | ... | float|
| snr_max_J | Simulated S/N magnitude in J-band | ... | float|
| peak_mag_Y | Simulated peak magnitude in Y-band  | AB mag | float|
| peak_mag_J | Simulated peak magnitude in J-band | AB mag | float|
| n_obs | Number of total observations | ... | integer|
| t_rest_min | Phase of first observation | days | float|
| t_rest_max | Phase of last observation | days | float|
| scone_prob_Ia | Probability of being a SN Ia via SCONE | ... | float|


### `hourglass_photometry.parquet`

| Column Name | Description Name | Units | Type |
|-------------|------------------|-------|------|
cid | Candidate ID | ... | integer |
mjd | Modified Julian date of observation | ... | float|
band | Photometric band, one of R, Z, Y, J, H, or F | ... | string |
phot_flag | Photometric Flag, 0 is pass | ... | integer|
fluxcal | Calibrated flux, mag = 27.5 - 2.5*log_{10}(`fluxcal`) | ... | float|
fluxcal_err | Poisson uncertainty on `fluxcal`,  sky+galaxy+source | ... | float|
psf_nea | PSF noise equivalent area | pixels | float|
sky_sig | Sky noise | ADU/pixel | float|
read_noise | Read noise | ADU/pixel | float|
zp | Zero-point | AB mag | float|
zp_err | Error on zero-point | AB mag | float |
sim_mag_obs | Input model mag | AB mag | float|

  
### `hourglass_spectra.parquet`

| Column Name | Description Name | Units | Type |
|-------------|------------------|-------|------|
cid | Candidate ID | ... | int |
mjd | Modified Julian date of observation | ... | float |
t_expose | Exposure time | seconds | float|
n_bin_lam | Number of wavelength bins | ... | int|
lam_min | Low wavelength edge of spectral bin | angstroms | list of floats |
lam_max | High wavelength edge of spectral bin | angstroms | list of floats|
flam | Flux per wavelength bin | dF/dlambda | list of floats|
flam_err | Error on `flam` | ... | list of floats|
sim_flam | Input or true flux | ...| list of floats|

# Nighttime Activity after the 2021 Henan Floods

This project studies light variation in Zhengzhou, Kaifeng, and Xinxiang from
June to September 2021. It combines noninterpolated HDNTL observations, 2020
population weights, and historical county geometries. A separate HISEA-1
experiment compares intensity thresholds and local-feature logistic regression
for water recognition.

# Data sources

All analyses use downloaded public observations and their supplied labels.

| Input | Source | Retained sample | License |
|---|---|---|---|
| HDNTL v2 | [Zenodo 17079409](https://doi.org/10.5281/zenodo.17079409) | Three urban windows, 2020 and 2021 light and flag files | CC BY 4.0 |
| WorldPop China 2020 | [Product 29818](https://hub.worldpop.org/geodata/summary?id=29818) | Three cropped windows from the 1 km population-count raster | CC BY 4.0, as stated on the product page |
| China county geometries | [geoBoundaries CHN ADM2](https://www.geoboundaries.org/api/current/gbOpen/CHN/ADM2/) | Historical geometries intersecting the three windows | PDDL v1.0, according to the metadata |
| HISEA-1 SAR water dataset | [Zenodo 7198950](https://doi.org/10.5281/zenodo.7198950) | Seeded subsets: 96 train, 32 validation, 64 test image/label pairs | CC BY 4.0, according to the Zenodo API metadata |

HDNTL citation: Pei, Z., Zhu, X., Hu, Y., Chen, J., and Tan, X. (2025),
“A high-quality daily nighttime light (HDNTL) dataset for global 600+ cities
(2012–2024),” Earth System Science Data, 17, 5675–5691,
https://doi.org/10.5194/essd-17-5675-2025.

WorldPop and CIESIN developed the Global High Resolution Population Denominators
Project, https://doi.org/10.5258/SOTON/WP00670. The selected product estimates
people per 30-arc-second grid cell. Bilinear resampling supplies relative
weights for the light index; the analysis does not sum resampled counts.

geoBoundaries metadata identify boundary ID CHN-ADM2-17275852, county level,
reference year 2017. The code retains the historical source names and IDs.
County-window units describe only the intersection with each urban window.
The download follows the metadata's pinned Git commit and retrieves actual
Git LFS content rather than the pointer file.

Radar citation: Lv, S., and Geng, X. (2022), “HISEA-1 SAR Floodwater Mapping
Dataset,” Zenodo, https://doi.org/10.5281/zenodo.7198950. The source describes
20 HISEA-1 scenes from three 2021 flood events, 3 m VV-polarized imagery,
and manual water labels. The distributed PNG patches have no georeferencing.
The project samples the published splits with seed 42 and preserves their
membership. The source's random patch allocation does not ensure independent
scenes across the splits.

The project modifies the datasets by selecting files and patches, cropping
population data, filtering light observations, reprojecting flags and weights,
and computing aggregates. Source archives and country-wide auxiliary rasters
are excluded from the final package after the selected inputs are retained.

`ntl_download_manifest.json` records the actual HTTP 206 checks and SHA-256
hashes of extracted members. The standard ZIP reader checks member CRCs.
`radar_download_manifest.json` records the full archive's published MD5 check.
`context_download_manifest.json` records population and geometry downloads.
`radar_subset_manifest.json` lists every selected patch and its SHA-256 hash.

Contextual sources:

- [NASA, Flooding in Central China, July 28, 2021](https://science.nasa.gov/earth/earth-observatory/flooding-in-central-china-148623/).
- [State Council, post-flood recovery account, August 19, 2021](https://english.www.gov.cn/premier/news/202108/19/content_WS611e5771c6d0df57f98dec09.html).


## Reproducing the data locally

Large downloaded inputs are intentionally not committed to this repository. After cloning, install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
```

The download script retrieves the public inputs, verifies archive checksums and selected-file hashes, and recreates the local `data/raw/` files needed by the analysis.

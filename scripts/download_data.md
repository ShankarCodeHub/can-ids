# Dataset download instructions

Raw data is **gitignored**. Download it manually, put it in the folders below, then run
`python scripts/build_dataset.py --checksums`. That records the SHA256 of every raw file in
`reports/results/raw_checksums.csv`, so anyone can check they have identical data.

---

## 1. HCRL Car-Hacking Dataset

- **Publisher:** Hacking and Countermeasure Research Lab (HCRL), Korea University
- **Official page:** <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset> (follow its download link, currently Dropbox)
- **Terms:** released for academic purposes. Cite the papers below.
- **Content:** traffic logged through the OBD-II port of a real vehicle. The four attack files each hold about 30–40 minutes of traffic with 300 intrusions of 3–5 s. The attacks are:
  - DoS: ID `0x000` injected every 0.3 ms
  - Fuzzy: random ID and data injected every 0.5 ms
  - Gear spoofing and RPM spoofing: a specific ID injected every 1 ms

**Place these files in `data/raw/hcrl/`** (subfolders are fine; the loader searches recursively):

| File | Unified label for injected rows (flag `T`) |
|---|---|
| `DoS_dataset.csv` | DoS |
| `Fuzzy_dataset.csv` | Fuzzy |
| `gear_dataset.csv` | Spoofing (`gear_spoofing`) |
| `RPM_dataset.csv` | Spoofing (`rpm_spoofing`) |
| `normal_run_data.txt` | Normal (attack-free log) |

Rows flagged `R` are always Normal. If your copy has different file names, edit
`hcrl.files` in `configs/data.yaml`. Don't rename the data itself.

**Cite:**
- H. M. Song, J. Woo, H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," *Vehicular Communications*, vol. 21, 100198, 2020.
- E. Seo, H. M. Song, H. K. Kim, "GIDS: GAN based Intrusion Detection System for In-Vehicle Network," *PST 2018*, IEEE. The HCRL page notes Table 2 of the PST version has errors and points to arXiv:1907.07377 for corrected numbers.

---

## 2. ROAD dataset

- **Publisher:** Oak Ridge National Laboratory
- **Download:** Zenodo record <https://doi.org/10.5281/zenodo.10462796>, file `road.zip` (about 557 MB, md5 `cab184cfc2fe12c0834bc46188c0f330`). The same data is also available at DOI 10.13139/ORNLNCCS/1728694.
- **License:** CC BY 4.0
- **Content:** 12 ambient captures (about 3 h) and 33 attack captures (about 30 min) from one vehicle on a dynamometer. The raw logs are candump format, and the frames are not labelled. Each folder's `capture_metadata.json` gives the injection ID, the data pattern and the interval.

**Extract `road.zip` into `data/raw/road/`**, so you get, for example:
```
data/raw/road/ambient/*.log
data/raw/road/ambient/capture_metadata.json
data/raw/road/attacks/*.log
data/raw/road/attacks/capture_metadata.json
data/raw/road/signal_extractions/...   (ignored by the loader)
```
It's fine if the zip adds an extra top-level `road/` folder.

**Label mapping**

| Capture | Label |
|---|---|
| `fuzzing_*` | Fuzzy |
| correlated signal, max speedometer, max engine coolant temp, reverse light (normal and `_masquerade` versions) | Spoofing |
| ambient captures, and frames in attack captures that don't match the injection rule | Normal |
| `accelerator_*` | excluded, because the authors did not disclose its labels |
| DoS | none: ROAD has no DoS flood, so DoS cannot be evaluated on ROAD |

**Cite:** M. E. Verma, R. A. Bridges, M. D. Iannacone, S. C. Hollifield, P. Moriano, S. C. Kay, J. Hespeler, F. L. Combs, "A comprehensive guide to CAN IDS data and introduction of the ROAD dataset," *PLoS ONE* 19(1): e0296879, 2024.

---

## 3. Assumptions to verify on the real files

The loaders were written from the official documentation **before** the files were downloaded.
After the first `build_dataset.py` run, check each item and record the result here:

| # | Assumption | How to check | Status |
|---|---|---|---|
| 1 | HCRL CSVs have no header row | `ingestion_report.csv`: `bad_timestamp` drop count is 0 (a header row would show up as 1) | ☐ (HCRL not downloaded yet) |
| 2 | In HCRL CSVs, the flag sits right after the last data byte (rows with DLC < 8 are shorter) | drops `too_few_fields` / `bad_flag` should be about 0 | ☐ |
| 3 | `normal_run_data.txt` lines look like `Timestamp: … ID: … 000 DLC: 8 …` | drops `bad_format` should be about 0 | ☐ |
| 4 | ROAD `capture_metadata.json` keys are `injection_id`, `injection_data_str`, `injection_interval` | open the file; fix `road.metadata_keys` if they differ | ✅ 2026-11-11. The JSON uses exactly these keys. Note: the bundled `readme.md` calls them `injected_aid` / `injection_data`; the JSON file is what the loader reads. |
| 5 | ROAD intervals are seconds elapsed from the first frame | per-file attack counts are > 0 and fall inside the stated windows | ✅ Logs start at `1110000000.000000`. 0 attack frames fall outside their window (`road_label_check.csv`). |
| 6 | Every ROAD attack capture (except accelerator) yields > 0 attack frames | no "0 frames matched" warnings in `ingestion_report.csv` | ✅ All 29 labelled attack captures have attack frames; 0 malformed rows in all 41 parsed captures. |
| 7 | The ROAD labelling rule separates injected from genuine target-ID frames | `scripts/verify_road_labels.py`: rows removed in each `_masquerade` file = unmatched target-ID frames in the original window | ✅ Equal for all 13 masquerade pairs. |

**ROAD verification record (2026-11-11):** `road.zip` md5 = `cab184cfc2fe12c0834bc46188c0f330` (matches Zenodo).
SHA256 of every extracted file is in `reports/results/raw_checksums.csv`.
The reverse-light payload patterns (`0x0D0`, byte 2 = `04`/`0C`) also occur in normal
traffic: 94.3% / 5.7% of ambient `0x0D0` frames (`road_rule_collisions.csv`). Labels stay
correct because the rule also requires the injection window, but it means payload values
alone are not a reliable attack signature for these attacks.

## 4. Recording checksums manually (optional)

```powershell
Get-FileHash data\raw\hcrl\* -Algorithm SHA256
```
```bash
sha256sum data/raw/hcrl/*
```

# Demo CAN logs (ROAD sample)

Small excerpts of the ROAD dataset in the project's unified frame schema, used by the
dashboard demo, Docker image and edge runner. Labels are included only to display the
replayed ground truth; they are not used for any evaluation.

| File | Frames | Content |
|---|---|---|
| `road_fuzzing_attack_3.parquet` | 13,238 | {'Normal': 13124, 'Fuzzy': 114} |
| `road_max_speedometer_attack_3.parquet` | 213,551 | {'Normal': 207444, 'Spoofing': 6107} |
| `road_reverse_light_on_attack_3_masquerade.parquet` | 153,701 | {'Normal': 151351, 'Spoofing': 2350} |
| `road_ambient_basic_short_60k.parquet` | 60,000 | {'Normal': 60000} |

**Source and licence.** ROAD: Real ORNL Automotive Dynamometer CAN Intrusion Dataset,
Oak Ridge National Laboratory, https://doi.org/10.5281/zenodo.10462796, licensed under
CC BY 4.0. Cite: M. E. Verma et al., "A comprehensive guide to CAN IDS data and
introduction of the ROAD dataset," PLoS ONE 19(1): e0296879, 2024.
Files were converted to the unified schema (scripts/build_dataset.py) and truncated.

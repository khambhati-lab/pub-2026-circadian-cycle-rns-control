# Cycle-based neurostimulation for control of circadian brain network dynamics in human epilepsy

This repository contains the analysis code accompanying the paper:

> Krishnakant V. Saboo, Joline M. Fan, Ehsan Tadayon, Edward F. Chang\*, Vikram R. Rao\*, Ankit N. Khambhati\*, "Cycle-based neurostimulation for control of circadian brain network dynamics in human epilepsy." *(citation to be updated upon publication)*

## Overview

This repository studies the relationship between circadian cycles and electrical stimulation in patients with epilepsy treated with the RNS System (NeuroPace), a responsive neurostimulation device. We analyze how stimulation interacts with circadian rhythms in hippocampal network activity, and relate this interaction to seizure outcomes. Kuramoto oscillator models are fit to this interaction to describe the coupling between circadian and stimulation-driven dynamics, with the goal of enabling model-based control of hippocampal network circadian dynamics via neurostimulation.

## Data availability

Patient ECoG recordings, extracted features, and clinical outcome data are **not included** in this repository, since they derive from protected health information. Only the following are shared, to document the methods and allow others to understand, build on, or verify the analysis and models:
- `src/`: library code for data processing, modeling, and visualization.
- `notebooks/process/`: notebooks that process raw extracted features into analysis-ready files.
- `notebooks/figures/`: notebooks that generate the paper's figures from processed data.

All notebooks and source files in this repository have been reviewed and scrubbed of patient-identifying information. Patient codes (e.g., `NPE####`) are retained as they are de-identified study IDs.

## Reproducibility note

Because the underlying patient data is not shared, the notebooks in this repository **cannot be re-run standalone**:
- `notebooks/process/` notebooks require raw extracted ECoG feature files and outcome spreadsheets (not included) as input.
- `notebooks/figures/` notebooks require the processed data files produced by the `process/` notebooks (also not included).

The code is shared for transparency and reuse (e.g., applying the same feature-extraction, circadian-cycle, or Kuramoto-model pipeline to a different dataset).

## Repository structure

```
notebooks/
  process/   - notebooks that clean and post-process extracted ECoG features
  figures/   - notebooks that generate each paper figure from processed data
src/
  conf/      - YAML configuration for feature extraction (per feature type) and data paths
  dataproc/  - data loading, feature extraction, and pre-processing functions
  model/     - regression, ML, and Kuramoto oscillator models
  runexps/   - scripts to run data processing / model fitting in parallel across patients
  visualize/ - plotting functions for cycles, Lomb-Scargle results, and Kuramoto trajectories
```

### `src/`
- `conf/features/{plv,pib,stimcounter}.yaml`: feature-extraction parameters for each ECoG feature type (phase-locking value, power-in-band, stimulation count).
- `conf/paths/local.yaml`: local data/path configuration.
- `dataproc/neuropaceIO.py`: reads ECoG data from NeuroPace catalog and binary ECoG files.
- `dataproc/rnsfeatureext.py`: extracts ECoG features (PLV, PIB, spectral synchrony) from RNS recordings.
- `dataproc/stimartifact.py`: detects and removes stimulation artifacts from ECoG signals.
- `dataproc/edgecases.py`: patient-specific handling of data irregularities in the recording catalog.
- `dataproc/spectralanalysis.py`: Lomb-Scargle spectral analysis of ECoG features to detect circadian cycles.
- `dataproc/utils.py`: general utility functions supporting data pre-processing and cycle analysis.
- `dataproc/ccutils.py`: shared constants and utilities used across the modeling code.
- `model/kuramotoattractor.py`: Kuramoto oscillator attractor model relating circadian and stimulation dynamics.
- `model/linearreg.py`: linear regression and cross-validation utilities.
- `model/hierarchmllinreg.py`: hierarchical/mixed-effects linear regression models.
- `model/mlmodel.py`: machine learning models (ensemble, SVM) for predicting outcomes from features.
- `runexps/featureext.py`: parallelized ECoG feature extraction across patients.
- `runexps/stimcounter.py`: parallelized computation of stimulation counts/artifacts across patients.
- `runexps/kuramotodiffcouplings.py`: parallelized fitting of Kuramoto coupling constants across patients.
- `runexps/altstimulationmulti.py`: parallelized in-silico stimulation simulations using the fitted Kuramoto model.
- `visualize/cycleviz.py`: plotting functions for circadian cycles, Lomb-Scargle results, and Kuramoto trajectories.

### `notebooks/process/`
| Notebook | Description |
|---|---|
| `Compute_Circadian_Cycles_PLV.ipynb` | Lomb-Scargle circadian cycle computation from PLV ECoG features. |
| `Compute_Circadian_Cycles_PIB.ipynb` | Lomb-Scargle circadian cycle computation from PIB ECoG features (supplementary biomarker). |
| `RNS_Postprocess_PLV.ipynb` | Cleans raw PLV feature files (removes incomplete/duplicated timestamps, handles lead-revision corrections). |
| `RNS_Postprocess_PIB.ipynb` | Cleans raw PIB feature files. |

### `notebooks/figures/` — figure-to-notebook mapping

| Figure | Notebook | Description |
|---|---|---|
| Figure 1 | `Figure_1_Overview.ipynb` | Study schematic; circadian cycles from data; FC-CC and stimulation cycles over time. |
| Figure 2 | `Figure_2_CircadianBiomarker.ipynb` | Longitudinal seizure outcomes; FC-CC/stimulation circadian biomarker vs. outcome; recursive feature selection prediction model. |
| Figure 3 | `Figure_3_FCCC_Clusters_and_Mechanism.ipynb` | Distribution of peak hour of FC-CC and stimulation; patient clustering and mechanism analysis. |
| Figure 4 | `Figure_4_Stimulation.ipynb` | Group-level stimulation trends; area-under-curve schematic; total stimulation (SPA) vs. outcome. |
| Figure 5 | `Figure_5_Mechanism.ipynb` | Predicted change in circadian peak as a mechanism of stimulation's effect on the circadian cycle. |
| Figure 6 | `Figure_6_Kuramoto.ipynb` | Kuramoto model fitted coupling constants; predicted circadian trajectories; model comparisons. |
| Figure 7 | `Figure_7_Insilico.ipynb` | In-silico Kuramoto simulations exploring hypothetical stimulation control strategies. |
| Supplementary | `RNS_DataSummary.ipynb` | Summary of cohort/data characteristics (supporting table). |
| Supplementary | `Supp_Figure_CircadianBiomarkers.ipynb` | Circadian cycle analysis using unfiltered PLV features; individual patient example |
| Supplementary | `Supp_Figure_ECoGTimes.ipynb` | ECoG recording time counts and per-patient heatmaps. |
| Supplementary | `Supp_Figure_LombScargle.ipynb` | Significance of Lomb-Scargle models; number of significant cycles; power distribution fit (PLV). |
| Supplementary | `Supp_Figure_PIB_LS_Biomarker.ipynb` | Significance of Lomb-Scargle models for the PIB biomarker. |
| Supplementary | `Supp_Figure_SigLS_CircadianBiomarker.ipynb` | Circadian concordance (alpha) biomarker and prediction models for significant LS fits. |
| Supplementary | `Supp_Figure_StimHourNUSMap.ipynb` | Stimulation-hour/SPA maps; SPA maps. |
| Supplementary | `Supp_Figure_Stimulation.ipynb` | Stimulation count trends by outcome group; total stimulation scatter plot. |

## Environment

Developed and run with Python 3.11.5. Key third-party packages and the versions used:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.3.2 |
| scipy | 1.16.0 |
| astropy | 7.1.0 |
| matplotlib | 3.10.5 |
| seaborn | 0.13.2 |
| scikit-learn | 1.7.1 |
| statsmodels | 0.15.0 (development build) |
| pycircstat2 | 0.1.12 |
| statannotations | 0.7.2 |
| omegaconf | 2.3.0 |

Note: `statsmodels` was installed from a development build (`0.15.0.dev706+gd0608bb1a`); the nearest stable release should work for most code paths, but has not been separately verified.

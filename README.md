<!-- # Monitoring Internal Displacement During the Syrian Civil War via Satellite-Based Vehicle Detection -->
# Toward Satellite-Based Monitoring of Internal Displacement: A Case Study of the Syrian Civil War
This is the official repository for the study *"Toward Satellite-Based Monitoring of Internal Displacement: A Case Study of the Syrian Civil War"*

#### Authors
[Noora Al-Emadi](https://orcid.org/0000-0003-4137-6082), Qatar Computing Research Institute, Hamad Bin Khalifa University

[Ingmar Weber](https://orcid.org/0000-0003-4169-2579), Saarland Informatics Campus, Saarland University

[Yin Yang](https://orcid.org/0000-0002-0549-3882), College of Science and Engineering, Hamad Bin Khalifa University

[Ferda Ofli](https://orcid.org/0000-0003-3918-3230), Qatar Computing Research Institute, Hamad Bin Khalifa University


## Abstract
<!-- Reliable data on human mobility is critical for understanding population displacement during armed conflict, yet such information is often scarce, delayed, or inaccessible in crisis-affected regions. In this study, we present a case study of the Syrian Civil War, where we use satellite-derived vehicle detections as a proxy for monitoring internal displacement patterns. We develop an approach that integrates a region-adapted vehicle detection model with a partial coverage correction method, enabling robust estimation of car counts from incomplete satellite observations. Using IDP reports from official sources, we evaluate the relationship between vehicle dynamics and displacement patterns across Syrian cities. Our validation results show that when the official reports and satellite images overlap in terms of time period and geography covered, then there is a good directional agreement between changes in car counts and reported internal displacement. However, we also observe that these two sources are often complementary in terms of exact geography and time periods covered, indicating that our remote sensing approach could help fill data gaps. These findings demonstrate that satellite-based vehicle detection can provide a valuable and scalable complement to traditional displacement monitoring methods in data-scarce conflict settings, as illustrated through the Syrian case. -->
Reliable data on human mobility is critical for understanding population displacement during armed conflict, yet such information is often scarce, delayed, or inaccessible in crisis-affected regions. In this study, we present a case study of the Syrian Civil War, where we use satellite-derived vehicle detections as a potential complementary source of information on internal displacement patterns. We develop an approach that integrates a region-adapted vehicle detection model with a partial coverage correction method to estimate car counts from incomplete satellite observations. Using IDP reports from official sources, we evaluate the relationship between vehicle dynamics and displacement patterns across Syrian cities. We find a positive but moderate association between changes in car counts and reported displacement, driven primarily by agreement in the direction rather than the magnitude of change; this directional agreement strengthens substantially under conditions of high temporal satellite revisit, but at the aggregate level is not clearly distinguishable from a naive baseline. We also observe that satellite imagery and IDP reports are often complementary in terms of exact geography and time periods covered, suggesting that our remote sensing approach could help fill data gaps where official reporting is sparse. These findings suggest that satellite-based vehicle detection can offer complementary, qualitative information about displacement direction in data-scarce conflict settings under favorable observation conditions, though further work is needed to establish its reliability for estimating displacement magnitude or as a general-purpose monitoring tool.

## Dataset

Syria-IDP-Car-Counts dataset is vailable for download on [**Zenodo repository**](https://doi.org/10.5281/zenodo.20643798)


<!-- # Reproducibility Materials -->
<!-- This repository contains the statistical analysis code supporting *"Toward Satellite-Based Monitoring of Internal Displacement: A Case Study of the Syrian Civil War."* The processed dataset these scripts run on is available on [**Zenodo**](https://doi.org/10.5281/zenodo.20643798). -->

## Scope

This repository covers the trend-generation pipeline and all statistical robustness analyses reported in the manuscript. It does **not** include the vehicle-detection model, report-filtering pipeline, or manual temporal-window-adjustment logic; those are internal processing tools and are not released as citable reference code. The manuscript documents these procedures in full methodological detail (see the "Humanitarian Reports" and "Temporal Buffering of Observation Windows" subsections), including exact exclusion criteria and counts, sufficient to understand and independently reimplement them.

## Repository structure

```
├── README.md
├── trend_analysis_pipeline/
   └── IDP_CarCount_trend_pipeline.py
```
<!-- └── correlation_robustness/
    ├── clustering_robustness_analysis.py
    ├── city_size_confound_analysis.py
    ├── detector_uncertainty_analysis.py
    └── significance_diagnostics_analysis.py
-->

## `trend_analysis_pipeline/`

| Script | Description |
|---|---|
| `IDP_CarCount_trend_pipeline.py` | Takes buffered per-report car-count time series and IDP report data as input; computes the observed (C_LF), regression-based (C_LR), and max-min (C_MM) trend measures for each report; computes IDP normalization against baseline population; and generates `analyzers_summary.tsv`, the primary derived dataset consumed by every other script in this repository. Also produces per-report visualizations and the aggregate IDP-vs-car-count correlation figures/tables. |

**Usage**: `python IDP_CarCount_trend_pipeline.py --inp <idp_reports.tsv> --outdir <output_directory>`

<!-- ## `correlation_robustness/`

Scripts supporting the robustness analyses added to the Results section, addressing the points where reviewer follow-up identified the most significant risk to the paper's central claims.

| Script | Reproduces |
|---|---|
| `clustering_robustness_analysis.py` | Governorate-clustered bootstrap CI, leave-one-governorate-out sensitivity, and mixed-effects model with random governorate intercept — tests whether the primary correlation survives the non-independence of observations nested within only 12 governorates. |
| `city_size_confound_analysis.py` | Partial correlations controlling for baseline population and baseline car population, and the corresponding multiple regression — rules out city size as the primary driver of the observed association. |
| `detector_uncertainty_analysis.py` | Analytical and empirical demonstration that a uniform detector-uncertainty correction leaves Pearson's r and directional concordance unchanged; precision/recall attenuation-bias discussion distinguishing the two error sources' effects. |
| `significance_diagnostics_analysis.py` | Spearman's rank correlation as a robustness check, and Cook's distance diagnostics identifying influential observations — surfaced the sensitivity of the primary correlation to a small number of high-magnitude reports. |
-->

## Requirements

```
pandas
numpy
scipy
matplotlib
statsmodels
geopandas   # required only by IDP_CarCount_trend_pipeline.py
```

Install with:
```
pip install pandas numpy scipy matplotlib statsmodels geopandas
```

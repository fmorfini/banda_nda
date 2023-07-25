# BANDA NDA data scoring repository
This repository contains scripts to combine and score behavioral data from the `BANDAImgManifestBeh` NDA package for the [BANDA project](https://nda.nih.gov/edit_collection.html?id=3037) (Boston Adolescent Neuroimaging of Depression and Anxiety), sometimes referred to as Human Connectome Project Related to Disease (HCP) or HCP for Anxiety and Depression in Adolescence.

## Scripts info
- inputs: `BANDAImgManifestBeh/*txt` files include demographic, behavioral, and cognitive data. Most of these data are item-level/raw, but some already aggregated (e.g., PEN and NIHToolbox). These data need to be downloaded directly from NDA
- `combine_score_banda_nda_behave_data.ipynb` is a wrapper script
  - loads all .txt files
  - combines them (by id, respondent, timepoint etc)
  - cleans dataframe (e.g. from missing values and empty columns)      
  - calls in `myfx_scoring.py` to generate subscales and total scores of included questionnaires
  - (optional) removes item-level info, calculates number of NANs, calculates number of theoretical items (for each subject for each subscale) 
- `myfx_scoring.py` list of functions used to score questionnaires
  - grabs item-level columns
  - generates subscales
  - removes item-level cols
- output: one dataframe with scored data from all files (or custom set) included in the `BANDAImgManifestBeh` datapackage

## Questionnaires supported 
- bisbas
- cbcl (scoring info is proprietary so not including here)
- chapman handedness
- cssrs
- ksads01 (technically not a scoring, here just creating grouping based on specific aims of our study)
- ksadsp201 (technically not a scoring, currently dropping this info)
- masq
- mfq
- nffi
- rbqa
- rcads
- rmbi
- shaps
- stai
- strain (scoring info is proprietary so not including here)
- tanner
- wasi
- PENN and NIH tasks (not scoring, just relabelling columns)

## More dataset info
[BANDA project](https://nda.nih.gov/edit_collection.html?id=3037)\
[BANDA NDA download instructions]( https://www.humanconnectome.org/study/connectomes-related-anxiety-depression/data-releases
)\
[BANDA NDA release manual](https://www.humanconnectome.org/storage/app/media/documentation/BANDA1.0/BANDA_Release_1.0_Manual.pdf)\
[BANDA crosswalk file](https://www.humanconnectome.org/storage/app/media/documentation/BANDA1.0/BANDA1.0_Crosswalk.csv)

## Papers 
### Protocols and Data Release Papers
Hubbard et al. (2020) NeuroImage Clinical: https://doi.org/10.1016/j.nicl.2020.102240 \
Siless et al. (2020) NeuroImage Clinical: https://doi.org/10.1016/j.nicl.2020.102242 

### Others using BANDA data
Hubbard et al., (2023) Clinical Psychological Science :https://doi.org/10.1177/21677026221079628 \
Auerbach et al., (2022) JAACAP: https://doi.org/10.1016/j.jaac.2021.04.014 \
Lee et al., (2021) The Cerebellum: https://doi.org/10.1007/s12311-020-01213-8 \
Tozzi et al., (2021) NeuroImage: https://doi.org/10.1016/j.neuroimage.2021.118694

# Infinite hidden Markov models can dissect the complexities of learning

Repository to the paper https://www.nature.com/articles/s41593-025-02130-x

## Abstract

Learning the contingencies of a task is difficult. Individuals learn in an idiosyncratic manner, revising their approach multiple times as they explore and adapt. Quantitative characterization of these learning curves requires a model that can capture both new behaviors and slow changes in existing ones. Here we suggest a dynamic infinite hidden semi-Markov model, whose latent states are associated with specific components of behavior. This model can describe new behaviors by introducing new states and capture more modest adaptations through dynamics in existing states. We tested the model by fitting it to behavioral data of >100 mice learning a contrast-detection task. Although animals showed large interindividual differences while learning this task, most mice progressed through three stages of task understanding, new behavior often arose at session onset, and early response biases did not predict later ones. We thus provide a new tool for comprehensively capturing behavior during learning.
## Installation

This code relies on the installation of two packages, which are custom extensions of existing packages, as described there (these will be installed with the rest of the needed packages in the next step): \
https://github.com/SebastianBruijns/sab_pybasicbayes \
https://github.com/SebastianBruijns/sab_pyhsmm

### Installing with `conda`/`mamba`
```sh
mamba env create -f environment.yml
mamba activate hdp_env
pip install -r requirements_specific.txt
```

### Installing with `pyenv` and `pip`
```sh
pyenv local 3.7
python -m venv .env_hdp
source .env_hdp/bin/activate # on linux/mac
.env_hdp/Scripts/activate # on windows
pip install -r requirements_general.txt
pip install -r requirements_specific.txt
```

(Tested on Ubuntu and Windows)
*Note that to download all the animal data used for the paper, you will also need to create a separate IBL environment*
The data for this analysis is downloaded with the script (note that you will need to get the correct password from https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html): \
``behavioral_state_data_easier.py``
\
For us, this is then processed with ``process_data.py``, to bring the data into the format expected by the fitting procedure.

## Running

``dynamic_GLMiHMM_fit.py`` fits the diHMM model to the specified subjects (depending on the data size this can take a substantial amount of time, for our example mouse it takes about 24 hours, running on 16 cores in parallel) -> produces `[subject_name]_fittype_prebias_var_0_04_[seed]_[id]` files and a fit_info dictionary containing the fit settings.
Supply this script with the number of the chain as an argument, i.e. for a typical run these are the numbers 0-15

The following 3 scripts process the MCMC-chains. They are split, because we usually run 1 and 3 on a cluster, as they are computationally somewhat intense, but 2, where one selects the samples to analyse, is run locally to view and interact with the results (the first and third script still run within around 1 hour on a desktop machine, script two runs in less than 5 minutes).\
``raw_fit_processing_part1.py``  specify argument (0 for a single mouse) \
``raw_fit_processing_part2.py``  \
``raw_fit_processing_part3.py`` specify argument (0 for a single mouse)

These will produce, in sequence, the following intermediate files:
- ``canonical_[infos/result]_[subject]_prebias_var_0_04`` files
- updates ``canonical_result_[subject]_prebias_var_0_04`` file and produces ``mode_indices_[subject]_prebias_var_0_04`` files
- ``state_sets_[subject]_prebias_var_0_04.p`` and ``mode_consistencies_[subject]_prebias_var_0_04.p`` files


``dyn_glm_chain_analysis.py`` goes through all the processed results, plotting overviews (figure 2 and 3), and collecting summaries of the data to process further. Also plots figure 5 and 7

``analysis_pmf.py`` plots figure 4\
``analysis_pmf_weights.py`` plots figure 6, 12, A16, and A17\
``analysis_regression.py`` plots figure 8

## Example data set

We directly provide the data of example mouse KS014, as well as intermediate results for ease of processing, checkout the branch "paper_example". Some files are larger, and are stored here, drop them into the `multi_chain_saves` folder: https://nextcloud.tuebingen.mpg.de/index.php/s/eZD9WskBfme9gDj
The chains (output of ``dynamic_GLMiHMM_fit.py``) are not provided at the moment, as these files are too large, as a consequence ``raw_fit_processing_part1.py`` cannot be run directly.

# Dissecting the Complexities of Learning With Infinite Hidden Markov Models

## Abstract

Learning to exploit the contingencies of a complex experiment is not an easy task for animals. Individuals learn in an idiosyncratic manner, revising their approaches multiple times as they are shaped, or shape themselves, and potentially end up with different strategies. Their long-run learning curves are therefore a tantalizing target for the sort of individualized quantitative characterizations that sophisticated modelling can provide. However, any such model requires a flexible and extensible structure which can capture radically new behaviours as well as slow changes in existing ones. To this end, we suggest a dynamic input-output infinite hidden semi-Markov model, whose latent states are associated with specific components of behaviour. This model includes an infinite number of potential states and so has the capacity to describe substantially new behaviours by unearthing extra states; while dynamics in the model allow it to capture more modest adaptations to existing behaviours. We individually fit the model to data collected from more than 100 mice as they learned a contrast detection task over tens of sessions and around fifteen thousand trials each. Despite large individual differences, we found that most animals progressed through three major stages of learning, the transitions between which were marked by distinct additions to task understanding. We furthermore showed that marked changes in behaviour are much more likely to occur at the very beginning of sessions, i.e. after a period of rest, and that response biases in earlier stages are not predictive of biases later on in this task.

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

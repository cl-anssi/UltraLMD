# Inductive Lateral Movement Detection in Enterprise Computer Networks

This repository contains the code associated with our ESANN submission
"Inductive lateral movement detection in enterprise computer networks".
It is based on a fork from the
[ULTRA](https://github.com/DeepGraphLearning/ULTRA) project, which we
use for lateral movement detection in our work.
We added scripts and configuration files enabling reproduction of the
experiments presented in the paper, including data preprocessing,
baseline models and implementation of our ULTRA-based anomaly detection
method.

### Usage

The `experiments.sh` script reproduces all the steps in our experiments
using ULTRA, from data preprocessing to model training and finetuning
and lateral movement detection.
It relies on the dependencies listed in `requirements.txt`.

In addition, the `baseline_experiments.sh` script reproduces our
experiments using the baseline models HPF and PTF.
Note that it relies on
[PyCP-APR](https://github.com/lanl/pyCP_APR), whose requirements are
incompatible with ULTRA's.
Therefore, you should use different virtual environments to run the two
scripts.

The raw datasets can be obtained by following these links:
[LANL](https://csr.lanl.gov/data/cyber1/),
[OpTC](https://github.com/FiveDirections/OpTC-data).
Their locations should be modified accordingly in the first two lines of
`experiments.sh`.
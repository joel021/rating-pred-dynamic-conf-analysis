# setup for development

# Install dependencies</h1>
    
    * run ```$pip install -r requirements.txt```

    * Install pytorch
    * pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Run setup.py for dev environment:

    ```$python setup.py develop ```

# Run the tests:

    ```$pytest tests``` run all tests

    ```$pytest tests/.../file.py``` run especific python file


# Configure poetry to create virtual environment locally
poetry config virtualenvs.in-project true


# Project structure

- recsysconfident: the main package
  - data_handling: responsible for any data preprocessing, dataset buildings, and dataloaders
  - ml: everything related to machine learning
    - fitting: responsible for perform the fit and evaluation of the models
    - models: Where the models implementations are. Each model implementation should have its on class on its on file. Additionally, the method of instantiating the model and its compatible dataloader should be implemented along with the model class.
  - utils: utilities scripts

- files:
  - data/{database_name}/info.json: Describes the database parameters, such as its columns, which one are used and how many columns are in the dataset
  - setups.json: Describes the supported setups of experiments.

- supported datasets and models:
  - recsysconfident/environment.Environment.database_name_fn: Defines which databases are supported to perform experiments. Add or remove instances from this dictionary to control the supported datasets.
  - recsysconfident/environment.Environment.model_name_fn: Defines which models are supported to perform experiments.

- setups: Are the set of configurations that describes an experiment.
- when running main.py with --setup_instance: means that you probably want to rexecute an experiment, also provide --fit_mode to specify whether you want to fit the model or just rerun the evaluation.


** Setups Examples**

- python main.py --setups ./setups-conf-benchmark.json --setup_name k_folds --k_folds 5 
- 



# Simulation Plan

## Overview

This project investigates the effect of distribution-aware regularization in recommender systems under temporal evaluation settings.

The main objective is to analyze whether recommendation models preserve the empirical distribution of observed ratings over time and how distribution regularization affects:

* predictive performance,
* distributional alignment,
* temporal stability,
* and confidence calibration.

The experiments use incremental time-aware cross validation and compare standard recommendation objectives against Wasserstein-regularized objectives.

---

# Experimental Objectives

The simulations aim to answer the following questions:

1. Do recommendation models reproduce the empirical rating distribution over time?

2. Does Wasserstein-based distribution regularization improve alignment between predicted and observed distributions?

3. What is the trade-off between predictive accuracy and distributional fidelity?

4. How do different recommendation architectures respond to distribution-aware optimization?

5. Does distribution regularization improve temporal stability under distribution drift?

6. What is the effect of distribution regularization on confidence calibration?

---

# Experimental Design

## Main Optimization Objective

The baseline optimization objective is:

$$
\mathcal{L}_{rec} = \text{MSE}(\hat{r}, r)
$$

where:

* ( \hat{r} ) = predicted ratings
* ( r ) = observed ratings

The proposed Wasserstein-regularized objective is:
$$
\text{MSE}(\hat{r}, r)
+
\lambda \cdot
W(\text{SoftHist}(\hat{r}), \text{SoftHist}(r))
$$

where:

* ( W(\cdot, \cdot) ) is the Wasserstein distance
* `SoftHist(.)` is the differentiable soft histogram operator
* ( \lambda ) controls the regularization intensity

The Wasserstein term aligns the marginal distribution of predicted ratings with the empirical distribution of observed ratings.

---

# Time-Aware Evaluation Protocol

All simulations use incremental time-series cross validation.

## Fold Construction

The dataset is:

1. globally sorted by timestamp,
2. divided into sequential temporal folds,
3. shuffled only within each fold.

This guarantees:

* fold ( t+1 ) contains only newer interactions than fold ( t ),
* no future interactions are observed during training.

## Incremental Training Procedure

For fold ( k ):

* Training set:
  $$
  \bigcup_{i=0}^{k-1} F_i
  $$

* Test set:
  $$
  F_k
  $$

The training history therefore grows incrementally over time.

This protocol simulates real-world temporal recommendation scenarios.

---

# Simulation 1 — Matrix Factorization Regularization Sweep

## Objective

Evaluate the effect of Wasserstein regularization intensity on Matrix Factorization models.

## Dataset

* MovieLens 1M

## Model

* Matrix Factorization (MF)

## Lambda Values

The following regularization intensities are evaluated:

```python
lambda_values = [
    0.0001,
    0.001,
    0.01,
    0.05,
    0.5,
    0.6,
    1.0
]
```

A baseline without Wasserstein regularization is also evaluated.

## Optimization Objective

$$
\text{MSE} + \lambda \cdot W
$$

## Metrics

The following metrics are computed for each temporal fold:

### Predictive Metrics

* RMSE
* MAE
* Ranking metrics (if enabled)

### Distributional Metrics

* KL divergence (histogram-based)

### Calibration Metrics

* Expected Calibration Error (ECE)

---

# Simulation 2 — Prior Distribution Models

## Objective

Evaluate whether prior-aware recommendation models respond differently to Wasserstein regularization.

## Datasets

* MovieLens 1M
* Amazon Movies & TVs
* Netflix Prize

## Models

The following models are evaluated:

* CPMF
* LBD
* ORDREC
* PRLIGHTGCN
* Dropout (MC dropout)

## Configurations

Each model is evaluated under two configurations:

### Baseline

No Wasserstein regularization.

$$
\mathcal{L} = \mathcal{L}_{model}
$$

### Wasserstein-Regularized

Wasserstein regularization enabled with:

$$
\lambda = 1
$$

Resulting objective:

$$

\mathcal{L}_{model} + W(\text{SoftHist}(\hat{r}), \text{SoftHist}(r))
$$

---

# Distribution Estimation

Distribution comparison metrics are computed using histogram-based distribution estimation.

## Histogram Construction

For each evaluation step:

1. predicted ratings are discretized into shared bins,
2. observed ratings are discretized using the same bin edges,
3. histograms are normalized into probability distributions.

This allows valid divergence computation between prediction and observation distributions.

---

# Temporal Analysis

All metrics are tracked across temporal folds.

This enables analysis of:

* temporal drift,
* distributional instability,
* calibration evolution,
* and robustness under changing user preferences.

---

# Expected Outcomes

The experiments investigate whether Wasserstein regularization:

* improves distributional fidelity,
* reduces temporal divergence,
* stabilizes prediction distributions,
* improves or harms calibration,
* and affects predictive accuracy.

The simulations also analyze whether different recommendation architectures exhibit different sensitivities to distribution-aware optimization.

---

# Reproducibility Notes

To ensure reproducibility:

* random seeds are fixed,
* fold generation is deterministic,
* histogram binning is shared between prediction and observation distributions,
* temporal ordering is strictly preserved.

---

# Important Implementation Constraints

The implementation must guarantee:

1. No temporal leakage between folds.

2. Histogram binning consistency between prediction and target distributions.

3. Wasserstein loss computed from differentiable soft histograms.

4. Temporal folds evaluated sequentially.

5. Distribution metrics computed on identical supports.

6. Fold-wise metrics stored independently for temporal analysis.

---

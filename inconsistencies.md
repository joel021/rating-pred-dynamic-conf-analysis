# Project Inconsistencies Report

This document details the inconsistencies, bugs, and mismatched behaviors identified in the main workflow and test suite of the project. All items have been resolved and verified.

---

## 1. CLI Arguments & Entrypoint (`main.py` vs `README.md`)

### [x] Argument Parser Discrepancy
* **Documentation (`README.md`):** Under the setups section, it states:
  > *"...when running main.py with `--setup_instance`: means that you probably want to rexecute an experiment, also provide `--fit_mode` to specify whether you want to fit the model or just rerun the evaluation."*
  
  And gives the command example:
  ```bash
  python main.py --setups ./setups-conf-benchmark.json --setup_name k_folds --k_folds 5
  ```
* **Implementation ([main.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/main.py)):** The command-line argument parser in [main.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/main.py#L50-L57) only registers one argument: `--setups`. It does not define or parse `--setup_instance`, `--fit_mode`, `--setup_name`, or `--k_folds`. Running the command listed in the documentation will raise an `Unrecognized Arguments` exception.

### [x] `fit_mode` Inability to Support Evaluation-Only Runs
* **Behavior:** If `setup.fit_mode` is set to any value other than `0` (e.g., `1` for evaluating a pre-trained model), [main.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/main.py#L39-L47) completely skips the execution because the condition `if setup.fit_mode == 0 and not setup_and_model_exists(...)` is not met. There is no alternative block to perform evaluation-only execution when `fit_mode` is non-zero.

---

## 2. Execution Resume & File Check Logic

### [x] Improper Setup and Model Checks
* **Implementation ([files.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/recsysconfident/utils/files.py#L20-L23)):**
  ```python
  def setup_and_model_exists(run_folder: str):
      existent_setups = glob.glob(f"{run_folder}/setup-[0-9]*.json" )
      return len(existent_setups) > 0
  ```
* **Problems:**
  1. **Name-Implementation Mismatch:** Despite its name, `setup_and_model_exists` only checks if `setup-[fold].json` files are present; it does not check if the PyTorch model checkpoint file (`model-[fold].pth`) actually exists.
  2. **Interrupted Runs Blocked:** If a training run is interrupted or fails *after* writing the setup JSON file (line 43 of `main.py`) but *before* writing the evaluation metrics or error csv files, subsequent executions of `main.py` will not resume training. 
     - The outer skip check `setup_model_results_exists` evaluates to `False` (because metric or error files are missing).
     - The inner conditional `not setup_and_model_exists(environ.instance_dir)` evaluates to `False` (because the setup JSON exists).
     - As a result, the training and evaluation block is completely bypassed, leaving the run incomplete without raising an error.

---

## 3. Fold Configurations & Data Splitting

### [x] Setup Folds Ignored by Dataset Splitting
* **Behavior:** When instantiating `Environment` in [main.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/main.py#L28-L34), `setup.folds` is never passed to `Environment`. Consequently, `Environment` instantiates `DatasetInfo` using its default parameter value (`folds = 7`), ignoring the actual folds configured in the setup.
* **Problems:**
  1. If `setup.folds = 5` is defined in the JSON configuration, `main.py` will run 4 folds (0 to 3), but the dataset splits generated under `DatasetInfo` will partition the data into 7 folds.
  2. If `setup.folds` is configured to be larger than 7 (e.g. 10), `main.py` will loop up to index 8. Because `DatasetInfo` only holds 7 splits (`df_folds`), accessing `df_folds[fold+1]` on index 7 or 8 will raise an `IndexError`.

---

## 4. Multi-Stage Model Checkpoint Overwriting

### [x] Rating vs. Ranking Weight Overwrites
* **Implementation ([setup_manager.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/recsysconfident/setup_manager.py#L15-L44)):**
  For models that support ranking-based training (e.g., those implementing `switch_to_ranking` and `ranking_loss`), the training is performed in two consecutive steps:
  1. Fits the model in rating mode:
     ```python
     history = train_model(model, ..., path=environ.model_uri)
     ```
  2. Switches to ranking mode and fits again:
     ```python
     model.switch_to_ranking()
     history = train_model(model, ..., path=environ.model_uri)
     ```
* **Problem:** Both training steps save their respective best checkpoints to the *same* destination URI (`environ.model_uri`). The second stage (ranking) overwrites the model weights saved in the first stage (rating). At the end of training, `model.switch_to_rating()` is called, but the saved checkpoint file only contains ranking-optimized weights.

---

## 5. Dataloader Signature Mismatch & Model Registration

### [x] Missing `fold` Parameter and Unpacking Errors
Across most models in `recsysconfident/ml/models/`, the data loader function `ui_ids_label` is called incorrectly. For example, in [cp_ordrec_gat.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/recsysconfident/ml/models/distribution_based/cp_ordrec_gat.py#L15) and [mf.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/recsysconfident/ml/models/simple_confidence/mf.py#L10):
```python
fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)
```
* **Problems:**
  1. **Signature Mismatch:** `ui_ids_label` requires two arguments: `(info, fold)`. Calling it with only `info` raises `TypeError: ui_ids_label() missing 1 required positional argument: 'fold'`.
  2. **Unpacking Error:** `ui_ids_label` returns a tuple of length 2 `(fit_dataloader, eval_dataloader)`. Attempting to unpack it into 3 variables (`fit_dataloader, eval_dataloader, test_dataloader`) will raise `ValueError: not enough values to unpack (expected 3, got 2)`.

### [x] Missing Model Registration
* **Problem:** Multiple models implemented in the package (including `CGPRankRatingPred`, `OrdRec`, and `CPOrdrecGAT`) are not registered in the `self.model_name_fn` dictionary within `Environment.read_split_datasets` ([environment.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/recsysconfident/environment.py#L75-L81)). This prevents these models from being selected or executed via the main workflow of the project.

---

## 6. Broken Test Suite

### [x] Broken Imports of Non-Existent Files/Modules
1. **[test_eval.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/tests/recsysconfident/ml/fit_eval/test_eval.py#L5):**
   ```python
   from recsysconfident.ml.eval.eval import filter_out_users_less_than_k_inter
   ```
   * *Inconsistency:* The module `recsysconfident.ml.eval.eval` does not exist. The helper function `filter_out_users_less_than_k_inter` is actually defined in `recsysconfident/data_handling/miscellaneous.py`.
2. **[test_elementwise_bpr_loss.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/tests/recsysconfident/ml/fit_eval/test_elementwise_bpr_loss.py#L4):**
   ```python
   from recsysconfident.ml.eval.learn_elementwise_loss import sample_unseen_item, get_low_rank_items
   ```
   * *Inconsistency:* The module `recsysconfident.ml.eval.learn_elementwise_loss` does not exist. The function `sample_unseen_item` is defined in `recsysconfident/ml/ranking/rank_helper.py`.
3. **[test_conf_threshold_searcher.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/tests/recsysconfident/ml/fit_eval/test_conf_threshold_searcher.py#L6):**
   ```python
   from recsysconfident.ml.eval.conf_threshold_searcher import find_best_conf_threshold
   ```
   * *Inconsistency:* The module `conf_threshold_searcher` and the function `find_best_conf_threshold` are completely missing from the codebase.
4. **[test_splitting.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/tests/recsysconfident/data_handling/test_splitting.py#L7):**
   ```python
   from recsysconfident.data_handling.splitting import split_ratings
   ```
   * *Inconsistency:* The function `split_ratings` is not defined anywhere in the `splitting.py` module (which only implements `time_ordered_folds`).

### [x] Broken Test Implementations
* **[test_datasets.py](file:///home/joel/Documents/rating-pred-dynamic-conf-analysis/tests/recsysconfident/data_handling/datasets/test_datasets.py#L14-L25):**
  1. The instantiation of `DatasetInfo` misses required positional arguments `run_data_uri` and `metadata_columns`, causing a `TypeError`.
  2. The call to `build` passes `0` as the `items_df` dataframe argument.
  3. The tests attempt to access `self.dataset.fit_df`, `self.dataset.val_df`, and `self.dataset.test_df` which do not exist under the `DatasetInfo` class (it only stores the folds under `self.df_folds`).

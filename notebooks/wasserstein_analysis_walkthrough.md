# Walkthrough: Generalized Wasserstein Regularization Analysis

This document summarizes the generalized and renamed Jupyter notebook and the methodology used to analyze whether Wasserstein regularization harms the recommendation performance of different models across various datasets.

---

## 1. Updated & Created Files

### [Jupyter Notebook: notebooks/wasserstein-analysis.ipynb](file:///home/joel/rating-pred-dynamic-conf-analysis/notebooks/wasserstein-analysis.ipynb)
A fully functional, generalized Jupyter notebook designed with dynamic path resolution, robust JSON parsing, quantitative summarization, publication-quality data visualization for any combinations of models, datasets, arbitrary metrics, and ratings distribution comparison charts.

---

## 2. Implementation & Design Details

### A. Generalization to Any Models & Datasets
The notebook has been fully generalized by introducing loop structures and config variables. You can analyze any quantity of models and datasets by updating the variables in the setup cell:
```python
models = ['mf_wasserstein']
datasets = ['ml-1m']
```
*Note: The `models` variable now holds the complete name of the models (e.g. `'mf_wasserstein'`). Hardcoded string concatenations of `_wasserstein` suffixes have been completely removed.*

### B. Generalized and Extensible Metric Definitions
The notebook supports arbitrary metrics via a generalized `metrics` definition list. This specifies the metric key inside the `metrics-*.json` files, the user-facing display name, and a direction description:
```python
metrics = [
    ('rmse', 'RMSE', 'lower is better'),
    ('mae', 'MAE', 'lower is better'),
    ('mNDCG@10', 'NDCG@10', 'higher is better'),
    ('MAP@10', 'MAP@10', 'higher is better'),
    ('kl_diverence', 'KL Divergence', 'lower is better')
]
```
Adding, removing, or renaming metrics in this list will automatically update:
- Metric loading logic.
- Quantitative summary aggregation.
- Subplot layouts and labels in the visual charts.

### C. Dynamic Path Resolution
To ensure a frictionless experience when you run the notebook, it automatically detects if it is executed from the **project root** or from the **`notebooks` directory**:
```python
if os.path.exists("runs"):
    runs_dir = "runs"
elif os.path.exists("../runs"):
    runs_dir = "../runs"
else:
    runs_dir = "runs"
```

### D. Multi-Model and Multi-Dataset Summaries
- For each unique dataset and model combination found in the metrics, the notebook displays a dedicated Quantitative Summary Table.
- It dynamically aggregates and calculates the `mean` and `std` of all metrics defined in the configuration.

### E. Aesthetic Parallel Visualization (Dynamic Subplots with Direction Labels)
- **Dynamic Layout Grid**: Automatically creates a single horizontal subplot row with shape `(1, len(metrics))` matching the number of active metrics configured.
- **Boxplot Distribution**: For each regularization level, a boxplot shows the metric distribution over the folds.
- **Fold Mean Indicator**: A distinct, vibrant red triangle (`^`) highlights the mean over the folds within each boxplot.
- **Mean Trend Line**: A dashed trend line connects the fold means across the different regularization levels to easily visualize performance trends.
- **Dynamic Metric Directions**: Automatically includes metric direction descriptors (e.g., `"lower is better"` or `"higher is better"`) on the Y-axes.
- **Clean Scientific Aesthetics**:
  - Omits titles on the plots for a clean, publication-ready design.
  - Subtle dotted grid lines (`:`) to guide the eyes without cluttering.
  - Modern sans-serif typography and clean spines (top and right spines are hidden).
- **Dynamic Image Saving**: The plot is saved with high-resolution (300 DPI) to `{database_name}-{model_name}-wasserstein.png`.

### F. Ratings Distribution Comparison Charts
- **Horizontal 2 x n Grid**: Plotted for the configured datasets (where *n* is the number of datasets).
- **Observed Ratings (Row 1)**: Discrete probability distribution bar chart of the actual rating values from the test fold (the last fold *k* of the time series cross validation).
- **Model Predicted Ratings (Row 2)**: Continuous predicted rating values distribution density with a smooth Kernel Density Estimate (KDE) plot and a configured `binwidth=0.5` to clearly show how values fall within the discrete integer ranges, computed from the `eval_error_conf-{k}.csv` file's `r_pred` column.
- **Visual Design**: Sleek color palette (SlateBlue observed vs. Tomato predicted), clean dotted grid backgrounds, standard spines, and aligned X-axis scales (`[0.5, 5.5]`).
- **Dynamic Image Saving**: Automatically saves the high-resolution comparison plot to `ratings-distribution-comparison.png`.

---

## 3. How to Execute the Analysis

1. Open your Jupyter interface.
2. Open the [notebooks/wasserstein-analysis.ipynb](file:///home/joel/rating-pred-dynamic-conf-analysis/notebooks/wasserstein-analysis.ipynb) notebook.
3. Configure the `models`, `datasets`, and `metrics` list in Cell 3 as needed.
4. Run all cells (`Cell` > `Run All` or `Ctrl + F9` / `Shift + Enter` on each cell).
5. **Output Results**:
   - The summary table will render as beautiful HTML tables for each combination.
   - The horizontal boxplots will render and save automatically.
   - The observed vs. predicted ratings distribution grid will render and save dynamically to `ratings-distribution-comparison.png`.

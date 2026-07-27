"""
The present script produces the following figure (or figure panels) in Garza et al 2026:
- Extended Data Fig. 7

Overview:
This script compares parameter correlations between synthetic DDM model fits
(control and sampling variants) and real experimental zebrafish model fits.
It bootstraps parameter samples to estimate correlation matrices and their
variability, computes p-values for deviations from baseline, and creates
multi-panel figures showing correlation structure and trajectories.

This plot expands on Extended Data Figure 7 shown in Garza et al 2026, by
providing further visualization of Pearson correlation, matrices of p-values
related to the bootstrap statistical test, and the different correlation values
across ages and mutations.
"""

import itertools

import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import dotenv_values
from mpl_toolkits.axes_grid1 import make_axes_locatable

from figures.style import BehavioralModelStyle
from service.behavioral_processing import BehavioralProcessing
from service.figure_helper import Figure
from utils.configuration_ddm import ConfigurationDDM

# ------------------------------------------------------------
# Load env and paths
# ------------------------------------------------------------
# Load environment variables with input/output paths
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])     # Input data directory
path_save = Path(env['PATH_SAVE'])   # Output directory for figures

# paths for synthetic controls and real fish datasets
path_data_control = Path(fr"{path_dir}/benchmark/base_dataset")
path_data_fish = Path(fr"{path_dir}/base_dataset_5dpfWT")

# ------------------------------------------------------------
# Configuration analysis
# ------------------------------------------------------------
number_bootstraps = int(1e4)   # Number of bootstrap iterations
sample_percentage_size = 1     # Fraction of samples used in each bootstrap
high_corr_threshold = 0.5      # Threshold used when marking "high" correlations
corr_threshold = 0.6
no_corr_threshold = 0.3
p_value_threshold = 0.01       # Significance threshold for p-values

# ------------------------------------------------------------
# Configurations figure
# ------------------------------------------------------------
style = BehavioralModelStyle()

# Starting positions for placing plots on the figure grid
xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

# Dimensions for different plot sizes
plot_height = style.plot_height
plot_height_small = plot_height / 2
plot_width = style.plot_width
plot_width_small = style.plot_width_small

# Padding inside and between plots
padding = style.padding_small

# color palette from style
palette = style.palette["default"]

# Feature toggles (turn visualization sections on/off)
do_basic_computation = True  # keep always True in the present version of the script
show_parameter_space = True
show_correlation_matrices = True
show_trajectory_correlation = False

# ------------------------------------------------------------
# Make a standard figure
# ------------------------------------------------------------
fig = Figure()


# ----------------------------------------------------------------------------
# SECTION 1: BASIC COMPUTATION
# Loads synthetic (control/sampling) and real fish DDM parameter fits,
# builds bootstrap distributions of their correlation matrices, and derives
# p-values / Cohen's d effect sizes comparing fish vs. synthetic-control data.
# ----------------------------------------------------------------------------
if do_basic_computation:
    ##### correlation matrices
    # control synthetic datasets
    model_dict = {}
    model_dict_nofit = {}
    # Two dicts collect synthetic-control model files found in the benchmark folder:
    # model_dict holds files whose name ends in 'fit.hdf5' (already fitted results);
    # model_dict_nofit holds the raw (non-fit) sampling files.
    # Iterate files in synthetic control directory and separate fitted vs not-fitted files
    for model_filepath in path_data_control.glob('model_test_*.hdf5'):
        model_filename = str(model_filepath.name)
        if model_filename.endswith("fit.hdf5"):
            model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}
        else:
            model_dict_nofit[model_filename.split("_")[2]] = {"data": model_filepath}

    # --- 'Sampling' synthetic dataset (no explicit optimization fit) ---
    # For each raw sampling file, the best (lowest-score) row is kept as that model's parameter set.
    # sampling (synthetic data without explicit fit results — take best-scoring record)
    model_array_sampling = np.zeros((len(ConfigurationDDM.parameter_list), len(model_dict_nofit.keys())))
    model_dict_sampling = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    model_dict_sampling["id"] = []
    i_m = 0
    # For each no-fit synthetic model, read the table and select the best-scoring row
    # Loop over every no-fit synthetic model: load its result table, keep the best-scoring row,
    # and store each DDM parameter value plus the model's ID.
    for i_model, id_model in enumerate(model_dict_nofit.keys()):
        df_model_fit_list = pd.read_hdf(model_dict_nofit[id_model]["data"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            model_array_sampling[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_sampling[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_sampling["id"].append(id_model)
        i_m += 1

    # --- 'Control' synthetic dataset (already-fitted models) ---
    # Same best-score selection logic as above, but applied to the pre-fitted synthetic files.
    # fit synthetic fish (models that have explicit fit results)
    model_array_control = np.zeros((len(ConfigurationDDM.parameter_list), len(model_dict.keys())))
    model_dict_control = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    model_dict_control["id"] = []
    i_m = 0
    # For each fitted synthetic model, pick the best-scoring fit and extract parameters
    # Loop over every fitted synthetic model, extract best-scoring parameter row.
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            model_array_control[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_control[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_control["id"].append(id_model)
        i_m += 1
    # Keep a transposed list-of-arrays copy of the control parameter matrix (one row per model) for potential downstream use.
    theta_fish_control_list = list(model_array_control.T)

    # --- Real (experimental) zebrafish dataset ---
    # Collect fitted model files from the main experimental data directory.
    # real fish — load fitted models from experimental dataset directory
    model_dict = {}
    for model_filepath in path_data_fish.glob('model_*_fit.hdf5'):
        model_filename = str(model_filepath.name)

        model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

    model_array = np.zeros((len(ConfigurationDDM.parameter_list), len(model_dict.keys())))
    model_dict_fish = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    model_dict_fish["id"] = []
    i_m = 0
    # Extract best fit parameters for each real fish model
    # Loop over every real-fish fitted model, extract the best-scoring parameter row (same pattern as above).
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            model_array[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_fish[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_fish["id"].append(id_model)
        i_m += 1

    # --- Build indexed DataFrames and bootstrap-resample each dataset ---
    # Each DataFrame is indexed by model/animal ID so that identity is preserved across resamples.
    # 'randomly_sample_df' draws 'number_bootstraps' resampled copies (with replacement) of the DataFrame,
    # used later to build an empirical distribution of the correlation matrix.
    # Prepare DataFrames for bootstrap resampling and create storage tensors
    df_model_sampling_original = pd.DataFrame(model_dict_sampling)
    df_model_sampling_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_sampling_list = BehavioralProcessing.randomly_sample_df(df=df_model_sampling_original,
                                                                     sample_number=number_bootstraps,
                                                                     sample_percentage_size=sample_percentage_size,
                                                                     with_replacement=True)
    relation_tensor_sampling = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

    df_model_control_original = pd.DataFrame(model_dict_control)
    df_model_control_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_control_list = BehavioralProcessing.randomly_sample_df(df=df_model_control_original,
                                                                    sample_number=number_bootstraps,
                                                                    sample_percentage_size=sample_percentage_size,
                                                                    with_replacement=True)
    relation_tensor_control = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

    df_model_fish_original = pd.DataFrame(model_dict_fish)
    df_model_fish_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_fish_list = BehavioralProcessing.randomly_sample_df(df=df_model_fish_original,
                                                                 sample_number=number_bootstraps,
                                                                 sample_percentage_size=sample_percentage_size,
                                                                 with_replacement=True)
    relation_tensor_fish = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

    # --- Baseline (observed) difference in correlation structure: sampling vs. control synthetic data ---
    # This 'delta' will be compared against the null-distribution deltas below to compute a p-value.
    # Compute baseline differences between sampling and control correlations (absolute difference)
    delta_corr_synthetic = np.abs(
        np.array(df_model_control_original.corr()) - np.array(df_model_sampling_original.corr()))
    # Pool sampling+control data together and draw two independent sets of bootstrap resamples;
    # differences between these two random draws approximate the null distribution of 'no real difference'.
    # Combine synthetic sampling and control to compute null distributions via resampling
    df_model_synthetic_combined_original = pd.concat((df_model_sampling_original, df_model_control_original))
    df_model_synthetic_combined_list_0 = BehavioralProcessing.randomly_sample_df(
        df=df_model_synthetic_combined_original,
        sample_number=number_bootstraps,
        sample_percentage_size=sample_percentage_size,
        with_replacement=True)
    df_model_synthetic_combined_list_1 = BehavioralProcessing.randomly_sample_df(
        df=df_model_synthetic_combined_original,
        sample_number=number_bootstraps,
        sample_percentage_size=sample_percentage_size,
        with_replacement=True)
    relation_tensor_synthetic_combined_delta = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

    # --- Same baseline-difference logic, but for real fish vs. control synthetic data ---
    # Control correlation used as baseline for fish comparison
    relation_fish_original = np.array(df_model_fish_original.corr())
    relation_control_original = np.array(df_model_control_original.corr())
    delta_corr_fish = np.abs(np.array(df_model_fish_original.corr()) - relation_control_original)
    # Pool fish+control data together and draw two independent bootstrap resample sets for the null distribution.
    # Combined fish+control resampling for null distributions
    df_model_fish_combined_original = pd.concat((df_model_fish_original, df_model_control_original))
    df_model_fish_combined_list_0 = BehavioralProcessing.randomly_sample_df(df=df_model_fish_combined_original,
                                                                            sample_number=number_bootstraps,
                                                                            sample_percentage_size=sample_percentage_size,
                                                                            with_replacement=True)
    df_model_fish_combined_list_1 = BehavioralProcessing.randomly_sample_df(df=df_model_fish_combined_original,
                                                                            sample_number=number_bootstraps,
                                                                            sample_percentage_size=sample_percentage_size,
                                                                            with_replacement=True)
    relation_tensor_fish_combined_delta = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

    # --- Main bootstrap loop ---
    # For every bootstrap draw, compute the Pearson correlation matrix of each resampled dataset,
    # and the null-distribution correlation deltas (difference between two independently resampled combined sets).
    # For each bootstrap iteration, compute correlation matrices for sampled datasets
    for i_df in range(number_bootstraps):
        df_model_sampling = df_model_sampling_list[i_df]
        # df_model_control.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_sampling[i_df] = np.array(df_model_sampling.corr())

        df_model_control = df_model_control_list[i_df]
        # df_model_control.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_control[i_df] = np.array(df_model_control.corr())

        df_model_fish = df_model_fish_list[i_df]
        # df_model_fish.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_fish[i_df] = np.array(df_model_fish.corr())

        df_model_synthetic_combined_0 = df_model_synthetic_combined_list_0[i_df]
        corr_synthetic_combined_0 = np.array(df_model_synthetic_combined_0.corr())
        df_model_synthetic_combined_1 = df_model_synthetic_combined_list_1[i_df]
        corr_synthetic_combined_1 = np.array(df_model_synthetic_combined_1.corr())
        relation_tensor_synthetic_combined_delta[i_df] = np.abs(corr_synthetic_combined_0 - corr_synthetic_combined_1)

        df_model_fish_combined_0 = df_model_fish_combined_list_0[i_df]
        corr_fish_combined_0 = np.array(df_model_fish_combined_0.corr())
        df_model_fish_combined_1 = df_model_fish_combined_list_1[i_df]
        corr_fish_combined_1 = np.array(df_model_fish_combined_1.corr())
        relation_tensor_fish_combined_delta[i_df] = np.abs(corr_fish_combined_0 - corr_fish_combined_1)

    # --- Convert bootstrap distributions into p-values and summary statistics ---
    # p-value = fraction of null-distribution deltas that are as extreme or more extreme than the observed delta.
    # Values on/above the diagonal (upper triangle, including diagonal) are set to 1 (not tested / not meaningful).
    # check statistical difference of biology from baseline of the model
    p_value_synthetic = np.mean(relation_tensor_synthetic_combined_delta >= delta_corr_synthetic, axis=0)
    # Boolean mask flagging which parameter-pairs show an 'acceptable' (non-significant) difference
    # between the two synthetic variants — used later to gate which trajectory results are considered valid.
    p_value_corr_acceptable = p_value_synthetic > p_value_threshold
    p_value_synthetic[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 1

    p_value_fish = np.mean(relation_tensor_fish_combined_delta >= delta_corr_fish, axis=0)
    p_value_fish[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 1

    # Average correlation matrices across all bootstrap draws (element-wise mean), and zero out the
    # upper triangle (including diagonal) since correlation matrices are symmetric and the diagonal is trivially 1.
    relation_matrix_sampling = np.mean(relation_tensor_sampling, axis=0)
    relation_matrix_sampling[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 0
    relation_matrix_sampling_std = np.std(relation_tensor_control, axis=0)

    relation_matrix_control = np.mean(relation_tensor_control, axis=0)
    relation_matrix_control[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 0
    relation_matrix_control_std = np.std(relation_tensor_control, axis=0)

    relation_matrix_fish = np.mean(relation_tensor_fish, axis=0)
    relation_matrix_fish[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 0
    relation_matrix_fish_std = np.std(relation_tensor_fish, axis=0)

    # --- Effect size (Cohen's d) between fish and control correlation matrices ---
    # Pooled standard deviation across bootstrap draws (identity matrix added to avoid divide-by-zero on the diagonal).
    # Cells failing the significance threshold are masked out (set to NaN) so only significant differences are shown.
    pooled_std = np.eye(relation_matrix_fish_std.shape[0]) + np.sqrt((relation_matrix_fish_std**2 + relation_matrix_control_std**2) / 2)
    cohens_d_matrix = (relation_matrix_fish - relation_matrix_control) / pooled_std
    cohens_d_matrix[np.where(p_value_fish > p_value_threshold)] = np.nan

    # --- Effect size (Cohen's d) between sampling and control synthetic correlation matrices (same logic as above) ---
    pooled_std_control = np.eye(relation_matrix_sampling.shape[0]) + np.sqrt((relation_matrix_sampling_std**2 + relation_matrix_control_std**2) / 2)
    cohens_d_matrix_control = (relation_matrix_sampling - relation_matrix_control) / pooled_std_control
    cohens_d_matrix_control[np.where(p_value_synthetic > p_value_threshold)] = np.nan

# ----------------------------------------------------------------------------
# SECTION 2: PARAMETER-SPACE SCATTERPLOTS
# For every pair of DDM parameters, draw a scatterplot of real fish parameter
# values (one point per fish) to visualize the raw parameter-space structure.
# ----------------------------------------------------------------------------
if show_parameter_space:
    # scatterplots in parameter space
    model_dict = {}
    for model_filepath in path_data_fish.glob('model_*_fit.hdf5'):
        model_filename = str(model_filepath.name)
        model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

    model_array = np.zeros((len(ConfigurationDDM.parameter_list), len(model_dict.keys())))
    i_m = 0
    # Extract best-fit parameter values for every real fish model (same best-score logic as Section 1).
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            model_array[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
        i_m += 1

    # Nested loop building a full grid of scatterplots: rows = one parameter (y-axis),
    # columns = another parameter (x-axis), covering all pairwise combinations.
    for i_y, parameter_y in enumerate(ConfigurationDDM.parameter_list):
        for i_x, parameter_x in enumerate(ConfigurationDDM.parameter_list):
            y_p = model_array[i_y, :]
            x_p = model_array[i_x, :]

            # process and plot
            plot_xy = fig.create_plot(plot_label=style.get_plot_label() if i_x == 0 and i_y == 0 else None,
                                      xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                      xmin=parameter_x["min"], xmax=parameter_x["max"],
                                      xticks=[parameter_x["min"], parameter_x["max"]],
                                      xl=parameter_x["label_show"] if i_y == 0 else None,
                                      yl=parameter_y["label_show"] if i_x == 0 else None,
                                      ymin=parameter_y["min"], ymax=parameter_y["max"],
                                      yticks=[parameter_y["min"], parameter_y["max"]],
                                      hlines=[0], vlines=[0])
            xpos += padding + plot_width

            # Draw the scatter of fish values for this parameter pair (elw=0 removes marker edge line width).
            plot_xy.draw_scatter(x_p, y_p, elw=0, alpha=0.5)

        # Move to the next row of the grid after finishing a row of columns.
        xpos = xpos_start
        ypos -= padding + plot_height
    # Extra vertical gap after the full scatterplot grid before the next section starts.
    ypos -= padding + plot_height

# ----------------------------------------------------------------------------
# SECTION 3: CORRELATION MATRICES AND P-VALUE / EFFECT-SIZE HEATMAPS
# Plots side-by-side heatmaps of the correlation matrices (sampling synthetic,
# fit synthetic/control, real fish), their p-value matrices, and Cohen's d
# effect-size matrices, followed by per-age and per-genotype Cohen's d panels.
# ----------------------------------------------------------------------------
if show_correlation_matrices:
    # configuration this plot
    plot_width_matrix = plot_width * 1.6
    plot_size_matrix = plot_width * 1.7

    # correlation matrices
    # --- Heatmap 1: correlation matrix of the sampling synthetic dataset ---
    plot_sampling = fig.create_plot(plot_label=style.get_plot_label(),
                                    plot_title="Correlation matrix\nsampling synthetic",
                                    xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                    plot_width=plot_size_matrix,
                                    xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5, xticklabels_rotation=90,
                                    xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                    xticklabels=[p["label_show"].capitalize() for p in ConfigurationDDM.parameter_list],
                                    ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5,
                                    yticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                    yticklabels=[p["label_show"].capitalize() for p in
                                                 reversed(ConfigurationDDM.parameter_list)], )
    xpos += padding + plot_size_matrix

    # Build coordinate grids (x, y) matching matrix indices; used implicitly by draw_image for pixel placement.
    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_sampling.draw_image(relation_matrix_sampling, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                             len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                                  colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    # --- Heatmap 2: correlation matrix of the fitted ('control') synthetic dataset ---
    plot_control = fig.create_plot(plot_title="Correlation matrix\nfit synthetic",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                   xticklabels=[p["label_show"].capitalize() for p in ConfigurationDDM.parameter_list],
                                   ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5,
                                   yticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                   yticklabels=[p["label_show"].capitalize() for p in
                                                reversed(ConfigurationDDM.parameter_list)], )
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_control.draw_image(relation_matrix_control, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                           len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                                 colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    # --- Heatmap 3: correlation matrix of the real experimental fish dataset, with colorbar ---
    plot_fish = fig.create_plot(plot_title="Correlation matrix\nexperiment",
                                xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5, xticklabels_rotation=90,
                                xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                xticklabels=[p["label_show"].capitalize() for p in ConfigurationDDM.parameter_list],
                                ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5)

    xpos += padding * 2 + plot_size_matrix

    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_fish.draw_image(relation_matrix_fish, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                     len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                              colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    # Attach a slim colorbar axis to the right of the fish correlation heatmap.
    divider = make_axes_locatable(plot_fish.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_fish.figure.fig.colorbar(im, cax=cax, orientation='vertical',
                                  ticks=[-1, -corr_threshold, 0, corr_threshold, 1])

    # matrices of p-values
    # --- Heatmap 4: p-value matrix testing sampling-vs-control synthetic correlation differences ---
    plot_fit = fig.create_plot(plot_title="p-value\ncorrelation fit",
                               xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                               plot_width=plot_size_matrix,
                               xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5, xticklabels_rotation=90,
                               xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                               xticklabels=[p["label_show"].capitalize() for p in ConfigurationDDM.parameter_list],
                               ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5,
                               yticks=np.arange(len(ConfigurationDDM.parameter_list)),
                               yticklabels=[p["label_show"].capitalize() for p in
                                            reversed(ConfigurationDDM.parameter_list)], )
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_fit.draw_image(p_value_synthetic, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                 len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                             colormap='gray', zmin=0, zmax=0.1, image_interpolation=None)

    # --- Heatmap 5: p-value matrix testing fish-vs-control correlation differences (the 'corrected'/biological comparison), with colorbar ---
    plot_corrected = fig.create_plot(plot_title="p-value\ncorrelation corrected",
                                     xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                     plot_width=plot_size_matrix,
                                     xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5,
                                     xticklabels_rotation=90,
                                     xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                     xticklabels=[p["label_show"].capitalize() for p in
                                                  ConfigurationDDM.parameter_list],
                                     ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5)
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_corrected.draw_image(p_value_fish, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                  len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                                   colormap='gray', zmin=0, zmax=0.1, image_interpolation=None)

    # Attach colorbar to the p-value heatmap, with ticks at common significance thresholds.
    divider = make_axes_locatable(plot_corrected.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_corrected.figure.fig.colorbar(im, cax=cax, orientation='vertical', ticks=[0, 0.01, 0.05, 0.1])

    # --- Heatmap 6: Cohen's d effect size for sampling-vs-control synthetic correlations (masked by significance), with colorbar ---
    plot_cohens_d_control = fig.create_plot(plot_title="Cohen's d Fit",
                                    xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                    plot_width=plot_size_matrix,
                                    xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5,
                                    xticklabels_rotation=90,
                                    xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                    xticklabels=[p["label_show"].capitalize() for p in
                                                 ConfigurationDDM.parameter_list],
                                    ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5)
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(ConfigurationDDM.parameter_list))
    x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
    y = x.T
    im = plot_cohens_d_control.draw_image(cohens_d_matrix_control, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                    len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                                  colormap='PRGn', zmin=-1, zmax=1, image_interpolation=None)

    # Attach colorbar to the Cohen's d heatmap, ticks spanning the -1 to 1 effect-size range.
    divider = make_axes_locatable(plot_cohens_d_control.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_cohens_d_control.figure.fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-1, -0.5, 0, 0.5, 1])

    # Reset plotting cursor to a new row for the upcoming per-group (age/genotype) Cohen's d panels.
    xpos = xpos_start
    ypos -= padding * 2 + plot_size_matrix
    # List of experimental subgroups to compare against the synthetic control baseline:
    # 5 developmental ages (5-9 days post-fertilization) followed by scn1lab and disc1 genotype groups
    # (two independent scn1lab clutches, plus disc1 wild-type/heterozygous/homozygous mutants).
    models_in_age_list = [
        {"label_show": "5dpf",
         "path": fr"{path_dir}/age_analysis/5_dpf", },
        {"label_show": "6dpf",
         "path": fr"{path_dir}/age_analysis/6_dpf", },
        {"label_show": "7dpf",
         "path": fr"{path_dir}/age_analysis/7_dpf", },
        {"label_show": "8dpf",
         "path": fr"{path_dir}/age_analysis/8_dpf", },
        {"label_show": "9dpf",
         "path": fr"{path_dir}/age_analysis/9_dpf", },
        {"label_show": "1 scn1lab+/+",
         "path": fr"{path_dir}/genome_analysis/scn1lab_NIBR_20200708/wt", },
        {"label_show": "1 scn1lab+/-",
         "path": fr"{path_dir}/genome_analysis/scn1lab_NIBR_20200708/het", },
        {"label_show": "2 scn1lab+/+",
         "path": fr"{path_dir}/genome_analysis/scn1lab_zirc_20200710/wt", },
        {"label_show": "2 scn1lab+/-",
         "path": fr"{path_dir}/genome_analysis/scn1lab_zirc_20200710/het", },
        {"label_show": "disc+/+",
         "path": fr"{path_dir}/genome_analysis/disc1_hetnix/wt", },
        {"label_show": "disc+/-",
         "path": fr"{path_dir}/genome_analysis/disc1_hetnix/het", },
        {"label_show": "disc-/-",
         "path": fr"{path_dir}/genome_analysis/disc1_hetnix/hom", },
    ]
    # Accumulator for each group's Cohen's d matrix (kept for potential later inspection/reuse).
    cohens_d_matrix_age_traj = []
    # --- For each subgroup (age or genotype), repeat the bootstrap correlation & Cohen's d pipeline from Section 1 ---
    for i_m, m in enumerate(models_in_age_list):
        path = m["path"]
        i_m_ = 0
        model_list = [[] for i_p in range(len(ConfigurationDDM.parameter_list))]
        model_dict_group = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
        model_dict_group["id"] = []
        # Load every fitted model file for this subgroup and keep the best-scoring parameter row per animal.
        for model_filepath in Path(path).glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            df_model_fit_list = pd.read_hdf(model_filepath)
            best_score = np.min(df_model_fit_list['score'])
            df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

            for i_p, p in enumerate(ConfigurationDDM.parameter_list):
                model_list[i_p].append(df_model_fit[p["label"]].iloc[0])
                model_dict_group[p["label"]].append(df_model_fit[p["label"]][0])
            model_dict_group["id"].append(model_filename.split("_")[2])
            i_m_ += 1

        # Build indexed DataFrame for this subgroup and bootstrap-resample it (same logic as Section 1).
        model_array = np.array(model_list)
        df_model_group_original = pd.DataFrame(model_dict_group)
        df_model_group_original.set_index('id', inplace=True)

        df_model_group_list = BehavioralProcessing.randomly_sample_df(
            df=df_model_group_original,
            sample_number=number_bootstraps,
            sample_percentage_size=sample_percentage_size,
            with_replacement=True
        )
        relation_tensor_group = np.zeros((number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))
        # Compute correlation matrix for each bootstrap resample of this subgroup.
        for i_df in range(number_bootstraps):
            df_model_group = df_model_group_list[i_df]
            relation_tensor_group[i_df] = np.array(df_model_group[[p["label"] for p in ConfigurationDDM.parameter_list]].corr())

        # Mean and std of the subgroup's bootstrap correlation matrices (upper triangle zeroed as before).
        relation_matrix_group = np.mean(relation_tensor_group, axis=0)
        relation_matrix_group[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 0
        relation_matrix_group_std = np.std(relation_tensor_group, axis=0)

        # Recompute the group's true (non-bootstrap) correlation matrix and its absolute difference from control,
        # then build the same combined-resample null distribution as in Section 1 to get a p-value for this subgroup.
        # Control correlation used as baseline for fish comparison
        relation_group_original = np.array(df_model_group_original.corr())
        relation_control_original = np.array(df_model_control_original.corr())
        delta_corr_group = np.abs(np.array(df_model_group_original.corr()) - relation_control_original)
        # Combined fish+control resampling for null distributions
        df_model_group_combined_original = pd.concat((df_model_group_original, df_model_control_original))
        df_model_group_combined_list_0 = BehavioralProcessing.randomly_sample_df(df=df_model_group_combined_original,
                                                                                sample_number=number_bootstraps,
                                                                                sample_percentage_size=sample_percentage_size,
                                                                                with_replacement=True)
        df_model_group_combined_list_1 = BehavioralProcessing.randomly_sample_df(df=df_model_group_combined_original,
                                                                                sample_number=number_bootstraps,
                                                                                sample_percentage_size=sample_percentage_size,
                                                                                with_replacement=True)
        relation_tensor_group_combined_delta = np.zeros(
            (number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

        # Bootstrap loop: correlation deltas between two independent resamples of the combined group+control data.
        # For each bootstrap iteration, compute correlation matrices for sampled datasets
        for i_df in range(number_bootstraps):
            df_model_group_combined_0 = df_model_group_combined_list_0[i_df]
            corr_group_combined_0 = np.array(df_model_group_combined_0.corr())
            df_model_group_combined_1 = df_model_group_combined_list_1[i_df]
            corr_group_combined_1 = np.array(df_model_group_combined_1.corr())
            relation_tensor_group_combined_delta[i_df] = np.abs(corr_group_combined_0 - corr_group_combined_1)

        # p-value for this subgroup's correlation vs. control, with upper-triangle cells set to 1 (untested).
        p_value_group = np.mean(relation_tensor_group_combined_delta >= delta_corr_group, axis=0)
        p_value_group[np.triu_indices(len(ConfigurationDDM.parameter_list))] = 1

        # Cohen's d effect size for this subgroup vs. control; mask out cells that are not significant either
        # for this specific comparison (p_value_group) or for the earlier sampling-vs-control reference test (p_value_corr_acceptable).
        pooled_std = np.eye(relation_matrix_group_std.shape[0]) + np.sqrt((relation_matrix_group_std ** 2 + relation_matrix_control_std ** 2) / 2)
        cohens_d_matrix_age = (relation_matrix_group - relation_matrix_control) / pooled_std
        cohens_d_matrix_age[np.where(p_value_group > p_value_threshold)] = np.nan
        cohens_d_matrix_age[np.where(np.logical_not(p_value_corr_acceptable))] = np.nan

        cohens_d_matrix_age_traj.append(cohens_d_matrix_age)

        # Create and draw this subgroup's Cohen's d heatmap panel, wrapping to a new row every 5 panels.
        plot_cohens_d = fig.create_plot(plot_title=f"Cohen's d {m['label_show']}",
                                         xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                         plot_width=plot_size_matrix,
                                         xmin=-0.5, xmax=len(ConfigurationDDM.parameter_list) - 0.5,
                                         xticklabels_rotation=90,
                                         xticks=np.arange(len(ConfigurationDDM.parameter_list)),
                                         xticklabels=[p["label_show"].capitalize() for p in
                                                      ConfigurationDDM.parameter_list],
                                         ymin=-0.5, ymax=len(ConfigurationDDM.parameter_list) - 0.5)
        xpos += padding + plot_size_matrix
        # Wrap to a new row of panels after every 5 groups plotted.
        if i_m == 4:
            xpos = xpos_start
            ypos -= padding*2 + plot_size_matrix

        x_ = np.arange(len(ConfigurationDDM.parameter_list))
        x = np.tile(x_, (len(ConfigurationDDM.parameter_list), 1))
        y = x.T
        im = plot_cohens_d.draw_image(cohens_d_matrix_age, (-0.5, len(ConfigurationDDM.parameter_list) - 0.5,
                                                      len(ConfigurationDDM.parameter_list) - 0.5, -0.5),
                                       colormap='PRGn', zmin=-1, zmax=1, image_interpolation=None)

        # Attach a shared colorbar only once, after the very last subgroup panel has been drawn.
        if i_m == len(models_in_age_list)-1:
            divider = make_axes_locatable(plot_cohens_d.ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plot_cohens_d.figure.fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-1, -0.5, 0, 0.5, 1])

    xpos = xpos_start
    ypos -= padding * 2 + plot_size_matrix

# ----------------------------------------------------------------------------
# SECTION 4: PARAMETER-CORRELATION TRAJECTORIES (disabled by default)
# Defines and calls a helper function that tracks how specific parameter-pair
# correlations change across ordered groups (ages) or categorical groups
# (genotypes), plotting one small trajectory subplot per parameter pair.
# ----------------------------------------------------------------------------
if show_trajectory_correlation:
    padding_here = style.plot_size
    xpos_start_here = xpos
    ypos_start_here = ypos
    # all datasets for which to compute parameters correlations
    # Ordered group list for the age trajectory (developmental stage in days post-fertilization).
    models_in_age_list = [
        {"label_show": "5dpf",
         "path": fr"{path_dir}/age_analysis/5_dpf", },
        {"label_show": "6dpf",
         "path": fr"{path_dir}/age_analysis/6_dpf", },
        {"label_show": "7dpf",
         "path": fr"{path_dir}/age_analysis/7_dpf", },
        {"label_show": "8dpf",
         "path": fr"{path_dir}/age_analysis/8_dpf", },
        {"label_show": "9dpf",
         "path": fr"{path_dir}/age_analysis/9_dpf", },
    ]
    # Group list for the scn1lab genotype trajectory (wild-type vs. heterozygous).
    models_in_mutation_scn_list = [
        {"label_show": "scn1lab +/+",
         "path": fr"{path_dir}/harpaz_2021/scn1lab_NIBR_20200708/wt", },
        {"label_show": "scn1lab +/-",
         "path": fr"{path_dir}/harpaz_2021/scn1lab_NIBR_20200708/het", },
    ]
    # Group list for the disc1 genotype trajectory (wild-type / heterozygous / homozygous).
    models_in_mutation_disc_list = [
        {"label_show": "disc1 +/+",
         "path": fr"{path_dir}/harpaz_2021/disc1_hetnix/wt", },
        {"label_show": "disc1 +/-",
         "path": fr"{path_dir}/harpaz_2021/disc1_hetnix/het", },
        {"label_show": "disc1 -/-",
         "path": fr"{path_dir}/harpaz_2021/disc1_hetnix/hom", },
    ]


    # Local helper function (defined here, not module-level) that performs the full bootstrap +
    # significance pipeline for an arbitrary list of model groups, then draws one subplot per
    # parameter pair showing how the mean correlation (with error bars) evolves across the groups.
    def show_trajectory_correlation(models_list, fig, xpos_start_here, ypos_start_here, xl=None, xticklabels=None,
                                    plot_width_here=plot_width_small, xticklabels_rotation=None,
                                    show_title=False, show_scalebar=False, show_ticks=False,
                                    p_value_corr_acceptable=None):
        """
        Plot the correlation trajectory of DDM parameters across a list of models.
        This function computes bootstrap-based correlation matrices for each group
        of models, compares them to control correlations, evaluates significance,
        and visualizes the parameter correlations as trajectories.

        Args:
            models_list (list): List of model groups (each entry contains path + metadata).
            fig (Figure): Figure object used for plotting.
            xpos_start_here (float): Initial x-position for plotting.
            ypos_start_here (float): Initial y-position for plotting.
            xl (str, optional): X-axis label.
            xticklabels (list, optional): Labels for x-axis ticks.
            plot_width_here (float, optional): Width of each plot.
            xticklabels_rotation (float, optional): Rotation for x-axis tick labels.
            show_title (bool, optional): Whether to display plot titles.
            show_scalebar (bool, optional): Whether to include a scale bar.
            show_ticks (bool, optional): Whether to display y-axis ticks.
            p_value_corr_acceptable (ndarray, optional): Matrix indicating which parameter
                correlations are acceptable for significance testing.

        Returns:
            tuple: parameter_corr_trajectory, parameter_corr_trajectory_std, parameter_pval_trajectory
                - parameter_corr_trajectory: mean correlation matrices across bootstraps
                - parameter_corr_trajectory_std: standard deviation of correlations
                - parameter_pval_trajectory: p-values comparing fish data to control
        """

        # Initialize grid position
        xpos = xpos_start_here
        ypos = ypos_start_here

        # Preallocate arrays for storing correlation statistics
        parameter_corr_trajectory = np.zeros(
            (len(models_list), len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))
        parameter_pval_trajectory = np.zeros(
            (len(models_list), len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))
        parameter_corr_trajectory_std = np.zeros_like(parameter_corr_trajectory)
        parameter_corr_trajectory_q25 = np.zeros_like(parameter_corr_trajectory)
        parameter_corr_trajectory_q75 = np.zeros_like(parameter_corr_trajectory)

        # All unique parameter-pair combinations (off-diagonal elements of correlation matrix)
        combination_list = list(itertools.combinations(range(len(ConfigurationDDM.parameter_list)), 2))

        # -------------------------------------------------------------------------
        # Iterate through each model group (e.g., experimental conditions or ages)
        # -------------------------------------------------------------------------
        for i_m, m in enumerate(models_list):
            path = m["path"]

            # ---------------------------------------------------------------------
            # Load fitted model results for all individuals in this group
            # ---------------------------------------------------------------------
            model_dict = {}
            for model_filepath in Path(path).glob('model_*_fit.hdf5'):
                model_filename = str(model_filepath.name)
                # Extract animal ID from filename and link to file path
                model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

            # Store parameters in array [parameters × individuals]
            model_array = np.zeros((len(ConfigurationDDM.parameter_list), len(model_dict.keys())))
            model_dict_group = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
            model_dict_group["id"] = []

            # Extract best-fit parameter values (lowest score) per animal
            i_m_ = 0
            for i_model, id_model in enumerate(model_dict.keys()):
                df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

                for i_p, p in enumerate(ConfigurationDDM.parameter_list):
                    model_array[i_p, i_m_] = df_model_fit[p["label"]].iloc[0]
                    model_dict_group[p["label"]].append(df_model_fit[p["label"]][0])
                model_dict_group["id"].append(id_model)
                i_m_ += 1

            # Convert extracted parameter values into DataFrame (indexed by fish ID)
            df_model_group_original = pd.DataFrame(model_dict_group)
            df_model_group_original.set_index('id', inplace=True)

            # Compute correlation matrix for original (non-bootstrapped) data
            relation_group_original = np.array(df_model_group_original.corr())

            # Bootstrap resampling of fish data
            df_model_group_list = BehavioralProcessing.randomly_sample_df(
                df=df_model_group_original,
                sample_number=number_bootstraps,
                sample_percentage_size=sample_percentage_size,
                with_replacement=True
            )
            relation_tensor_group = np.zeros(
                (number_bootstraps, len(ConfigurationDDM.parameter_list), len(ConfigurationDDM.parameter_list)))

            # Bootstrap resampling of combined control + fish data for null distribution
            df_model_group_combined_original = pd.concat((df_model_control_original, df_model_group_original))
            df_model_group_combined_list_0 = BehavioralProcessing.randomly_sample_df(
                df=df_model_group_combined_original,
                sample_number=number_bootstraps,
                sample_percentage_size=sample_percentage_size,
                with_replacement=True)
            df_model_group_combined_list_1 = BehavioralProcessing.randomly_sample_df(
                df=df_model_group_combined_original,
                sample_number=number_bootstraps,
                sample_percentage_size=sample_percentage_size,
                with_replacement=True)
            relation_tensor_group_combined_delta = np.zeros_like(relation_tensor_group)

            # ---------------------------------------------------------------------
            # Compute correlation matrices for each bootstrap sample
            # ---------------------------------------------------------------------
            for i_df in range(number_bootstraps):
                df_model_group = df_model_group_list[i_df]
                relation_tensor_group[i_df] = np.array(df_model_group.corr())

                # Correlation differences between two random combined samples
                df_model_combined_0 = df_model_group_combined_list_0[i_df]
                corr_combined_0 = np.array(df_model_combined_0.corr())
                df_model_combined_1 = df_model_group_combined_list_1[i_df]
                corr_combined_1 = np.array(df_model_combined_1.corr())
                relation_tensor_group_combined_delta[i_df] = np.abs(corr_combined_0 - corr_combined_1)

            # ---------------------------------------------------------------------
            # Significance testing: compare fish vs control correlations
            # ---------------------------------------------------------------------
            corr_delta_group = np.abs(relation_group_original - relation_control_original)
            p_value_group = np.mean(relation_tensor_group_combined_delta >= corr_delta_group, axis=0)

            # Store statistics for this model group
            # Store this group's mean, std, and 25th/75th-percentile correlation values, plus its p-value vs. control,
            # at index i_m of the trajectory arrays (one slot per group in the ordered/categorical list).
            parameter_corr_trajectory[i_m, :, :] = np.nanmean(relation_tensor_group, axis=0)
            parameter_corr_trajectory_std[i_m, :, :] = np.nanstd(relation_tensor_group, axis=0)
            parameter_corr_trajectory_q25[i_m, :, :] = np.nanquantile(relation_tensor_group, q=0.25, axis=0)
            parameter_corr_trajectory_q75[i_m, :, :] = np.nanquantile(relation_tensor_group, q=0.75, axis=0)
            parameter_pval_trajectory[i_m, :, :] = p_value_group
            parameter_pval_trajectory[i_m, :, :] = p_value_group

        # -------------------------------------------------------------------------
        # Visualization: plot correlation trajectories for each parameter pair
        # -------------------------------------------------------------------------
        x = np.arange(len(models_list))  # x-axis: model groups
        i_p1_old = 0
        not_shown_scalebar_yet = True

        # Loop over every unique parameter pair and draw its trajectory subplot.
        for i_p1, i_p2 in combination_list:
            p1 = ConfigurationDDM.parameter_list[i_p1]
            p2 = ConfigurationDDM.parameter_list[i_p2]

            # Reset row position when moving to a new first-parameter index
            if i_p1_old != i_p1:
                xpos = xpos_start_here
                ypos -= padding * 2 + plot_height
                i_p1_old = i_p1

            # Create subplot for this parameter pair
            plot_pp = fig.create_plot(
                plot_title=f"{p1['label_show']}-{p2['label_show']}" if show_title else None,
                xpos=xpos, ypos=ypos,
                plot_height=plot_height, plot_width=plot_width_here,
                xmin=-0.3, xmax=len(models_list) - 0.3, xticks=x, xl=xl, xticklabels=xticklabels,
                xticklabels_rotation=xticklabels_rotation,
                ymin=-1, ymax=1, yticks=[-1, 0, 1] if show_ticks else None,
                errorbar_area=False, hlines=[0]
            )
            xpos += padding * 3 + plot_width_small

            # Plot mean correlation ± std for this parameter pair
            plot_pp.draw_line(x, parameter_corr_trajectory[:, i_p1, i_p2], lc="k",
                              yerr=parameter_corr_trajectory_std[:, i_p1, i_p2])

            # Mark significant correlations (based on p-values and thresholds)
            for i_x_, x_ in enumerate(x):
                if p_value_corr_acceptable[i_p1, i_p2] \
                        and parameter_pval_trajectory[i_x_, i_p1, i_p2] < p_value_threshold \
                        and np.abs(parameter_corr_trajectory[i_x_, i_p1, i_p2]) - parameter_corr_trajectory_std[
                    i_x_, i_p1, i_p2] > high_corr_threshold:
                    plot_pp.draw_text(x_, 1, "*")

            # Add scale bar once (to rightmost subplot)
            if show_scalebar and not_shown_scalebar_yet:
                plot_pp.draw_line(np.ones(2) * x[-1] - 0.2, (-1, -0.8), lc="k", lw=0.8)
                plot_pp.draw_text(x[-1], -1, "Corrected\ncoherence\n0.2", textlabel_ha="left", textlabel_rotation=90)
                not_shown_scalebar_yet = False

        return parameter_corr_trajectory, parameter_corr_trajectory_std, parameter_pval_trajectory


    # compute correlation for all the datasets analysed in the manuscript
    # --- Run the trajectory function once per dataset grouping, placing each result set at a different x-offset ---
    xpos_start_here = xpos_start
    res_age = show_trajectory_correlation(models_in_age_list, fig, xpos_start_here, ypos_start_here, "Age (dpf)",
                                          ["5", "6", "7", "8", "9"], plot_width_small,
                                          0, False, False, True, p_value_corr_acceptable)

    xpos_start_here = xpos_start + padding_here * 0.3 + plot_width_small
    res_scn = show_trajectory_correlation(models_in_mutation_scn_list, fig, xpos_start_here, ypos_start_here, "scn1",
                                          ["+/+", "+/-"],
                                          plot_width_small * len(models_in_mutation_scn_list) / len(models_in_age_list),
                                          45, True, False, False, p_value_corr_acceptable)

    xpos_start_here = xpos_start + padding_here + plot_width_small
    res_disc = show_trajectory_correlation(models_in_mutation_disc_list, fig, xpos_start_here, ypos_start_here, "disc1",
                                           ["+/+", "+/-", "-/-"],
                                           plot_width_small * len(models_in_mutation_disc_list) / len(
                                               models_in_age_list),
                                           45, False, False, False, p_value_corr_acceptable)

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
fig.save(path_save / "figure_s7.pdf", open_file=False, tight=style.page_tight)
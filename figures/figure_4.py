import itertools

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.service.behavioral_processing import BehavioralProcessing
from rg_behavior_model.service.statistics_service import StatisticsService
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import ConfigurationExperiment
from rg_behavior_model.utils.constants import StimulusParameterLabel

# =============================================================================
# OVERVIEW
# =============================================================================
# This script compares behavioral data and model fits across several age groups.
# Main actions:
#  - Load data and synthetic model outputs for each age group (5dpf..9dpf).
#  - Produce summary plots:
#      * Loss reduction per fitted model (show training progression)
#      * Psychometric curves (data vs. model)
#      * Interbout interval (IBI) as a function of coherence (data vs. model)
#      * Distributions of fitted DDM parameters across animals/models and statistical tests
#  - Uses utilities from analysis.utils.figure_helper for plotting.
# Notes:
#  - The code assumes environment variables PATH_DIR and PATH_SAVE are set (dotenv).
# =============================================================================

# =============================================================================
# Environment / paths
# =============================================================================
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_data = path_dir / "age_analysis"
path_save = Path(env['PATH_SAVE'])

# =============================================================================
# Plot style and layout defaults
# =============================================================================
style = BehavioralModelStyle(plot_label_i=1)

plot_height = style.plot_height
plot_width = style.plot_width_small

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

padding = style.padding
padding_small = style.padding_small

palette = style.palette["default"]
number_bins_hist = 15  # number of bins used in histogram plots

# =============================================================================
# Age-specific dataset definitions
# =============================================================================
# Each dictionary contains:
#  - label_show: readable label used in plots
#  - path: base folder for model fit files and errors
#  - path_fish: aggregated HDF5 file with empirical fish data for that age
#  - path_simulation: aggregated HDF5 file with synthetic/model data for that age
models_in_age_list = [
    {"label_show": "5dpf",
     "path": f"{path_data}/5_dpf",
     "path_fish": f"{path_data}/5_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/5_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "6dpf",
     "path": f"{path_data}/6_dpf",
     "path_fish": f"{path_data}/6_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/6_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "7dpf",
     "path": f"{path_data}/7_dpf",
     "path_fish": f"{path_data}/7_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/7_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "8dpf",
     "path": f"{path_data}/8_dpf",
     "path_fish": f"{path_data}/8_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/8_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "9dpf",
     "path": f"{path_data}/9_dpf",
     "path_fish": f"{path_data}/9_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/9_dpf/data_synthetic_fish_all.hdf5"},
]

# =============================================================================
# Load data files for each age group (empirical + synthetic)
# =============================================================================
# We try to read the aggregated HDF5 files for both data and simulation.
# If files are missing or unsupported, a message is printed and that group
# will lack df_data/df_simulation keys.
for m in models_in_age_list:
    try:
        m["df_data"] = pd.read_hdf(m["path_fish"])
        m["df_simulation"] = pd.read_hdf(m["path_simulation"])
        # ensure simulation df has a flat integer index (useful later)
        m["df_simulation"].reset_index(inplace=True)
    except (KeyError, NotImplementedError):
        # KeyError might occur if HDF keys aren't present; NotImplementedError can be raised by some HDF backends
        print(f"No data for group {m['label_show']}")

# Time window filter used repeatedly later: restrict analysis to stimulus window
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

# =============================================================================
# Toggle which figures to show
# =============================================================================
# You can flip these booleans to enable/disable whole sections quickly.
show_loss_reduction = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_distribution_parameters = True

# =============================================================================
# Initialize figure container
# =============================================================================
fig = Figure()

# =============================================================================
# Plot A — Loss reduction over training for model fits
# =============================================================================
# Purpose:
#  - For each age group, read per-model 'error_fish_*.hdf5' files and plot the
#    initial and final loss for each model as a short line (iteration 0 -> iteration end).
#  - This gives a compact view of how much loss decreased during training/fitting.
if show_loss_reduction:
    ypos = ypos - padding_small
    for i_age, models_in_age in enumerate(models_in_age_list):
        loss_list = []
        loss_start_list = []
        loss_end_list = []

        # Look for files named error_fish_*.hdf5 inside each group's path
        for path_error in Path(models_in_age["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            # first value of the score series (starting loss)
            loss_start_list.append(df_error["score"][0])

            # last value of the score series (final loss)
            loss_end = df_error["score"][len(df_error)-1]
            loss_end_list.append(loss_end)
            loss_list.append(loss_end)

        # Plot title includes label and number of models found
        plot_title = f"{models_in_age['label_show']}\n" fr"n={len(loss_list)}"
        plot_loss = fig.create_plot(plot_label=style.get_plot_label() if i_age == 0 else None,
                                    plot_title=plot_title,
                                    xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                    ymin=0, ymax=20, yticks=[0, 10, 20] if i_age == 0 else None,
                                    yl="Loss" if i_age == 0 else None,
                                    xl="Iteration", xmin=0.5, xmax=2.5, xticks=[1, 2],
                                    xticklabels=["0", "1500"])
        # Increment horizontal position for the next small panel
        xpos += plot_width + padding_small
        ypos = ypos

        # Draw a thin line connecting start and end loss for each model (and a small scatter on endpoints)
        for i_loss in range(len(loss_list)):
            loss_start = loss_start_list[i_loss]
            loss_end = loss_end_list[i_loss]
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

    # Reset to left margin and move vertically down after the row of small panels
    xpos = xpos_start
    ypos = ypos - padding - plot_height

# =============================================================================
# Plot B — Psychometric curve: empirical data vs. synthetic model
# =============================================================================
# Purpose:
#  - For each age group plot percentage correct vs coherence for both empirical
#    data and the synthetic data produced by the fitted models.
# Notes:
#  - Dataframes might use 'fish_ID' as the column for experiment ID — code normalizes
#    this to 'experiment_ID' when needed for downstream functions.
if show_psychometric_curve:
    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        # apply the time window filter and keep only relevant coherence values
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]
        # Some datasets store the subject id under 'fish_ID' — unify to 'experiment_ID'
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            # if fish_ID missing, assume it's already experiment_ID or indexed appropriately
            pass

        # Compute mean % correct across fish for each coherence (returns means and std)
        parameter_list_data, correct_bout_list_data, std_correct_bout_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE.value)
        # number of individuals (for reference / potential normalization)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        # Load & prepare simulation (model) outputs for this age group
        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, correct_bout_list_sim, std_correct_bout_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(df_simulation_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE.value)

        # Ensure coherence lists are integer arrays for plotting
        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # Create a small panel per age group, optionally showing y-axis for the first panel
        plot_0 = fig.create_plot(plot_label=style.get_plot_label() if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                                 xticks=None,
                                 yl="Percentage\ncorrect swims (%)" if i_m == 0 else None, ymin=45, ymax=100, yticks=[50, 100] if i_m == 0 else None, hlines=[50])

        # Draw empirical data (solid) and model (dashed). Errorbars drawn from std lists.
        plot_0.draw_line(x=parameter_list_data, y=correct_bout_list_data*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data)*100,  # / np.sqrt(number_individuals),
                         lc="k", lw=1)
        plot_0.draw_line(x=parameter_list_sim, y=correct_bout_list_sim*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim)*100,  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2))

        # spacing: small pad between intermediate plots, larger pad to finalize the row
        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_small
        xpos = xpos + pad + plot_width

    # reset position for the next row
    xpos = xpos_start
    ypos = ypos - padding - plot_height

# =============================================================================
# Plot C — Coherence vs Interbout Interval (IBI), data vs. model
# =============================================================================
# Purpose:
#  - Compute and show average IBI as a function of coherence for data and synthetic
#    model outputs. Performs basic filtering to remove 'straight' bouts and extreme quantiles.
if show_coherence_vs_interbout_interval:
    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        # time window + coherence filter
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]

        # Report fraction of trials removed when excluding bouts flagged as straight (CorrectBoutColumn == -1)
        original_len = len(df_data_filtered)
        df_data_filtered = df_data_filtered[df_data_filtered[ConfigurationExperiment.CorrectBoutColumn] != -1]
        filtered_len = len(df_data_filtered)
        print(f"EXCLUDE STRAIGHT BOUT |at {m['label_show']} {(original_len - filtered_len)/original_len*100:.03f}% dropped")

        # Remove extreme IBI values by keeping central 90% (5th to 95th percentiles)
        ibi_quantiles = np.quantile(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn], [0.05, 0.95])
        df_data_filtered = df_data_filtered[np.logical_and(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] > ibi_quantiles[0],
                                                           df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] < ibi_quantiles[1])]
        # Normalize fish ID column name if necessary
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass

        # Compute per-coherence statistics (mean IBI and std across fish)
        parameter_list_data, interbout_interval_list_data, std_interbout_interval_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE.value, column_name=ConfigurationExperiment.ResponseTimeColumn)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        # Prepare simulation dataset (model) for the same analysis
        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]

        number_models = len(df_simulation_filtered["fish_ID"].unique())

        parameter_list_sim, interbout_interval_list_sim, std_interbout_interval_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
                df_simulation_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE.value, column_name=ConfigurationExperiment.ResponseTimeColumn)
        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # Create plot panel(s) across age groups; show y-axis label only for first panel
        plot_0 = fig.create_plot(plot_label=style.get_plot_label() if i_m == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width,
                                 errorbar_area=True,
                                 xl=ConfigurationExperiment.coherence_label,
                                 xmin=min(ConfigurationExperiment.coherence_list),
                                 xmax=max(ConfigurationExperiment.coherence_list),
                                 xticks=[int(p) for p in ConfigurationExperiment.coherence_list],
                                 yl="Interbout\ninterval (s)" if i_m == 0 else None,
                                 ymin=0, ymax=3, yticks=[0, 1.50, 3] if i_m == 0 else None)

        # Draw data (solid) and simulation (dashed) with errorbars
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),  # / np.sqrt(number_individuals),
                         lc="k", lw=1, label=f"data" if i_m == 0 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2), label=f"simulation" if i_m == 0 else None)

        # spacing between small panels
        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_small
        xpos = xpos + pad + plot_width

    xpos = xpos_start
    ypos = ypos - padding - plot_height

# =============================================================================
# Plot D — Distribution of DDM parameter estimates and statistical tests
# =============================================================================
# Purpose:
#  - For each age group, gather parameter medians across fitted models (model_*_fit.hdf5)
#  - Build histograms of medians (to visualize cross-individual variability)
#  - Perform bootstrap/resampling to estimate median quantiles per age group
#  - Perform pairwise permutation-like tests (resampling from combined sample) to flag
#    significant differences between groups for each parameter.
# Notes & variables:
#  - from_best_model = True -> for each fitted fish/model, only the best-fit entry (min score) is kept.
#  - number_resampling_bootstrapping and threshold_p_value_significant are read from StatisticsService configuration.
if show_distribution_parameters:
    # small-panel sizes and padding for this section
    plot_height_here = style.plot_height_small
    padding_here = style.padding_in_plot_small

    from_best_model = True

    number_resampling_bootstrapping = StatisticsService.number_resampling_bootstrapping
    threshold_p_value_significant = StatisticsService.threshold_p_value_significant

    # Initialize containers to collect distributions and raw medians
    distribution_trajectory_dict = {p["label"]: np.zeros((number_bins_hist, len(models_in_age_list))) for p in ConfigurationDDM.parameter_list}
    raw_data_dict_per_fish = {p["label"]: {i_age: {} for i_age in range(len(models_in_age_list))} for p in ConfigurationDDM.parameter_list}
    raw_data_dict = {p["label"]: {i_age: [] for i_age in range(len(models_in_age_list))} for p in ConfigurationDDM.parameter_list}
    median_groups = {p["label"]: np.zeros(len(models_in_age_list)) for p in ConfigurationDDM.parameter_list}
    quantiles_groups = {p["label"]: np.zeros((len(models_in_age_list), 3)) for p in ConfigurationDDM.parameter_list}

    # Loop through age groups, read models, and compute per-model medians
    for i_age, models_in_age in enumerate(models_in_age_list):
        model_dict = {}
        path_dir = Path(models_in_age["path"])

        # Collect model fit files named model_*_fit.hdf5; use filename token to index models
        for model_filepath in path_dir.glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

        fish_list = np.arange(len(model_dict.keys()))

        # These will hold medians and full distributions per parameter per model
        model_parameter_median_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_median_dict["score"] = {}
        model_parameter_dict["score"] = {}
        # matrix to collect medians: row 0 -> score, rows 1.. -> parameters, cols->models (fish_list)
        model_parameter_median_array = np.zeros((len(ConfigurationDDM.parameter_list)+1, len(fish_list)))

        reset_list = []

        # Iterate all model ids (keys from model_dict)
        for i_model, id_model in enumerate(model_dict.keys()):
            df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])

            id_fish = i_model
            # If requested, keep only the best-fitting entry (lowest score) per model
            if from_best_model:
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit_list = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

            # Score medians and distributions
            model_parameter_median_dict["score"][id_fish] = np.median(df_model_fit_list["score"])
            model_parameter_dict["score"][id_fish] = np.array(df_model_fit_list["score"])
            model_parameter_median_array[0, i_model] = np.median(df_model_fit_list["score"])

            # For each DDM parameter, store medians & full arrays
            for i_p, p in enumerate(ConfigurationDDM.parameter_list):
                p_median = np.median(df_model_fit_list[p["label"]])
                model_parameter_median_dict[p["label"]][id_fish] = p_median
                model_parameter_dict[p["label"]][id_fish] = np.array(df_model_fit_list[p["label"]])
                model_parameter_median_array[i_p+1, i_model] = p_median

                # Save per-model medians to raw_data_dict_per_fish and per-age raw_data_dict
                if id_model not in raw_data_dict_per_fish[p["label"]][i_age].keys():
                    raw_data_dict_per_fish[p["label"]][i_age][id_model] = [p_median]
                else:
                    raw_data_dict_per_fish[p["label"]][i_age][id_model].append(p_median)

                raw_data_dict[p["label"]][i_age].append(p_median)

                if p["label_show"] == "reset":
                    reset_list.append(p_median)

        # For each parameter: compute median across models in this age group and bootstrap quantiles
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            median_groups[p["label"]][i_age] = np.median(raw_data_dict[p["label"]][i_age])

            # Draw bootstrap samples of the per-model medians to compute empirical quantiles
            model_parameter_sampling_list = StatisticsService.sample_random(
                array=model_parameter_median_array[i_p + 1, :], sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)

            median_list = np.array([np.median(m) for m in model_parameter_sampling_list])
            quantiles_groups[p["label"]][i_age, :] = np.quantile(median_list, [0.05, 0.5, 0.95])
            print(f"group {models_in_age['label_show']} | {p['label_show']}: {quantiles_groups[p['label']][i_age, 1]:.05f} [{quantiles_groups[p['label']][i_age, 0]:.05f}, {quantiles_groups[p['label']][i_age, 2]:.05f}]")

        # Build histograms (density) of medians for the score and each parameter
        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[ConfigurationDDM.score_config["label"]], bin_model_parameter_median_dict[ConfigurationDDM.score_config["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[0, :], center_bin=True,  hist_range=[ConfigurationDDM.score_config["min"], ConfigurationDDM.score_config["max"]],
                bins=number_bins_hist,
                density=True
            )
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[p["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :], center_bin=True,  hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,
                density=True
            )
            # Collect histogram values across ages (columns = age groups)
            distribution_trajectory_dict[p["label"]][:, i_age] = hist_model_parameter_median_dict[p["label"]]

    models_in_age["reset_list"] = reset_list

    # =============================================================================
    # Plot D1 — Histogram panels for each parameter x age group (small panels grid)
    # =============================================================================
    # Draw a grid: for each age (columns) and parameter (rows), show the distribution of fitted medians.
    for i_age in range(len(models_in_age_list)):
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            plot_n = fig.create_plot(plot_label=style.get_plot_label() if i_p == 0 and i_age == 0 else None, xpos=xpos, ypos=ypos,
                                     plot_height=plot_height_here, plot_width=plot_width,
                                     yl="Percentage fish (%)" if i_p == 0 and i_age == 0 else None, ymin=0, ymax=50, yticks=[0, 50] if i_p == 0 and i_age == 0 else None,
                                     xmin=p["min"], xmax=p["max"],
                                     vlines=[p["mean"]] if p["mean"] == 0 else [])

            # Draw the histogram line for this age & parameter.
            # Visual attributes (dashes/alpha) taken from models_in_age_list entries if provided there.
            plot_n.draw_line(bin_model_parameter_median_dict[p["label"]], distribution_trajectory_dict[p["label"]][:, i_age] * 100,
                             lc=palette[i_p], label=models_in_age_list[i_age]["label_show"] if i_p == len(ConfigurationDDM.parameter_list)-1 else None)

            xpos = xpos + padding_small + plot_width

        # move to next row (next set of parameter panels)
        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here

    xpos = xpos_start
    ypos = ypos - (padding-padding_here)

    # =============================================================================
    # Plot D2 — Summary panels showing median & quantile bars per group and tests
    # =============================================================================
    # This block draws a horizontal layout where each parameter gets its own
    # subplot showing quantile lines per age group (used for comparison / annotation).
    plot_list = []
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        plot_n = fig.create_plot(plot_label=style.get_plot_label() if i_p==0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width, errorbar_area=False,
                                 ymin=0, ymax=len(models_in_age_list) + 1,
                                 yticks=np.arange(len(models_in_age_list), 0, -1) if i_p == 0 else None,
                                 yticklabels=[g["label_show"] for g in models_in_age_list] if i_p == 0 else None,
                                 xl=p['label_show'].capitalize(), xmin=p["min"], xmax=p["max"],
                                 xticks=[p["min"], p["mean"], p["max"]], vlines=[p["mean"]] if p["mean"] == 0 else [])
        xpos = xpos + padding_small + plot_width
        # Draw the quantile lines (0.05, 0.5, 0.95) horizontally for each group
        for i_group, models_in_group in enumerate(models_in_age_list):
            plot_n.draw_line(quantiles_groups[p["label"]][i_group, :],
                             np.ones(quantiles_groups[p["label"]].shape[1]) * (len(models_in_age_list) - i_group),
                             lc=palette[i_p])
        plot_list.append(plot_n)

    # =============================================================================
    # Pairwise comparisons: resampling-based test (approximate permutation)
    # =============================================================================
    # For each parameter, compute pairwise comparisons between age groups using resampling:
    #  - Combine the two groups' median samples into a single array.
    #  - Draw two bootstrap samples of that combined array (with replacement).
    #  - Compute distribution of absolute differences in sample medians.
    #  - p-value = fraction of bootstrap deltas >= observed median difference.
    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_age_list)), 2))
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        parameter_range = p["max"] - p["min"]
        x_sig = p["max"] - parameter_range / 2 + parameter_range / 10
        is_sig = False
        print(f"{p['label_show']}")
        for pairs_to_compare in pairs_to_compare_list:
            # observed median difference between the two groups
            median_delta = np.abs(median_groups[p["label"]][pairs_to_compare[0]] - median_groups[p["label"]][pairs_to_compare[1]])

            # combine data from both groups as a null distribution for resampling
            combined_array = np.concatenate((np.array(raw_data_dict[p["label"]][pairs_to_compare[0]]),
                                             np.array(raw_data_dict[p["label"]][pairs_to_compare[1]])))
            # generate two independent resampled sets from combined array
            model_parameter_sampling_control_0 = StatisticsService.sample_random(
                array=combined_array, sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)
            model_parameter_sampling_control_1 = StatisticsService.sample_random(
                array=combined_array, sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)

            median_control_0 = np.array([np.median(m) for m in model_parameter_sampling_control_0])
            median_control_1 = np.array([np.median(m) for m in model_parameter_sampling_control_1])
            median_control_delta = np.abs(median_control_0 - median_control_1)

            # approximate p-value: fraction of resampled deltas >= observed delta
            p_value = np.mean(median_control_delta >= median_delta)
            print(f"{models_in_age_list[pairs_to_compare[0]]['label_show']} vs {models_in_age_list[pairs_to_compare[1]]['label_show']} | p = {p_value}")

            plot_n = plot_list[i_p]
            # annotate figure with a small connecting line if significant
            if p_value < threshold_p_value_significant:
                is_sig = True
                x_sig_array = np.ones(2) * x_sig
                y_sig_array = np.array(
                    (len(models_in_age_list) - pairs_to_compare[0], len(models_in_age_list) - pairs_to_compare[1]))
                plot_n.draw_line(x_sig_array, y_sig_array, lc="k")

                # adjust x position for next potential significance marker to avoid overlap
                x_sig += parameter_range / 10

        # If any significant pair was found, show a star annotation near the center
        if is_sig:
            plot_n.draw_text(x=x_sig + parameter_range / 6, y=np.ceil(len(models_in_age_list) / 2), text="*",
                             textlabel_rotation=-90)
    xpos = xpos_start
    ypos = ypos - (plot_height + padding)

# =============================================================================
# Save final figure
# =============================================================================
fig.save(path_save / f"figure_4.pdf", open_file=False, tight=style.page_tight)

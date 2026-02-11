"""
Overview:
This script generates and saves a multi-panel figure (figure_3.pdf) that summarizes behavioral model results
from experimental fish datasets. It integrates raw experimental data, fitted model outputs, and statistical analyses
to visualize several aspects of fish decision-making behavior, including:

1. **Loss reduction** during model training (convergence of optimization).
2. **Psychometric curves** showing the percentage of correct responses as a function of stimulus coherence.
3. **Coherence vs. interbout interval** relationships.
4. **Reaction time (RT) distributions** for correct and incorrect responses across stimulus strengths.
5. **Individual parameter estimations** for each fish from fitted decision-making models.
6. **Parameter distributions** across the fish population.

The script relies on configurations, helper classes, and services from the `rg_behavior_model` package
to load data, preprocess behavioral responses, and apply consistent figure styling.
"""

import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import dotenv_values

# Local imports for plotting and analysis
from figures.style import BehavioralModelStyle
from service.behavioral_processing import BehavioralProcessing
from service.figure_helper import Figure
from service.statistics_service import StatisticsService
from utils.configuration_ddm import ConfigurationDDM
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel

# =====================================================================
# Load environment variables to locate data and output directories
# =====================================================================
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_data = path_dir / "base_dataset_5dpfWT"  # directory containing behavioral datasets
path_data_repeats = path_dir / "base_dataset_5dpfWT_repeats"  # containing datasets for repeatability test
path_save = Path(env['PATH_SAVE'])     # directory to save output figures

# =====================================================================
# Initialize figure style (layout, dimensions, palette, etc.)
# =====================================================================
style = BehavioralModelStyle(plot_label_i=1)

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5
plot_width = style.plot_width
plot_width_small = style.plot_width_small

padding = style.padding
padding_short = style.padding / 2
padding_in_plot_vertical = plot_height_small
padding_in_plot_horizontal = style.padding / 3

plot_height_row = plot_height_small * 2 + padding_in_plot_vertical

color_line = "gray"
lw = 1  # default line width

# =====================================================================
# Toggles to enable/disable different parts of the figure
# =====================================================================
show_loss_reduction = True
show_repeatability = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_rt_distributions = True
show_individual_estimations = True
show_distribution_parameters = True

# =====================================================================
# Experimental configuration parameters
# =====================================================================
analysed_parameter_list = ConfigurationExperiment.coherence_list
time_start_stimulus = ConfigurationExperiment.time_start_stimulus
time_end_stimulus = ConfigurationExperiment.time_end_stimulus
time_experimental_trial = ConfigurationExperiment.time_experimental_trial

# The parameter used for analysis (default: stimulus coherence)
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_label = ConfigurationExperiment.coherence_label

# Query filter to extract valid trials within stimulus time window
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

# =====================================================================
# Plotting parameters
# =====================================================================
number_bins_hist = 15  # used for histograms

# Dictionary to store datasets (raw + synthetic fits) per fish
df_dict = {}
fish_to_include_list = ConfigurationExperiment.example_fish_list

# Plot styling configurations for raw data vs model fit
config_list = [
    {"label": "data", "line_dashes": None, "alpha": 0.5, "color": None},
    {"label": "fit", "line_dashes": (2, 4), "alpha": 1, "color": "k"}
]

# Load datasets for each example fish
for i_fish, fish in enumerate(fish_to_include_list):
    df_data = pd.read_hdf(path_data / f"data_fish_{fish}.hdf5")
    # Load one corresponding synthetic dataset (fit)
    for path_fit in path_data.glob(f"data_synthetic_fish_{fish}_*.hdf5"):
        df_fit = pd.read_hdf(path_fit)
        break
    df_dict[fish] = {
        "fit": df_fit,
        "data": df_data,
        "color": style.palette["fish_code"][i_fish]
    }

# =====================================================================
# Create base figure canvas
# =====================================================================
fig = Figure()

# =====================================================================
# PLOT 1: Loss reduction curves
# =====================================================================
if show_loss_reduction:
    ypos = ypos - padding_short

    # Create loss vs iteration panel
    plot_loss = fig.create_plot(plot_label=style.get_plot_label(),
                                xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                ymin=0, ymax=20, yticks=[0, 10, 20],
                                yl="Loss",
                                xl="Iteration", xmin=0.5, xmax=2.5, xticks=[1, 2],
                                xticklabels=["0", "1500"])
    # For each fish, extract error/loss trajectories and plot
    for i_fish_id, fish_id in enumerate(fish_to_include_list):
        loss_array = np.zeros(2)
        for path_error in path_data.glob(f"error_fish_{fish_id}_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_array[0] = df_error["score"][0]                  # initial loss
            loss_array[1] = df_error["score"][len(df_error)-1]    # final loss

            # Plot line + scatter points for each fish
            plot_loss.draw_line((1, 2), loss_array, lc=df_dict[fish_id]["color"])
            plot_loss.draw_scatter((1, 2), loss_array, ec=df_dict[fish_id]["color"], pc=df_dict[fish_id]["color"], label=f"fish {i_fish_id}")
    ypos = ypos - padding - plot_height
    xpos = xpos_start

# =====================================================================
# PLOT 2: Testing repeatability of latent variable estimation
# =====================================================================
if show_repeatability:
    # Configuration for repeatability analysis
    test_id_list = ["205", "506", "201"]
    plot_width_here = 0.7  # style.plot_width_small
    padding_here = style.padding_small
    palette_here = style.palette["default"]

    # Iterate over repeatability test runs
    for i_test_id, test_id in enumerate(test_id_list):
        # Plot each parameter trajectory
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):

            # Create subplot per parameter
            plot_p = fig.create_plot(
                plot_label=style.get_plot_label() if i_p == 0 and i_test_id == 0 else None,
                plot_title=f"{p['label_show'].capitalize()}" if i_test_id == 0 else None,
                xpos=xpos, ypos=ypos,
                plot_height=plot_height, plot_width=plot_width_here,
                errorbar_area=False,
                ymin=p["min"], ymax=p["max"],
                yticks=[p["min"], p["mean"], p["max"]],
                yl=f"Fish {i_test_id+1}" if i_p == 0 else None, xl="Iteration" if i_test_id == len(test_id_list)-1 else None,
                xmin=0.5, xmax=2.5, xticks=[1, 2] if i_test_id == len(test_id_list)-1 else None,
                xticklabels=["0", "1500"] if i_test_id == len(test_id_list)-1 else None,
            )
            xpos += (plot_height + padding_short)

            # Collect end-of-fit parameter values
            p_fit_end_list = []
            for path_fit in path_data_repeats.glob(f"error_fish_{test_id}_*_fit.hdf5"):
                df_fit = pd.read_hdf(path_fit)
                p_fit_start = df_fit[f"{p['label']}_value"][0]
                p_fit_end = df_fit[f"{p['label']}_value"][len(df_fit) - 1]
                p_fit_end_list.append(p_fit_end)

                # Plot line connecting start → end values for each fit
                plot_p.draw_line((1, 2), (p_fit_start, p_fit_end),
                                 lc=palette_here[i_p], lw=0.05, alpha=0.5)
                plot_p.draw_scatter((1, 2), (p_fit_start, p_fit_end),
                                    pc=palette_here[i_p], ec=palette_here[i_p])

            # Overlay median final value
            plot_p.draw_scatter([2], [np.median(p_fit_end_list)], pc="k")

        xpos = xpos_start
        ypos -= plot_height + padding


# =====================================================================
# PLOT 2: Psychometric curve (performance vs coherence)
# =====================================================================
if show_psychometric_curve:
    show_label = True

    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=plot_height_row,
                             plot_width=plot_width,
                             xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
                             xticks=None,
                             yl="Percentage\ncorrect swims (%)",
                             ymin=0, ymax=100,
                             yticks=[0, 50, 100], hlines=[0.5])

    i_fish = 1
    for k_fish, df_fish_dict in df_dict.items():
        for config in config_list:
            # Filter trials for valid times and parameter values
            df = df_fish_dict[config["label"]]
            df_filtered = df.query(query_time)
            df_filtered = df_filtered[df_filtered[analysed_parameter].isin(analysed_parameter_list)]

            # Compute percentage correct per parameter value
            p_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(df_filtered, analysed_parameter=analysed_parameter)
            correct_bout_list *= 100
            p_list = np.array([int(p) for p in p_list])

            # Plot psychometric curve
            plot_0.draw_line(x=p_list, y=correct_bout_list, lc=df_fish_dict["color"], lw=lw, alpha=config["alpha"], line_dashes=config["line_dashes"])
            if show_label and config["label"] == "data":
                plot_0.draw_text(max(p_list) + 0.1, correct_bout_list[-1], f"fish {i_fish}",
                                 textlabel_rotation='horizontal', textlabel_ha='left', textcolor=df_fish_dict["color"])
                i_fish += 1

    xpos = xpos_start
    ypos = ypos - padding - plot_height

# =====================================================================
# PLOT 3: Interbout interval vs coherence
# =====================================================================
if show_coherence_vs_interbout_interval:
    show_label = True

    plot_0 = fig.create_plot(xpos=xpos, ypos=ypos,
                             plot_height=plot_height_row,
                             plot_width=plot_width,
                             errorbar_area=True,
                             xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
                             xticks=[int(p) for p in analysed_parameter_list], yl="Interbout interval (s)",
                             ymin=0, ymax=2,
                             yticks=[0, 1, 2])

    i_fish = 1
    for k_fish, df_fish_dict in df_dict.items():
        for config in config_list:
            # Filter trials
            df = df_fish_dict[config["label"]]
            df_filtered = df.query(query_time)
            df_filtered = df_filtered[df_filtered[analysed_parameter].isin(analysed_parameter_list)]

            # Compute interbout intervals per coherence
            p_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
                df_filtered,
                analysed_parameter=analysed_parameter,
                column_name=ConfigurationExperiment.ResponseTimeColumn
            )
            p_list = np.array([int(p) for p in p_list])

            # Plot line per fish
            plot_0.draw_line(x=p_list, y=correct_bout_list, lc=df_fish_dict["color"], lw=lw, alpha=config["alpha"], line_dashes=config["line_dashes"])
            if show_label and config["label"] == "data":
                plot_0.draw_text(max(p_list) + 0.1, correct_bout_list[-1], f"fish {i_fish}",
                                 textlabel_rotation='horizontal', textlabel_ha='left', textcolor=df_fish_dict["color"])
                i_fish += 1

    xpos = xpos_start
    ypos = ypos - plot_height * len(df_dict) - padding

# =====================================================================
# PLOT 4: Reaction time (RT) distributions
# =====================================================================
if show_rt_distributions:
    # Parameters for reaction time histograms
    padding_here = plot_height_small * 2
    palette = style.palette["stimulus"]
    x_limits = [0, 2]   # RT window
    y_limits = [0, 0.2] # density range

    plot_height_here = plot_height_small * 3
    plot_width_here = 0.7

    plot_section = {"corr": {}, "err": {}}

    # Remove pooled dataset (all fish merged), keep only per-fish data
    df_dict.pop(ConfigurationExperiment.all_fish_label, None)

    # Loop over fish and stimulus coherence values
    for i_k, k in enumerate(df_dict.keys()):
        for i_param, parameter in enumerate(analysed_parameter_list):
            # Create RT plots for correct and incorrect responses
            line_dashes = None
            plot_title = None
            if i_k == 0:
                if i_param == 0:
                    plot_title = f"Coh={parameter}%"
                else:
                    plot_title = f"{parameter}%"
            plot_section["corr"][i_param] = fig.create_plot(plot_label=style.get_plot_label() if i_k == 0 and i_param == 0 else None,
                                                            plot_title=plot_title,
                                                            xpos=xpos + i_param * (plot_width_here + padding_in_plot_horizontal),
                                                            ypos=ypos + plot_height_small + padding_in_plot_vertical,
                                                            plot_height=plot_height_here,
                                                            plot_width=plot_width_here,
                                                            xmin=x_limits[0], xmax=x_limits[-1],
                                                            xticks=None, yticks=None,
                                                            ymin=-y_limits[-1], ymax=y_limits[-1],
                                                            hlines=[0])
            plot_section["err"][i_param] = plot_section["corr"][i_param]

            # Add scalebars on final subplot
            if i_param == len(analysed_parameter_list) - 1 and i_k == len(df_dict)-1:
                y_location_scalebar = y_limits[-1] / 6
                x_location_scalebar = x_limits[-1] / 6
                plot_section["corr"][i_param].draw_line((1.7, 1.7), (y_location_scalebar, y_location_scalebar + 0.1), lc="k")
                plot_section["corr"][i_param].draw_text(2, y_location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

                plot_section["corr"][i_param].draw_line((x_location_scalebar, x_location_scalebar + 0.5), (-y_location_scalebar, -y_location_scalebar), lc="k")
                plot_section["corr"][i_param].draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                        textlabel_rotation='horizontal', textlabel_ha='left', textlabel_va="bottom")

            for config in config_list:
                df = df_dict[k][config["label"]]
                plot_section_corr = plot_section["corr"][i_param]
                plot_section_err = plot_section["err"][i_param]

                # Filter dataset for this coherence level
                df_filtered = df[df[analysed_parameter] == parameter]
                df_filtered = df_filtered.query(query_time)

                # Compute total duration (needed for normalization)
                duration = np.sum(
                    BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=time_end_stimulus-time_start_stimulus)
                )

                # Extract RTs for correct and incorrect responses
                data_corr = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1][ConfigurationExperiment.ResponseTimeColumn]
                data_err = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0][ConfigurationExperiment.ResponseTimeColumn]

                # Build histograms for correct RTs
                data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                       bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                       duration=duration,
                                                                                       center_bin=True)
                index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
                data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
                data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

                # Build histograms for incorrect RTs
                data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                     bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                     duration=duration,
                                                                                     center_bin=True)
                index_in_limits = np.argwhere(
                        np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
                data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
                data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

                # Assign line colors depending on whether this is raw data or model fit
                if config["color"] is None:
                    lc_correct = style.palette["correct_incorrect"][0]
                    lc_incorrect = style.palette["correct_incorrect"][1]
                    alpha_correct = 1
                    alpha_incorrect = 1
                else:
                    lc_correct = config["color"]
                    lc_incorrect = config["color"]
                    alpha_correct = 0.7
                    alpha_incorrect = 0.3

                # Plot RT histograms (positive for correct, negative for incorrect)
                plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=lc_correct,
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_correct)
                plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=lc_incorrect,
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_incorrect)

        ypos = ypos - (padding_here + plot_height_small)

    xpos = xpos_start
    ypos = ypos - padding


# ===============================================================
# Plot 5: Individual parameter estimations and parameter distributions
# ===============================================================
if show_individual_estimations or show_distribution_parameters:
    from_best_model = True  # Flag: use only best scoring model for each fish
    plot_height_here = style.plot_height
    plot_width_here = plot_width_small
    padding_here = style.padding
    palette = style.palette["default"]
    number_resampling = 10000  # Number of resamples for bootstrapping histograms

    # Initialize dictionaries to store distributions and raw data
    distribution_trajectory_dict = {p["label"]: np.zeros(number_bins_hist) for p in ConfigurationDDM.parameter_list}
    raw_data_dict_per_fish = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
    raw_data_dict = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    model_dict = {}
    n_models = 0

    # Collect all model fit files
    for model_filepath in path_data.glob('model_*_fit.hdf5'):
        model_filename = str(model_filepath.name)
        # Key by the model ID extracted from filename
        model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

    fish_list = np.arange(len(model_dict.keys()))

    # Initialize dictionaries for median and full parameter values
    model_parameter_median_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
    model_parameter_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
    model_parameter_median_dict["score"] = {}
    model_parameter_dict["score"] = {}
    model_parameter_median_array = np.zeros((len(ConfigurationDDM.parameter_list)+1, len(fish_list)))

    # Loop over models to extract median parameter values
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        id_fish = i_model

        if from_best_model:
            best_score = np.min(df_model_fit_list['score'])
            df_model_fit_list = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

        # Store score statistics
        model_parameter_median_dict["score"][id_fish] = np.median(df_model_fit_list["score"])
        model_parameter_dict["score"][id_fish] = np.array(df_model_fit_list["score"])
        model_parameter_median_array[0, i_model] = np.median(df_model_fit_list["score"])

        # Loop over all model parameters
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            p_median = np.median(df_model_fit_list[p["label"]])
            model_parameter_median_dict[p["label"]][id_fish] = p_median
            model_parameter_dict[p["label"]][id_fish] = np.array(df_model_fit_list[p["label"]])
            model_parameter_median_array[i_p+1, i_model] = p_median

            # Organize raw data per fish
            if id_model not in raw_data_dict_per_fish[p["label"]].keys():
                raw_data_dict_per_fish[p["label"]][id_model] = [p_median]
            else:
                raw_data_dict_per_fish[p["label"]][id_model].append(p_median)

            raw_data_dict[p["label"]].append(p_median)

    # -----------------------------------------------------------
    # Section: Individual estimations (scatter plots per fish)
    # -----------------------------------------------------------
    if show_individual_estimations:
        plot_height_here = plot_height * 3
        number_fish = model_parameter_median_array.shape[1]
        fish_id_array = np.flip(np.arange(number_fish))  # Y-axis: fish IDs in reverse order

        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            # Determine vertical lines to mark mean or relevant values
            vlines = [p["mean"]] if p["mean"] == 0 else [0]
            # if p["relevant_values"] is not None:
            #     vlines.extend(p["relevant_values"])

            # Create individual scatter plot for this parameter
            plot_individual = fig.create_plot(
                plot_label=style.get_plot_label() if i_p == 0 else None,
                xpos=xpos,
                ypos=ypos,
                plot_height=plot_height_here,
                plot_width=plot_width_here,
                yl="Fish ID" if i_p == 0 else None,
                ymin=0-0.5,
                ymax=number_fish-0.5,
                yticks=None,
                xmin=p["min"]-(p["max"]-p["min"])/20,
                xmax=p["max"]+(p["max"]-p["min"])/20,
                xticks=[p["min"], p["mean"], p["max"]],
                xl=p['label_show'].capitalize(),
                vlines=vlines
            )

            # Increment X position for next parameter
            xpos += plot_width_here + padding_short

            # Draw scatter for median parameter values
            plot_individual.draw_scatter(model_parameter_median_array[i_p + 1, :], fish_id_array, pc=palette[i_p], elw=0)

        # Reset X position and update Y position for next row of plots
        xpos = xpos_start
        ypos -= plot_height_here + padding_here

    # -----------------------------------------------------------
    # Section: Parameter distributions (histograms)
    # -----------------------------------------------------------
    if show_distribution_parameters:
        plot_height_here = style.plot_height_small

        # Compute histograms for score
        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[ConfigurationDDM.score_config["label"]], bin_model_parameter_median_dict[ConfigurationDDM.score_config["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[0, :],
                center_bin=True,
                hist_range=[ConfigurationDDM.score_config["min"], ConfigurationDDM.score_config["max"]],
                bins=number_bins_hist,
                density=True
            )

        # Loop over all model parameters and compute histograms
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[p["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :],
                center_bin=True,
                hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,
                density=True
            )
            distribution_trajectory_dict[p["label"]] = hist_model_parameter_median_dict[p["label"]]

            # Print coverage of parameter space
            print(f"{p['label_show']} spans {(np.max(model_parameter_median_array[i_p + 1, :])-np.min(model_parameter_median_array[i_p + 1, :]))/(p['max']-p['min'])*100}% of the parameter space")
            if p["label"] == "leak":
                around_optimal_integration = hist_model_parameter_median_dict[p["label"]][9:18]
                print(f"LEAK | {np.sum(around_optimal_integration)*100}% is around optimal")
                optimal_integration = hist_model_parameter_median_dict[p["label"]][11:16]
                print(f"LEAK | {np.sum(optimal_integration)*100}% is optimal")

        # Plot histograms for each parameter
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            plot_n = fig.create_plot(
                plot_label=style.get_plot_label() if i_p == 0 else None,
                xpos=xpos,
                ypos=ypos,
                plot_height=plot_height_here,
                plot_width=plot_width_here,
                yl="Percentage fish (%)" if i_p == 0 else None,
                ymin=0,
                ymax=50,
                yticks=[0, 50] if i_p == 0 else None,
                xmin=p["min"]-(p["max"]-p["min"])/20,
                xmax=p["max"]+(p["max"]-p["min"])/20,
                xticks=[p["min"], p["mean"], p["max"]],
                xl=p['label_show'].capitalize(),
                vlines=[p["mean"]] if p["mean"] == 0 else [0]
            )

            # Draw histogram line
            plot_n.draw_line(bin_model_parameter_median_dict[p["label"]], distribution_trajectory_dict[p["label"]] * 100,
                             lc=palette[i_p])

            # Increment X position for next parameter
            xpos = xpos + padding_short + plot_width_here

        # Reset X and update Y for next plots
        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here


# =====================================================================
# Finalize and save figure
# =====================================================================
fig.save(path_save / "figure_3.pdf")

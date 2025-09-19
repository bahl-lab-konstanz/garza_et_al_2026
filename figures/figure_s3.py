"""
Overview
---------------
This script generates and saves a set of figures analyzing the behavior of Drift Diffusion Model (DDM) fitting
and identifiability. It visualizes:

1. Parameter sampling from model fits.
2. The effect of increasing data amounts on confidence intervals (CIs) and parameter estimation error.
3. Model performance in terms of loss across iterations of optimization.
4. Distribution changes of error metrics across optimization steps.

The output is a multi-panel figure saved as a PDF.

Dependencies:
- pandas, numpy, scipy, dotenv, pathlib
- Custom project modules: Figure, BehavioralModelStyle, StatisticsService, ConfigurationDDM, ConfigurationExperiment
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values
from scipy.stats import mannwhitneyu

from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.service.statistics_service import StatisticsService
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import ConfigurationExperiment

# -----------------------------------------------------------------------------
# Load environment variables and set paths
# -----------------------------------------------------------------------------
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])     # Input data directory
path_save = Path(env['PATH_SAVE'])   # Output directory for figures
path_data = path_dir / 'benchmark'
path_data_base = path_data / 'base_dataset'

# -----------------------------------------------------------------------------
# Plotting style and layout configuration
# -----------------------------------------------------------------------------
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
plot_width_small = style.plot_size_small
plot_width_big = style.plot_width * 2

# Padding inside and between plots
padding = style.padding
padding_plot = style.padding_in_plot

# Default color palette
palette = style.palette["default"]

# -----------------------------------------------------------------------------
# Flags for enabling/disabling specific analyses
# -----------------------------------------------------------------------------
show_parameter_sampling = True
show_data_amount_vs_ci_and_error_double_plot = True
show_objective_function_vs_iterations = True

# -----------------------------------------------------------------------------
# Initialize figure canvas
# -----------------------------------------------------------------------------
fig = Figure()

# -----------------------------------------------------------------------------
# Parameter sampling visualization
# -----------------------------------------------------------------------------
if show_parameter_sampling:
    ypos -= 1  # Adjust vertical position before plotting

    # Calculate total width needed for parameter plots
    plot_width_here = plot_width_small * len(ConfigurationDDM.parameter_list) + padding_plot * (
                len(ConfigurationDDM.parameter_list) - 1)

    # Collect sampled parameter values from model files
    parameter_dict = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    for path_model in path_data_base.glob("model_test_*.hdf5"):
        if "_fit" in path_model.name:
            continue
        df_model = pd.read_hdf(path_model)
        for p in ConfigurationDDM.parameter_list:
            parameter_dict[p["label"]].append(df_model[p["label"]])

    # Normalize parameter values across defined ranges
    parameter_array = np.squeeze(np.stack([
        (np.array(parameter_dict[p["label"]]) - p["min"]) / (p["max"] - p["min"])
        for p in ConfigurationDDM.parameter_list
    ]))

    # Create plot for sampled parameters
    xticks = np.arange(0, 15, 3) + 1
    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos,
                                 plot_height=plot_height,
                                 plot_width=plot_width_here,
                                 errorbar_area=False,
                                 xmin=0, xmax=15,
                                 ymin=0, ymax=1)
    xpos += xpos_start
    ypos -= plot_height + padding

    # Draw faint parameter trajectories
    for i_link in range(parameter_array.shape[1]):
        plot_0.draw_line(xticks, parameter_array[:, i_link], pc=style.palette["neutral"][0], alpha=0.1, lw=0.1)

    # Scatter plot of parameter samples, with min/max annotated
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        x = np.ones(len(parameter_dict[p["label"]])) + i_p * 3
        y = (np.array(parameter_dict[p["label"]]) - p["min"]) / (p["max"] - p["min"])
        plot_0.draw_scatter(x, y, pc=palette[i_p], ec=palette[i_p])

        plot_0.draw_line(x[:2]-0.5, [0, 1], lc="k")
        plot_0.draw_text(x[0]-1, 0, int(p["min"]))
        plot_0.draw_text(x[0]-1, 1, int(p["max"]))

    xpos = xpos_start
ypos_start_here = ypos

# -----------------------------------------------------------------------------
# Analysis: Data amount vs confidence interval and error
# -----------------------------------------------------------------------------
if show_data_amount_vs_ci_and_error_double_plot:
    this_plot_width = plot_width * 3

    # Define data configurations with different trial counts per coherence level
    score_max = None
    analysed_parameter_list = ConfigurationExperiment.coherence_list
    data_amount_unit = "Simulated time used to fit (s)"
    data_dicts = [
        {"path_fit": fr"{path_data}\model_identifiability\1",
         "path_control": fr"{path_data}\model_identifiability_control\1",
         "data_amount": int(1)},
        {"path_fit": fr"{path_data}\model_identifiability\10",
         "path_control": fr"{path_data}\model_identifiability_control\10",
         "data_amount": int(10)},
        {"path_fit": fr"{path_data}\model_identifiability\20",
         "path_control": fr"{path_data}\model_identifiability_control\20",
         "data_amount": int(20)},
        {"path_fit": fr"{path_data}\model_identifiability\30",
         "path_control": fr"{path_data}\model_identifiability_control\30",
         "data_amount": int(30)},
        {"path_fit": fr"{path_data}\model_identifiability\50",
         "path_control": fr"{path_data}\model_identifiability_control\50",
         "data_amount": int(50)},
        {"path_fit": fr"{path_data}\model_identifiability\100",
         "path_control": fr"{path_data}\model_identifiability_control\100",
         "data_amount": int(100)},
    ]

    # Compute effective data sizes (trial count × duration × coherence levels)
    duration_fit_trial = ConfigurationExperiment.time_end_stimulus - ConfigurationExperiment.time_start_stimulus
    data_amount_array = np.array([d["data_amount"] * duration_fit_trial * len(analysed_parameter_list) for d in data_dicts])

    # Initialize data structures for confidence intervals, errors, and loss scores
    x = np.arange(len(data_amount_array)) + 1
    ci_array = {"fit": {p["label"]: np.zeros(len(data_dicts)) for p in ConfigurationDDM.parameter_list},
                "control": {p["label"]: np.zeros(len(data_dicts)) for p in ConfigurationDDM.parameter_list}}
    error_array = {"fit": {p["label"]: np.zeros(len(data_dicts)) for p in ConfigurationDDM.parameter_list},
                   "control": {p["label"]: np.zeros(len(data_dicts)) for p in ConfigurationDDM.parameter_list}}
    error_array_raw = [{"fit": {p["label"]: [] for p in ConfigurationDDM.parameter_list},
                       "control": {p["label"]: [] for p in ConfigurationDDM.parameter_list}}
                       for _ in data_dicts]

    score_mean_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}
    score_sem_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}
    score_min_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}

    # Collect raw error/loss results per parameter and trial size
    error_all = {parameter["label"]: {data["data_amount"]: {"fit": [], "control": []} for data in data_dicts} for parameter in ConfigurationDDM.parameter_list}
    error_all["loss"] = {data["data_amount"]: {"fit": [], "control": []} for data in data_dicts}

    # -------------------------------------------------------------------------
    # Loop over each data amount and extract model fits, controls, and metrics
    # -------------------------------------------------------------------------
    for i_data, data in enumerate(data_dicts):
        number_models = 0
        source_path = Path(data["path_fit"])
        model_dict = {}

        # Gather fitted models and their corresponding test sets
        for model_filepath in source_path.glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            test_label = model_filename.split("_")[2].replace(".hdf5", "")
            model_dict[test_label] = {"fit": model_filepath}
            number_models += 1

            # Match the corresponding test dataset for each fitted model
            for target_filepath in source_path.glob(f'model_test_{test_label}_*.hdf5'):
                if "_fit." in target_filepath.name or "_fit_all." in target_filepath.name:
                    continue
                else:
                    model_dict[test_label]["target"] = target_filepath
                    break

        # If control models are provided, load them as well
        if data["path_control"] is not None:
            source_path = Path(data["path_control"])
            for model_filepath in source_path.glob('model_*.hdf5'):
                model_filename = str(model_filepath.name)
                test_label = model_filename.split("_")[2].replace(".hdf5", "")
                try:
                    model_dict[test_label]["control"] = model_filepath
                except KeyError:
                    print(f"model {test_label} only available as control, no fit")
            label_list = ["fit", "control"]
        else:
            label_list = ["fit"]

        # ---------------------------------------------------------------------
        # Compute confidence intervals, errors, and loss statistics
        # ---------------------------------------------------------------------
        ci = {}
        for label in label_list:
            ci[label] = {parameter["label"]: [] for parameter in ConfigurationDDM.parameter_list}
            ci[label]["score"] = []
            for i_model, id_model in enumerate(model_dict.keys()):
                df_model_fit_list = pd.read_hdf(model_dict[id_model][label])
                df_model_target = pd.read_hdf(model_dict[id_model]["target"])

                # Parameter-level error and CI
                for i_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
                    parameter_values = np.array(df_model_fit_list[parameter["label"]])
                    ci[label][parameter["label"]].append(
                        np.percentile(parameter_values, 95) - np.percentile(parameter_values, 5)
                    )
                    error_all[parameter["label"]][data["data_amount"]][label].extend(
                        np.abs(parameter_values - df_model_target[parameter["label"]][0])
                    )

                    normalized_error = np.abs(parameter_values - df_model_target[parameter["label"]][0]) / (parameter["max"] - parameter["min"])
                    error_array_raw[i_data][label][parameter["label"]].append(normalized_error)

                # Model score values (loss function)
                score_values = np.array(df_model_fit_list["score"])
                ci[label]["score"].extend(score_values)
                error_all["loss"][data["data_amount"]][label].extend(score_values)

            # Aggregate metrics for this dataset size
            for i_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
                mean_ci = np.std(np.array(error_all[parameter["label"]][data["data_amount"]][label]) / (parameter["max"] - parameter["min"])) * 100
                ci_array[label][parameter["label"]][i_data] = mean_ci
                mean_error = np.median(np.array(error_all[parameter["label"]][data["data_amount"]][label]) / (parameter["max"] - parameter["min"])) * 100
                error_array[label][parameter["label"]][i_data] = mean_error

            mean_score = np.mean(ci[label]["score"])
            sem_score = np.std(ci[label]["score"])
            min_score = np.min(ci[label]["score"])

            score_mean_array_dict[label][i_data] = mean_score
            score_sem_array_dict[label][i_data] = sem_score
            score_min_array_dict[label][i_data] = min_score

    # -------------------------------------------------------------------------
    # Plot loss values (fit vs control) against increasing data amounts
    # -------------------------------------------------------------------------
    plot_loss = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=plot_height,
                                plot_width=this_plot_width, errorbar_area=True,
                                xmin=0, xmax=np.max(data_amount_array),
                                yl="Loss", ymin=0, ymax=30,
                                yticks=[0, 15, 30])

    for label in ["fit", "control"]:
        color = "k" if label == "fit" else style.palette["neutral"][0]
        plot_loss.draw_line(data_amount_array, score_mean_array_dict[label], lc=color, lw=0.75, line_dashes=None, yerr=score_sem_array_dict[label])
        plot_loss.draw_scatter(data_amount_array, score_mean_array_dict[label], pc=color, ec=color)

    # Annotate significant differences with markers (*, ‡)
    for i_data, data in enumerate(data_dicts):
        stat, p_value = mannwhitneyu(error_all["loss"][data["data_amount"]]["fit"], error_all["loss"][data["data_amount"]]["control"])
        if p_value < 0.05:
            plot_loss.draw_text(data_amount_array[i_data], 25, "*")

        if i_data != 0:
            data_previous = data_dicts[i_data-1]
            stat, p_value = mannwhitneyu(error_all["loss"][data["data_amount"]]["fit"],
                                         error_all["loss"][data_previous["data_amount"]]["fit"])
            if p_value < 0.05:
                plot_loss.draw_text((data_amount_array[i_data] + data_amount_array[i_data-1]) / 2, 20, "‡")

    xpos = xpos_start
    ypos = ypos - plot_height - padding

    # -------------------------------------------------------------------------
    # Plot absolute parameter errors across data sizes
    # -------------------------------------------------------------------------
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        plot_error = fig.create_plot(xpos=xpos, ypos=ypos,
                                     plot_height=plot_height, plot_width=this_plot_width,
                                     xmin=0, xmax=np.max(data_amount_array),
                                     xl=data_amount_unit if i_p == len(ConfigurationDDM.parameter_list) - 1 else None,
                                     xticks=data_amount_array if i_p == len(ConfigurationDDM.parameter_list)-1 else None,
                                     xticklabels_rotation=45 if i_p == len(ConfigurationDDM.parameter_list)-1 else None,
                                     yl=f"Absolute\n{p['label_show']} error (%)", ymin=0, ymax=50,
                                     yticks=[0, 25, 50])
        xpos = xpos + this_plot_width + padding

        for label in ["fit", "control"]:
            color = "k" if label == "fit" else style.palette["neutral"][0]
            plot_error.draw_line(data_amount_array, error_array[label][p["label"]], lc=color,
                                 lw=0.75, line_dashes=None, yerr=ci_array[label][p["label"]]/2)
            plot_error.draw_scatter(data_amount_array, error_array[label][p["label"]], pc=color, ec=color)

        # Add significance markers
        for i_data, data in enumerate(data_dicts):
            stat, p_value = mannwhitneyu(error_all[p["label"]][data["data_amount"]]["fit"], error_all[p["label"]][data["data_amount"]]["control"])
            if p_value < 0.05:
                plot_error.draw_text(data_amount_array[i_data], 45, "*")

            if i_data != 0:
                data_previous = data_dicts[i_data-1]
                stat, p_value = mannwhitneyu(error_all[p["label"]][data["data_amount"]]["fit"],
                                             error_all[p["label"]][data_previous["data_amount"]]["fit"])
                if p_value < 0.05:
                    plot_error.draw_text((data_amount_array[i_data] + data_amount_array[i_data-1]) / 2, 40, "‡")

        xpos = xpos_start
        ypos = ypos - plot_height - padding

    xpos += padding + this_plot_width
    ypos = ypos_start_here

# -----------------------------------------------------------------------------
# Analysis: Objective function (loss) vs optimization iterations
# -----------------------------------------------------------------------------
xpos_start_here = xpos
if show_objective_function_vs_iterations:
    size_plot_loss = plot_height
    size_plot_p = plot_height

    parameter_error_trajectory_list_dict = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    loss = []
    leak_list = []
    model_accepted_list = []

    # Gather accepted models with valid leak values
    for i_model, model_path in enumerate(path_data_base.glob("model_test_*.hdf5")):
        if model_path.is_dir() or model_path.name.endswith("_fit.hdf5"):
            continue
        df_model = pd.read_hdf(str(model_path))
        if df_model["leak"].iloc[0] < 0:
            leak_list.append(df_model["leak"].iloc[0])
            model_accepted_list.append(model_path.name.split("_")[2])

    # Track error trajectories across optimization runs
    leak_fit_list = []
    for i_error, error_path in enumerate(path_data_base.glob("error_test_*_fit.hdf5")):
        if error_path.is_dir():
            continue
        df_error = pd.read_hdf(str(error_path))
        if error_path.name.split("_")[2] not in model_accepted_list:
            continue

        leak_fit_list.append(df_error["leak_value"].iloc[len(df_model)-1])
        for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
            parameter_error_trajectory_list_dict[parameter['label']].append(np.array(df_error[f"{parameter['label']}_error"]))
        loss.append(np.array(df_error["score"]))

    print(f"DEBUG | correctly identified positive leak values: {np.sum(np.array(leak_fit_list) < 0)/len(leak_list)*100}%")

    # Convert collected trajectories into arrays for analysis
    parameter_error_trajectory_list_dict["score"] = np.array(loss)
    for p in ConfigurationDDM.parameter_list:
        parameter_error_trajectory_list_dict[p['label']] = np.array(parameter_error_trajectory_list_dict[p['label']])

    loss_trajectory_list = parameter_error_trajectory_list_dict["score"]
    loss_mean = np.array([np.mean(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
    loss_std = np.array([np.std(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
    iteration_list = np.arange(len(loss_mean))

    parameter_error_mean = {}
    parameter_error_std = {}
    parameter_error_mean["score"] = loss
    for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
        parameter_error_mean[parameter["label"]] = np.array([np.mean(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])
        parameter_error_std[parameter["label"]] = np.array([np.std(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])

    xpos = xpos_start_here

    # Plot loss trajectory over iterations
    plot_0a = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=size_plot_p,
                             plot_width=size_plot_p, errorbar_area=True,
                             xl=None, xmin=-0.5, xmax=1500.5, xticks=None,
                             yl="Loss", ymin=0, ymax=3, yticks=[0, 1.5, 3])
    for loss_trajectory in loss_trajectory_list:
        plot_0a.draw_line(x=iteration_list[:len(loss_trajectory)], y=loss_trajectory,
                          lc=style.palette["neutral"][0], lw=style.linewidth_single_fish, alpha=0.5)
    plot_0a.draw_line(x=iteration_list, y=loss_mean, lc="k", lw=0.75)

    xpos = xpos + size_plot_p + padding

    # Plot histograms of loss distribution at start vs end of optimization
    error_array_start = parameter_error_trajectory_list_dict["score"][:, 0]
    error_array_end = parameter_error_trajectory_list_dict["score"][:, -1]
    data_hist_value_start, data_hist_time_start = StatisticsService.get_hist(error_array_start, bins=np.arange(0, 3, 0.1), density=True, center_bin=True)
    data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(0, 3, 0.1), density=True, center_bin=True)

    plot_0b = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=size_plot_p, plot_width=size_plot_p,
                              errorbar_area=False, ymin=0, ymax=3,
                              xl=None, xmin=0, xmax=50, xticks=None)
    plot_0b.draw_line(data_hist_value_start * 100, data_hist_time_start, lc=style.palette["neutral"][0], elw=1, alpha=0.4, label="Iteration 0")
    plot_0b.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=style.palette["neutral"][0], elw=1, label="Iteration 1500")

    xpos = xpos_start_here
    ypos = ypos - padding - size_plot_loss

    # -------------------------------------------------------------------------
    # Plot parameter error trajectories (time course + histogram)
    # -------------------------------------------------------------------------
    p_final_pdf = {p["label"]: None for p in ConfigurationDDM.parameter_list}
    for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
        plot_na = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=size_plot_p, plot_width=size_plot_p, errorbar_area=True,
                                  xl="Iteration" if index_parameter==len(ConfigurationDDM.parameter_list)-1 else None, xmin=-0.5, xmax=1500.5,
                                  xticks=[0, 1500] if index_parameter==len(ConfigurationDDM.parameter_list)-1 else None,
                                  yl=f"{parameter['label_show']} error (%)", ymin=-100, ymax=100, yticks=[-100, 0, 100])
        for parameter_error_trajectory in parameter_error_trajectory_list_dict[parameter['label']]:
            plot_na.draw_line(x=iteration_list[:len(parameter_error_trajectory)], y=parameter_error_trajectory * 100,
                              lc=palette[index_parameter], lw=style.linewidth_single_fish, alpha=0.5)
        plot_na.draw_line(x=iteration_list, y=parameter_error_mean[parameter["label"]] * 100, lc="k", lw=0.75)

        xpos = xpos + size_plot_p + padding

        # Histogram of parameter error at start vs end
        error_array_start = parameter_error_trajectory_list_dict[parameter["label"]][:, 0]
        error_array_end = parameter_error_trajectory_list_dict[parameter["label"]][:, -1]
        data_hist_value_start, data_hist_time_start = StatisticsService.get_hist(error_array_start, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)
        data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)

        plot_nb = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=size_plot_p, plot_width=size_plot_p,
                                  errorbar_area=False, ymin=-100, ymax=100,
                                  xl="Percentage models (%)" if index_parameter==len(ConfigurationDDM.parameter_list)-1 else None,
                                  xmin=0, xmax=50, xticks=[0, 25, 50] if index_parameter==len(ConfigurationDDM.parameter_list)-1 else None,
                                  hlines=[0])
        plot_nb.draw_line(data_hist_value_start * 100, data_hist_time_start*100, lc=palette[index_parameter],
                          elw=1, alpha=0.4, label="Iteration 0")
        plot_nb.draw_line(data_hist_value_end * 100, data_hist_time_end * 100, lc=palette[index_parameter],
                          elw=1, label="Iteration 1500")

        xpos = xpos_start_here
        ypos = ypos - padding - size_plot_p

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
fig.save(path_save / "figure_s3.pdf", open_file=False, tight=style.page_tight)

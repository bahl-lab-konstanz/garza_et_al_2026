"""
Overview:
---------
This script performs a series of analyses and visualizations on behavioral model
data, comparing synthetic and real datasets. It loads precomputed fitting results,
evaluates model identifiability under varying noise conditions, and generates
summary figures. The main outputs are distributions of inter-bout intervals (IBIs),
objective function trajectories, parameter estimation errors, and group-level
comparisons. Results are plotted using a consistent visual style and saved as a
PDF figure.
"""
import itertools

import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import dotenv_values
from scipy.stats import mannwhitneyu

# Local project imports for figure generation, model config, and statistics
from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.service.behavioral_processing import BehavioralProcessing
from rg_behavior_model.service.statistics_service import StatisticsService
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import ConfigurationExperiment
from rg_behavior_model.utils.constants import StimulusParameterLabel

# -------------------------------------------------------------------------
# Environment setup and paths
# -------------------------------------------------------------------------
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])     # Directory containing input data
path_save = Path(env['PATH_SAVE'])   # Directory for saving output figures
path_data = path_dir / 'test_loss_synthetic-real'

# -------------------------------------------------------------------------
# Plotting style and layout configuration
# -------------------------------------------------------------------------
style = BehavioralModelStyle()

# Starting positions for placing plots on the figure grid
xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

# Dimensions for plots
plot_height = style.plot_height
plot_height_small = plot_height / 2
plot_width = style.plot_width
plot_width_small = style.plot_size_small

padding = style.plot_size
padding_small = style.padding_small

palette = style.palette["default"]

# -------------------------------------------------------------------------
# Experimental configurations
# -------------------------------------------------------------------------
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_label = ConfigurationExperiment.coherence_label
analysed_parameter_list = ConfigurationExperiment.coherence_list

# Example dataset for visualizing IBI distributions
path_data_sample = fr"{path_dir}\benchmark\model_identifiability\100\data_synthetic_test_006_2024-11-19_17-27-07.hdf5"
df_data_sample = pd.read_hdf(path_data_sample)

# Groups of models with different levels of synthetic noise added to IBIs
models_in_group_list = [
    {"label_show": r"$\Sigma_{noise}$=0",
     "path": fr"{path_data}/0",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.001",
     "path": fr"{path_data}/0_001",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.0025",
     "path": fr"{path_data}/0_0025",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.005",
     "path": fr"{path_data}/0_005",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.01",
     "path": fr"{path_data}/0_01",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.05",
     "path": fr"{path_data}/0_05",
     "loss_list": [],},
    {"label_show": r"$\Sigma_{noise}$=0.1",
     "path": fr"{path_data}/0_1",
     "loss_list": [],},
]

# Query for filtering trial times
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'
number_bins_hist = 15   # Number of bins for histogramming

# Flags for toggling analysis sections
show_sample_ibi_distribution = True
show_objective_function_vs_iterations = True
show_distribution_parameters = False
show_loss_real = True

# -----------------------------------------------------------------------------
# Initialize figure canvas
# -----------------------------------------------------------------------------
fig = Figure()

# Dictionary to store parameter fit statistics for each model group
distribution_fit_dict = {data['label_show']: {p["label"]: {"loc": [], "scale": [], "rmse": []}
                                              for p in ConfigurationDDM.parameter_list}
                         for data in models_in_group_list}

# -----------------------------------------------------------------------------
# Main loop over model groups (different noise levels)
# -----------------------------------------------------------------------------
for i_group, models_in_group in enumerate(models_in_group_list):
    noise_scale = float(models_in_group["label_show"].split("=")[-1])

    # -------------------------------------------------------------------------
    # Section 1: Plot example IBI distributions for one coherence level
    # -------------------------------------------------------------------------
    if show_sample_ibi_distribution:
        coh_to_show = 50       # Coherence level chosen for visualization
        x_limits = [0, 2]      # X-axis (time) limits in seconds
        y_limits = [0, 0.15]   # Y-axis (rate) limits

        # Create plot for this group
        title = f"{models_in_group["label_show"]}"
        plot_dist = fig.create_plot(plot_title=title,
                                    plot_label=style.get_plot_label() if i_group == 0 else None,
                                    xpos=xpos, ypos=ypos,
                                    plot_height=plot_height,
                                    plot_width=plot_width,
                                    xmin=x_limits[0], xmax=x_limits[-1],
                                    xticks=None, yticks=None,
                                    ymin=-y_limits[-1], ymax=y_limits[-1],
                                    hlines=[0])

        # Add a scale bar only to the second-to-last group for reference
        if i_group == len(models_in_group_list)-2:
            y_location_scalebar = y_limits[-1] / 6
            x_location_scalebar = x_limits[-1] / 6
            plot_dist.draw_line((1.7, 1.7), (y_location_scalebar, y_location_scalebar + 0.1), lc="k")
            plot_dist.draw_text(2, y_location_scalebar, "0.1 events/s",
                                                    textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

            plot_dist.draw_line((x_location_scalebar, x_location_scalebar + 0.5), (-y_location_scalebar, -y_location_scalebar), lc="k")
            plot_dist.draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                    textlabel_rotation='horizontal', textlabel_ha='left', textlabel_va="bottom")

        # Filter dataset for the chosen coherence and trial time window
        df_filtered = df_data_sample[df_data_sample[analysed_parameter] == coh_to_show]
        df_filtered = df_filtered.query(query_time)

        # Calculate trial duration (used to normalize histograms)
        duration = np.sum(
            BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=ConfigurationExperiment.time_end_stimulus - ConfigurationExperiment.time_start_stimulus)
        )

        # Extract response times for correct and incorrect trials
        data_corr = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1][ConfigurationExperiment.ResponseTimeColumn]
        data_err = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0][ConfigurationExperiment.ResponseTimeColumn]

        # Build histogram for correct responses
        data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(
            data_corr,
            bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1] - x_limits[0]) / 50),
            duration=duration,
            center_bin=True
        )
        # Keep only bins within plotting limits
        index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
        data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
        data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

        # Add Gaussian noise to histogram and prevent zero/negative values
        data_hist_value_corr += np.random.normal(scale=noise_scale, size=len(data_hist_value_corr))
        data_hist_value_corr[data_hist_value_corr <= 0] = np.finfo(float).eps

        # Build histogram for incorrect responses
        data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(
            data_err,
            bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1] - x_limits[0]) / 50),
            duration=duration,
            center_bin=True
        )
        index_in_limits = np.argwhere(np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
        data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
        data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

        data_hist_value_err += np.random.normal(scale=noise_scale, size=len(data_hist_value_err))
        data_hist_value_err[data_hist_value_err <= 0] = np.finfo(float).eps

        # Plot correct trials (positive y) and incorrect trials (negative y)
        plot_dist.draw_line(data_hist_time_corr, data_hist_value_corr, lc=style.palette["correct_incorrect"][0], lw=0.75)
        plot_dist.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=style.palette["correct_incorrect"][1], lw=0.75)

        # Move y-position down for next plot
        ypos -= (padding + plot_height)

    # -------------------------------------------------------------------------
    # Section 2: Objective function trajectories and estimation error
    # -------------------------------------------------------------------------
    if show_objective_function_vs_iterations:
        ypos_start_here = ypos
        loss = []   # Stores loss trajectories per model
        parameter_error_trajectory_list_dict = {p["label"]: [] for p in ConfigurationDDM.parameter_list}

        # Loop through all error trajectory files for this group
        for i_error, error_path in enumerate(Path(models_in_group["path"]).glob("error_test_*_fit.hdf5")):
            if error_path.is_dir():
                continue

            df_error = pd.read_hdf(str(error_path))
            if len(df_error) < 3:   # Skip if file has too few data points
                continue

            # Collect parameter error trajectories
            for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
                parameter_error_trajectory_list_dict[parameter['label']].append(np.array(df_error[f"{parameter['label']}_error"]))

            # Collect loss trajectory
            loss.append(np.array(df_error["score"]))

        # Convert collected lists into numpy arrays for easier analysis
        parameter_error_trajectory_list_dict["score"] = np.array(loss)
        for p in ConfigurationDDM.parameter_list:
            parameter_error_trajectory_list_dict[p['label']] = np.array(parameter_error_trajectory_list_dict[p['label']])

        # Compute mean and standard deviation of loss over iterations
        loss_trajectory_list = parameter_error_trajectory_list_dict["score"]
        loss_mean = np.array([np.mean(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
        loss_std = np.array([np.std(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
        iteration_list = np.arange(len(loss_mean))

        # Compute mean/std trajectories for parameter errors
        parameter_error_mean = {}
        parameter_error_std = {}

        parameter_error_mean["score"] = loss
        for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
            parameter_error_mean[parameter["label"]] = np.array([np.mean(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])
            parameter_error_std[parameter["label"]] = np.array([np.std(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])

        # Distribution of final losses after fitting (iteration 1500)
        error_array_end = parameter_error_trajectory_list_dict["score"][:, -1]
        data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(0, 3, 0.05), density=True, center_bin=True)

        # Plot histogram of final loss distribution
        plot_loss = fig.create_plot(plot_label=style.get_plot_label() if i_group == 0 else None, xpos=xpos, ypos=ypos,
                                  plot_height=plot_height, plot_width=plot_width,
                                  errorbar_area=False, ymin=0, ymax=3,
                                  yticks=[0, 1.5, 3] if i_group == 0 else None,
                                  yl=f"Loss at\niteration 1500" if i_group == 0 else None,
                                  xl=None, xmin=0, xmax=50, xticks=None)
        plot_loss.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=style.palette["neutral"][0], elw=1)
        print(fr"{models_in_group['label_show']} | loss | {np.mean(error_array_end):.04f}$\pm${np.std(error_array_end):.04f}")

        ypos = ypos - padding - plot_height

        df_noise_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        p_final_pdf = {p["label"]: None for p in ConfigurationDDM.parameter_list}

        # ---------------------------------------------------------------------
        # Plot parameter estimation error distributions
        # ---------------------------------------------------------------------
        for index_parameter, parameter in enumerate(ConfigurationDDM.parameter_list):
            error_array_end = parameter_error_trajectory_list_dict[parameter["label"]][:, -1]
            data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)

            plot_p = fig.create_plot(plot_label=style.get_plot_label() if i_group == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                      errorbar_area=False, ymin=-1, ymax=1,
                                      yticks=[-1, 0, 1] if i_group == 0 else None,
                                      yl=f"Estimation error at iteration 1500" if (index_parameter == 0 and i_group == 0) else None,
                                      xl="Percentage models (%)" if (index_parameter == len(ConfigurationDDM.parameter_list)-1 and i_group == 0) else None,
                                      xmin=0, xmax=50, xticks=[0, 25, 50] if index_parameter == len(ConfigurationDDM.parameter_list)-1 else None, hlines=[0])
            plot_p.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=palette[index_parameter], elw=1)

            # Report interquartile range of estimation error as percentage of parameter range
            q = np.quantile(parameter_error_trajectory_list_dict[parameter["label"]][:, -1], q=[0.25, 0.75])
            q = q / (parameter["max"] - parameter["min"])
            print(f"{models_in_group['label_show']} | {parameter['label_show']} | 90% of the error is between {100*q[0]:.04f} and {100*q[1]:.04f}%")

            ypos = ypos - padding - plot_height

        # Adjust x/y placement for next group
        if i_group == len(models_in_group_list)-1:
            xpos = xpos_start
            ypos -= plot_height
        else:
            xpos = xpos + plot_width + padding
            ypos = ypos_start

    if i_group == len(models_in_group_list)-1:
        xpos = xpos_start
        ypos -= (padding + plot_height)

# -----------------------------------------------------------------------------
# Section 3: Distribution of fitted parameters across groups (optional)
# -----------------------------------------------------------------------------
if show_distribution_parameters:
    from_best_model = True
    plot_height_here = style.plot_height_small
    padding_here = style.padding_in_plot_small

    distribution_trajectory_dict = {p["label"]: np.zeros((number_bins_hist, len(models_in_group_list))) for p in
                                    ConfigurationDDM.parameter_list}
    raw_data_dict_per_fish = {p["label"]: {i_age: {} for i_age in range(len(models_in_group_list))} for p in
                              ConfigurationDDM.parameter_list}
    raw_data_dict = {p["label"]: {i_age: [] for i_age in range(len(models_in_group_list))} for p in
                     ConfigurationDDM.parameter_list}

    for i_age, models_in_age in enumerate(models_in_group_list):
        model_dict = {}
        n_models = 0
        path_dir = Path(models_in_age["path"])
        for model_filepath in path_dir.glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

        fish_list = np.arange(len(model_dict.keys()))

        model_parameter_median_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_median_dict["score"] = {}
        model_parameter_dict["score"] = {}
        model_parameter_median_array = np.zeros((len(ConfigurationDDM.parameter_list) + 1, len(fish_list)))
        for i_model, id_model in enumerate(model_dict.keys()):
            df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])

            id_fish = i_model
            if from_best_model:
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit_list = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

            model_parameter_median_dict["score"][id_fish] = np.median(df_model_fit_list["score"])
            model_parameter_dict["score"][id_fish] = np.array(df_model_fit_list["score"])
            model_parameter_median_array[0, i_model] = np.median(df_model_fit_list["score"])

            for i_p, p in enumerate(ConfigurationDDM.parameter_list):
                p_median = np.median(df_model_fit_list[p["label"]])
                model_parameter_median_dict[p["label"]][id_fish] = p_median
                model_parameter_dict[p["label"]][id_fish] = np.array(df_model_fit_list[p["label"]])
                model_parameter_median_array[i_p + 1, i_model] = p_median

                if id_model not in raw_data_dict_per_fish[p["label"]][i_age].keys():
                    raw_data_dict_per_fish[p["label"]][i_age][id_model] = [p_median]
                else:
                    raw_data_dict_per_fish[p["label"]][i_age][id_model].append(p_median)

                raw_data_dict[p["label"]][i_age].append(p_median)

        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[ConfigurationDDM.score_config["label"]], bin_model_parameter_median_dict[
            ConfigurationDDM.score_config["label"]] = StatisticsService.get_hist(
            model_parameter_median_array[0, :], center_bin=True,
            hist_range=[ConfigurationDDM.score_config["min"], ConfigurationDDM.score_config["max"]],
            bins=number_bins_hist,
            density=True
        )
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[
                p["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :], center_bin=True, hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,
                density=True
            )

            distribution_trajectory_dict[p["label"]][:, i_age] = hist_model_parameter_median_dict[p["label"]]

    for i_age in range(len(models_in_group_list)):
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            plot_n = fig.create_plot(plot_label=style.get_plot_label() if i_p == 0 and i_age == 0 else None, xpos=xpos,
                                     ypos=ypos,
                                     plot_height=plot_height_here, plot_width=plot_width,
                                     yl="Percentage fish (%)" if i_p == 0 and i_age == 0 else None, ymin=0, ymax=50,
                                     yticks=[0, 50] if i_p == 0 and i_age == 0 else None,
                                     xl=p['label_show'].capitalize() if i_age == len(
                                         models_in_group_list) - 1 else None, xmin=p["min"], xmax=p["max"],
                                     xticks=[p["min"], p["mean"], p["max"]] if i_age == len(
                                         models_in_group_list) - 1 else None,
                                     vlines=[p["mean"]] if p["mean"] == 0 else [])

            plot_n.draw_line(bin_model_parameter_median_dict[p["label"]],
                             distribution_trajectory_dict[p["label"]][:, i_age] * 100,
                             lc=palette[i_p], label=models_in_group_list[i_age]["label_show"] if i_p == len(
                    ConfigurationDDM.parameter_list) - 1 else None)

            xpos = xpos + padding_small + plot_width

        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here

    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_group_list)), 2))
    for pairs_to_compare in pairs_to_compare_list:
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            data_0 = raw_data_dict[p["label"]][pairs_to_compare[0]]
            data_1 = raw_data_dict[p["label"]][pairs_to_compare[1]]

            U1, p_value = mannwhitneyu(data_0, data_1, method="exact")

            print(
                f"{models_in_group_list[pairs_to_compare[0]]['label_show']} vs {models_in_group_list[pairs_to_compare[1]]['label_show']} | {p['label_show']} | parameter p value {p_value}")

    xpos = xpos_start
    ypos = ypos - (padding - padding_here)

# -----------------------------------------------------------------------------
# Section 4: Real data loss analysis
# -----------------------------------------------------------------------------
if show_loss_real:
    # Define real experimental groups (age- and genotype-based comparisons)
    models_real_in_group_list = [
        {"label_show": "5dpf",
         "path": fr"{path_dir}\age_analysis\5_dpf",
         "loss_list": []},
        {"label_show": "6dpf",
         "path": fr"{path_dir}\age_analysis\6_dpf",
         "loss_list": []},
        {"label_show": "7dpf",
         "path": fr"{path_dir}\age_analysis\7_dpf",
         "loss_list": []},
        {"label_show": "8dpf",
         "path": fr"{path_dir}\age_analysis\8_dpf",
         "loss_list": []},
        {"label_show": "9dpf",
         "path": fr"{path_dir}\age_analysis\9_dpf",
         "loss_list": []},
        {"label_show": "scn1lab +/+",
         "path": fr"{path_dir}\harpaz_2021\scn1lab_NIBR_20200708\wt",
         "loss_list": []},
        {"label_show": "scn1lab +/-",
         "path": fr"{path_dir}\harpaz_2021\scn1lab_NIBR_20200708\het",
         "loss_list": []},
        {"label_show": "disc +/+",
         "path": fr"{path_dir}\harpaz_2021\disc1_hetnix\wt",
         "loss_list": []},
        {"label_show": "disc +/-",
         "path": fr"{path_dir}\harpaz_2021\disc1_hetnix\het",
         "loss_list": []},
        {"label_show": "disc -/-",
         "path": fr"{path_dir}\harpaz_2021\disc1_hetnix\hom",
         "loss_list": []},
    ]

    # Allocate space for a large summary plot
    ypos -= plot_height * 4

    # Create combined loss plot (synthetic + real groups)
    x_ticks = np.arange(len(models_real_in_group_list)+len(models_in_group_list))
    x_tick_labels = [m["label_show"] for m in models_in_group_list] + [m["label_show"] for m in models_real_in_group_list]
    plot_loss = fig.create_plot(plot_label=style.get_plot_label(),
                                xpos=xpos, ypos=ypos,
                                plot_height=plot_height * 4, plot_width=plot_width * 8,
                                ymin=0, ymax=3, errorbar_area=False,
                                yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], yl="Loss at\niteration 1500",
                                xl=None, xmin=-0.5, xmax=len(x_ticks) - 0.5,
                                xticks=x_ticks, xticklabels=x_tick_labels, xticklabels_rotation=45,)

    # Plot average ± std loss for each synthetic group
    for i_m, m in enumerate(models_in_group_list):
        path = m["path"]

        for i_model, model_path in enumerate(Path(path).glob("model_*_fit.hdf5")):
            if model_path.is_dir():
                continue

            df_m = pd.read_hdf(str(model_path))
            m["loss_list"].append(np.median(df_m["score"]))

        plot_loss.draw_scatter(x_ticks[i_m], np.mean(m["loss_list"]), yerr=np.std(m["loss_list"]))

    # Plot average ± std loss for each real experimental group
    x_offset = len(models_in_group_list)
    for i_m_real, m_real in enumerate(models_real_in_group_list):
        path_real = m_real["path"]

        for i_model, model_path in enumerate(Path(path_real).glob("model_*_fit.hdf5")):
            if model_path.is_dir():
                continue

            df_m_real = pd.read_hdf(str(model_path))
            m_real["loss_list"].append(np.median(df_m_real["score"]))

        plot_loss.draw_scatter(x_offset + x_ticks[i_m_real], np.mean(m_real["loss_list"]), yerr=np.std(m_real["loss_list"]))

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
fig.save(path_save / "figure_s4.pdf", open_file=False, tight=style.page_tight)

"""
Overview: Compare behavioral model fits across different simulation time steps (dt)
-------------------------------------------------------------------------------

This script evaluates the influence of different time step values (dt) on
behavioral model fits. It loads experimental and simulated fish data,
computes response time distributions for correct and incorrect responses
across coherence levels, and compares fitted models to experimental data
using KL divergence.

Key functionalities:
- Load experimental and synthetic datasets for multiple fish and dt values.
- Compute and visualize response time distributions (correct vs. incorrect).
- Plot comparison figures for different dt values.
- Quantify model fit quality by computing KL divergence losses between
  experimental and simulated distributions.
- Save summary plots as PDF.

Dependencies:
- pandas, numpy, pathlib, dotenv
- Custom modules: Figure, BehavioralModelStyle, BehavioralProcessing,
  StatisticsService, ConfigurationExperiment, StimulusParameterLabel

Output:
- A multi-panel figure saved as "figure_s1_compare_dt.pdf" in the output directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from garza_et_al_2026.figures.style import BehavioralModelStyle
from garza_et_al_2026.service.behavioral_processing import BehavioralProcessing
from garza_et_al_2026.service.statistics_service import StatisticsService
from garza_et_al_2026.utils.configuration_experiment import ConfigurationExperiment
from garza_et_al_2026.utils.constants import StimulusParameterLabel

# --------------------------------------------------------------------------
# Load environment variables (data and save paths)
# --------------------------------------------------------------------------
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_data = path_dir / "dt_analysis"
path_save = Path(env['PATH_SAVE'])

# --------------------------------------------------------------------------
# Plot style parameters (layout, sizes, padding)
# --------------------------------------------------------------------------
style = BehavioralModelStyle()

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5
plot_width = style.plot_width

padding = style.padding
padding_plot = style.padding_in_plot
plot_height_row = plot_height_small * 2 + padding_plot

# --------------------------------------------------------------------------
# Flags to toggle plots
# --------------------------------------------------------------------------
show_distribution_error_across_repeats = True
show_distribution_parameters = True

# --------------------------------------------------------------------------
# Experimental configuration
# --------------------------------------------------------------------------
analysed_parameter_list = ConfigurationExperiment.coherence_list
time_start_stimulus = ConfigurationExperiment.time_start_stimulus
time_end_stimulus = ConfigurationExperiment.time_end_stimulus
time_experimental_trial = ConfigurationExperiment.time_experimental_trial
analysed_parameter = StimulusParameterLabel.COHERENCE.value
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

# --------------------------------------------------------------------------
# Plot configuration
# --------------------------------------------------------------------------
number_bins_hist = 15

# --------------------------------------------------------------------------
# Simulation models with varying dt values
# --------------------------------------------------------------------------
models_in_dt_list = [
    {"label_show": "dt=0.0001", "path": fr"{path_data}\5_dpf_0_0001_sim",},
    {"label_show": "dt=0.001", "path": fr"{path_data}\5_dpf_0_001_sim",},
    {"label_show": "dt=0.01",  "path": fr"{path_data}\5_dpf_0_01_fit",},
    {"label_show": "dt=0.1",   "path": fr"{path_data}\5_dpf_0_1_sim",},
    {"label_show": "dt=0.5",   "path": fr"{path_data}\5_dpf_0_5_sim",},
    {"label_show": "dt=1",     "path": fr"{path_data}\5_dpf_1_sim",},
]

# --------------------------------------------------------------------------
# Initialize figure object
# --------------------------------------------------------------------------
fig = Figure()

# Data storage and configuration for plotting
df_dict = {}
fish_to_include_list = [ConfigurationExperiment.example_fish_list[1]]
config_list = [
    {"label": "data", "line_dashes": None, "alpha": 0.5, "color": None,
     "time_start_stimulus": ConfigurationExperiment.time_start_stimulus,
     "time_end_stimulus": ConfigurationExperiment.time_end_stimulus},
    {"label": "fit", "line_dashes": (2, 4), "alpha": 1, "color": "k",
     "time_start_stimulus": ConfigurationExperiment.time_start_stimulus,
     "time_end_stimulus": ConfigurationExperiment.time_end_stimulus}
]

# --------------------------------------------------------------------------
# Plot limits and parameters
# --------------------------------------------------------------------------
palette = style.palette["stimulus"]
x_limits = [0, 2]
y_limits = [0, 0.2]

plot_height_here = plot_height_small * 3
plot_width_here = 0.7

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def compute_distributions(df_filtered, duration):
    """
    Compute response time distributions for correct and incorrect responses.

    Returns:
        - histogram values for correct responses
        - raw correct RT data
        - histogram values for incorrect responses
        - raw incorrect RT data
        - histogram bin centers
    """
    # Separate correct and incorrect trials
    data_corr = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1][ConfigurationExperiment.ResponseTimeColumn]
    data_err = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0][ConfigurationExperiment.ResponseTimeColumn]

    # Histogram for correct responses
    data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(
        data_corr,
        bins=int((x_limits[1] - x_limits[0]) / (0.05)),
        hist_range=x_limits,
        duration=duration,
        center_bin=True
    )
    data_hist_time_corr = data_hist_time_corr.flatten()
    data_hist_value_corr = data_hist_value_corr.flatten()

    # Histogram for incorrect responses
    data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(
        data_err,
        bins=int((x_limits[1] - x_limits[0]) / (0.05)),
        hist_range=x_limits,
        duration=duration,
        center_bin=True
    )
    data_hist_value_err = data_hist_value_err.flatten()

    return data_hist_value_corr, data_corr, data_hist_value_err, data_err, data_hist_time_corr


def extract_rt_hist_from_df(df, analysed_parameter_list, time_start_stimulus=10, time_end_stimulus=40, duration_dict=None):
    """
    Extract response time distributions for each parameter (e.g., coherence level).

    Args:
        df: DataFrame with fish data.
        analysed_parameter_list: List of parameter values to filter by.
        time_start_stimulus: Start of analysis window.
        time_end_stimulus: End of analysis window.
        duration_dict: Optional dict of durations per parameter.

    Returns:
        distributions_dict: Dictionary of RT histograms and raw data.
    """
    distributions_dict = {}
    for i_param, parameter in enumerate(analysed_parameter_list):
        # Filter by parameter and time window
        df_filtered = df[df[analysed_parameter] == parameter]
        df_filtered = df_filtered.query(query_time)

        # Determine trial duration
        if duration_dict is None:
            duration = np.sum(
                BehavioralProcessing.get_duration_trials_in_df(
                    df_filtered,
                    fixed_time_trial=time_end_stimulus - time_start_stimulus
                )
            )
        else:
            duration = duration_dict[parameter]

        # Compute distributions
        data_hist_value_corr, data_corr, data_hist_value_err, data_err, data_hist_time_corr = compute_distributions(df_filtered, duration)

        distributions_dict[parameter] = {
            "corr": data_hist_value_corr,
            "data_corr": data_corr,
            "err": data_hist_value_err,
            "data_err": data_err,
            "bins": data_hist_time_corr,
            "duration": duration
        }

    return distributions_dict

# --------------------------------------------------------------------------
# Main analysis loop: iterate over dt values and fish
# --------------------------------------------------------------------------
dt_array = np.zeros(len(models_in_dt_list))
loss_dict = {}

for i_m, m in enumerate(models_in_dt_list):
    dt_array[i_m] = float(m["label_show"].split("=")[-1])
    path_dir = Path(m["path"])

    # Load data for each fish
    for i_fish, fish in enumerate(fish_to_include_list):
        df_data = pd.read_hdf(path_dir / f"data_fish_{fish}.hdf5")
        for path_fit in path_dir.glob(f"data_synthetic_fish_{fish}_*.hdf5"):
            df_fit = pd.read_hdf(path_fit)
            break
        df_dict[fish] = {"fit": df_fit, "data": df_data, "color": style.palette["fish_code"][i_fish]}

    # Storage for plots
    plot_section = {"corr": {}, "err": {}}

    # Remove merged "all fish" dataset if present
    df_dict.pop(ConfigurationExperiment.all_fish_label, None)

    # Plot RT distributions across parameters and configurations
    for i_k, k in enumerate(df_dict.keys()):
        for i_param, parameter in enumerate(analysed_parameter_list):
            # Define plot title only for first row/column
            line_dashes = None
            plot_title = None
            if i_k == 0 and i_m == 0:
                plot_title = f"Coh={parameter}%" if i_param == 0 else f"{parameter}%"

            # Create plot for correct/incorrect distributions
            plot_section["corr"][i_param] = fig.create_plot(
                plot_label=style.get_plot_label() if i_k == 0 and i_param == 0 and i_m == 0 else None,
                plot_title=plot_title,
                xpos=xpos + i_param * (plot_width_here + padding_plot),
                ypos=ypos + plot_height_small + padding_plot,
                plot_height=plot_height_here,
                plot_width=plot_width_here,
                xmin=x_limits[0], xmax=x_limits[-1],
                xticks=None, yticks=None,
                yl=m["label_show"] if i_param == 0 else None,
                ymin=-y_limits[-1], ymax=y_limits[-1],
                hlines=[0]
            )
            plot_section["err"][i_param] = plot_section["corr"][i_param]

            # Add scale bars only for last panel
            if i_param == len(analysed_parameter_list) - 1 and i_k == len(df_dict) - 1 and i_m == len(models_in_dt_list) - 1:
                y_location_scalebar = y_limits[-1] / 6
                x_location_scalebar = x_limits[-1] / 6
                plot_section["corr"][i_param].draw_line((1.7, 1.7),
                                                        (y_location_scalebar, y_location_scalebar + 0.1),
                                                        lc="k")
                plot_section["corr"][i_param].draw_text(2, y_location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left',
                                                        textlabel_va="bottom")

                plot_section["corr"][i_param].draw_line((x_location_scalebar, x_location_scalebar + 0.5),
                                                        (-y_location_scalebar, -y_location_scalebar), lc="k")
                plot_section["corr"][i_param].draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                        textlabel_rotation='horizontal', textlabel_ha='left',
                                                        textlabel_va="bottom")

            # Draw data and fit distributions
            for config in config_list:
                df = df_dict[k][config["label"]]
                target_distributions_dict = extract_rt_hist_from_df(
                    df,
                    analysed_parameter_list,
                    time_start_stimulus=config["time_start_stimulus"],
                    time_end_stimulus=config["time_end_stimulus"]
                )
                data_hist_time_corr = data_hist_time_err = target_distributions_dict[parameter]["bins"]
                data_hist_value_corr = target_distributions_dict[parameter]["corr"]
                data_hist_value_err = target_distributions_dict[parameter]["err"]

                # Define line colors and alpha
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

                # Draw correct and incorrect RT distributions
                plot_section_corr = plot_section["corr"][i_param]
                plot_section_err = plot_section["err"][i_param]
                plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=lc_correct,
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_correct)
                plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=lc_incorrect,
                                           lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_incorrect)

        # Move to next row of plots
        ypos = ypos - (padding_plot + plot_height_small)

# Reset x-position for later plots
xpos = xpos_start

# --------------------------------------------------------------------------
# Compute KL divergence losses between experimental and fit distributions
# --------------------------------------------------------------------------
for i_m, m in enumerate(models_in_dt_list):
    path_dir = Path(m["path"])
    for path_fish_experiment in path_dir.glob(f"data_fish_*.hdf5"):
        label_fish = path_fish_experiment.name.split("_")[2].replace(".hdf5", "")

        if label_fish == "all":
            continue
        df_experiment = pd.read_hdf(path_fish_experiment)
        distribution_experiment_dict = extract_rt_hist_from_df(
            df_experiment, analysed_parameter_list,
            time_start_stimulus=time_start_stimulus,
            time_end_stimulus=time_end_stimulus
        )

        if label_fish not in loss_dict.keys():
            loss_dict[label_fish] = {}

        # Load fit distribution (simulate durations to account for missed trials at large dt)
        for path_fish_fit in path_dir.glob(f"data_synthetic_fish_{label_fish}*.hdf5"):
            df_fit = pd.read_hdf(path_fish_fit)
        duration_dict_fit = {p: 30 * 30 for p in analysed_parameter_list}
        distribution_fit_dict = extract_rt_hist_from_df(
            df_fit, analysed_parameter_list,
            time_start_stimulus=time_start_stimulus,
            time_end_stimulus=time_end_stimulus,
            duration_dict=duration_dict_fit
        )

        # Compute total loss for this fish and dt
        loss_dict[label_fish][m["label_show"]] = 0
        for p in analysed_parameter_list:
            loss_dict[label_fish][m["label_show"]] += BehavioralProcessing.kl_divergence_rt_distribution_weight(
                distribution_experiment_dict[p]["data_corr"],
                distribution_fit_dict[p]["data_corr"],
                resolution=int((x_limits[1] - x_limits[0]) / (0.05)),
                focus_scope=x_limits,
                duration_0=distribution_experiment_dict[p]["duration"],
                duration_1=distribution_fit_dict[p]["duration"],
                order_max_result=True,
                correct_by_area=False
            )
            loss_dict[label_fish][m["label_show"]] += BehavioralProcessing.kl_divergence_rt_distribution_weight(
                distribution_experiment_dict[p]["data_err"],
                distribution_fit_dict[p]["data_err"],
                resolution=int((x_limits[1] - x_limits[0]) / (0.05)),
                focus_scope=x_limits,
                duration_0=distribution_experiment_dict[p]["duration"],
                duration_1=distribution_fit_dict[p]["duration"],
                order_max_result=True,
                correct_by_area=False
            )

# --------------------------------------------------------------------------
# Aggregate losses across fish and dt
# --------------------------------------------------------------------------
loss_array = np.zeros((len(loss_dict), len(models_in_dt_list)))
for i_fish, label_fish in enumerate(loss_dict.keys()):
    for i_m, m in enumerate(models_in_dt_list):
        loss_array[i_fish, i_m] = loss_dict[label_fish][m["label_show"]] if m["label_show"] in loss_dict[label_fish].keys() else np.nan
loss_mean_array = np.nanmean(loss_array, axis=0)

# --------------------------------------------------------------------------
# Final plot: Loss vs. dt comparison
# --------------------------------------------------------------------------
plot_dt_compare = fig.create_plot(
    plot_label=style.get_plot_label(),
    xpos=xpos, ypos=ypos,
    plot_height=plot_height,
    plot_width=len(analysed_parameter_list)-1 * (plot_width_here + padding_plot),
    xlog=True, xmin=dt_array[0], xmax=dt_array[-1], xl="dt (s)",
    yticks=[0, 5, 10], xticks=dt_array,
    ymin=0, ymax=10, yl="Loss",
    vlines=dt_array
)
# Draw individual fish loss curves
for i_fish in range(len(loss_dict)):
    plot_dt_compare.draw_line(dt_array, loss_array[i_fish, :], lc=style.palette["neutral"][0], lw=0.05)
# Draw mean loss curve
plot_dt_compare.draw_line(dt_array, loss_mean_array, lc="k", lw=1)

# --------------------------------------------------------------------------
# Save final figure
# --------------------------------------------------------------------------
fig.save(path_save / "figure_s1_compare_dt.pdf", open_file=False, tight=style.page_tight)

"""
Overview: Validation of behavioral model loss and response characteristics
------------------------------------------------------------------------

This script validates behavioral model fitting by comparing experimental
data with simulated data. It evaluates different loss formulations
(e.g., KL divergence and its variants) and examines how models reproduce
key behavioral measures.

Key functionalities:
- Load experimental and simulated datasets for multiple loss functions.
- Plot reduction in loss across optimization iterations.
- Generate psychometric curves (percent correct vs. stimulus coherence).
- Analyze and compare interbout intervals (response times) across coherence levels.
- Save the final multi-panel figure as a PDF.

Dependencies:
- pandas, numpy, pathlib, dotenv
- Custom modules: Figure, BehavioralModelStyle, BehavioralProcessing,
  ConfigurationExperiment, StimulusParameterLabel

Output:
- A multi-panel figure saved as "figure_s1_loss_validation.pdf".
"""

import pandas as pd
import numpy as np
import pathlib

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from garza_et_al_2026.figures.style import BehavioralModelStyle
from garza_et_al_2026.service.behavioral_processing import BehavioralProcessing
from garza_et_al_2026.utils.configuration_experiment import ConfigurationExperiment
from garza_et_al_2026.utils.constants import StimulusParameterLabel

# --------------------------------------------------------------------------
# Load environment variables (data paths, save directory)
# --------------------------------------------------------------------------
env = dotenv_values()
path_dir = pathlib.Path(env['PATH_DIR'])
path_save = pathlib.Path(env['PATH_SAVE'])

# --------------------------------------------------------------------------
# Plot style configuration
# --------------------------------------------------------------------------
style = BehavioralModelStyle(plot_label_i=1)

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_width = style.plot_width

padding = style.padding
padding_small = style.padding_small

# --------------------------------------------------------------------------
# Experimental configuration
# --------------------------------------------------------------------------
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = ConfigurationExperiment.coherence_list

# List of models to compare with different loss functions
models_in_loss_list = [
    {"label_show": r"$D_{KL}^*$",
     "path": fr"{path_dir}\base_dataset_5dpfWT",
     "path_data": fr"{path_dir}\base_dataset_5dpfWT\data_fish_all.hdf5",
     "path_simulation": fr"{path_dir}\base_dataset_5dpfWT\data_synthetic_fish_all.hdf5",
     "ylim": [0, 20]},
    {"label_show": r"$D_{KL}$",
     "path": fr"{path_dir}\5_dpf_dkl",
     "path_data": fr"{path_dir}\5_dpf_dkl\data_fish_all.hdf5",
     "path_simulation": fr"{path_dir}\5_dpf_dkl\data_synthetic_fish_all.hdf5",
     "ylim": [0, 500]},
]

# --------------------------------------------------------------------------
# Initialize figure
# --------------------------------------------------------------------------
fig = Figure()

# --------------------------------------------------------------------------
# Load experimental and simulated data for each model
# --------------------------------------------------------------------------
for m in models_in_loss_list:
    try:
        m["df_data"] = pd.read_hdf(m["path_data"])
        m["df_simulation"] = pd.read_hdf(m["path_simulation"])
        m["df_simulation"].reset_index(inplace=True)
    except (KeyError, NotImplementedError):
        print(f"No data for group {m['label_show']}")

# Query time window used to filter trials
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

# --------------------------------------------------------------------------
# Plot options (set True/False to enable/disable)
# --------------------------------------------------------------------------
show_loss_reduction = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True

# --------------------------------------------------------------------------
# Plot 1: Loss reduction across optimization iterations
# --------------------------------------------------------------------------
if show_loss_reduction:
    ypos = ypos - padding_small
    for i_group, models_in_group in enumerate(models_in_loss_list):
        loss_list = []
        loss_start_list = []
        loss_end_list = []
        ylim = models_in_group["ylim"]

        # Collect loss values from error logs
        for path_error in pathlib.Path(models_in_group["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_start_list.append(df_error["score"][0])
            loss_end = df_error["score"][len(df_error)-1]
            loss_end_list.append(loss_end)
            loss_list.append(loss_end)

        # Create loss plot
        ylim_low_with_offset = ylim[0] - ylim[-1]/20
        plot_title = models_in_group['label_show']
        plot_loss = fig.create_plot(
            plot_label=style.get_plot_label() if i_group == 0 else None,
            plot_title=plot_title,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height, plot_width=plot_width,
            ymin=ylim_low_with_offset, ymax=ylim[-1],
            yticks=[ylim[0], int(np.mean(ylim)), ylim[-1]],
            yl="Loss" if i_group == 0 else None,
            xl="Iteration", xmin=0.5, xmax=2.5,
            xticks=[1, 2], xticklabels=["0", "1500"]
        )
        xpos += plot_width + padding_small

        # Draw loss trajectories
        for i_loss in range(len(loss_list)):
            loss_start = loss_start_list[i_loss]
            loss_end = loss_end_list[i_loss]
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

    xpos = xpos_start
    ypos = ypos - padding - plot_height

# --------------------------------------------------------------------------
# Plot 2: Psychometric curve (percent correct vs. coherence)
# --------------------------------------------------------------------------
if show_psychometric_curve:
    for i_m, m in enumerate(models_in_loss_list):
        # Experimental data filtering
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[analysed_parameter].isin(analysed_parameter_list)]
        try:
            # Harmonize ID column naming across datasets
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, correct_bout_list_data, std_correct_bout_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=analysed_parameter)

        # Simulated data filtering
        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, correct_bout_list_sim, std_correct_bout_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_simulation_filtered, analysed_parameter=analysed_parameter)

        # Ensure parameter arrays are integers
        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # Create psychometric curve plot
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_m == 0 else None,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height, plot_width=plot_width,
            errorbar_area=True,
            xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
            xticks=None,
            yl="Percentage\ncorrect swims (%)" if i_m == 0 else None,
            ymin=45, ymax=100,
            yticks=[50, 75, 100] if i_m == 0 else None,
            hlines=[50]
        )

        # Draw experimental vs. simulated psychometric curves
        plot_0.draw_line(x=parameter_list_data, y=np.array(correct_bout_list_data)*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data)*100,
                         lc="k", lw=1)
        plot_0.draw_line(x=parameter_list_sim, y=np.array(correct_bout_list_sim)*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim)*100,
                         lc="k", lw=1, line_dashes=(1, 2))

        # Adjust spacing depending on model position
        pad = padding if i_m == len(models_in_loss_list) - 1 else padding_small
        xpos = xpos + pad + plot_width

    ypos = ypos - padding - plot_height*2
    xpos = xpos_start

# --------------------------------------------------------------------------
# Plot 3: Interbout interval vs. coherence
# --------------------------------------------------------------------------
if show_coherence_vs_interbout_interval:
    plot_height_here = plot_height * 2

    for i_m, m in enumerate(models_in_loss_list):
        # Experimental data filtering with outlier removal (5th–95th quantile)
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[analysed_parameter].isin(analysed_parameter_list)]
        df_data_filtered = df_data_filtered[df_data_filtered[ConfigurationExperiment.CorrectBoutColumn] != -1]
        ibi_quantiles = np.quantile(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn], [0.05, 0.95])
        df_data_filtered = df_data_filtered[
            np.logical_and(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] > ibi_quantiles[0],
                           df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] < ibi_quantiles[1])
        ]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, interbout_interval_list_data, std_interbout_interval_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=analysed_parameter, column_name=ConfigurationExperiment.ResponseTimeColumn)

        # Simulated data
        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, interbout_interval_list_sim, std_interbout_interval_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_simulation_filtered, analysed_parameter=analysed_parameter, column_name=ConfigurationExperiment.ResponseTimeColumn)

        # Ensure parameter arrays are integers
        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # Create interbout interval plot
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_m == 0 else None,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height_here, plot_width=plot_width,
            errorbar_area=True,
            xl=ConfigurationExperiment.coherence_label,
            xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
            xticks=[int(p) for p in analysed_parameter_list],
            yl="Interbout interval (s)" if i_m == 0 else None,
            ymin=0, ymax=6,
            yticks=[0, 1.5, 3, 4.5, 6] if i_m == 0 else None
        )

        # Draw experimental vs. simulated interbout interval curves
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),
                         lc="k", lw=1, label="data" if i_m == 0 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),
                         lc="k", lw=1, line_dashes=(1, 2), label="simulation" if i_m == 0 else None)

        # Adjust spacing depending on model position
        pad = padding if i_m == len(models_in_loss_list) - 1 else padding_small
        xpos = xpos + pad + plot_width

    xpos = xpos_start
    ypos = ypos - padding - plot_height_here

# --------------------------------------------------------------------------
# Save final figure
# --------------------------------------------------------------------------
fig.save(path_save / "figure_s1_loss_validation.pdf", open_file=False, tight=style.page_tight)

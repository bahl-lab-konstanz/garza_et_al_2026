"""
====================================================================
Overview
====================================================================
This script generates behavioral analysis figures from zebrafish
decision-making experiments and model fitting.

It produces:
1. Decision variable trajectories simulated with a Drift Diffusion Model (DDM).
2. Repeatability plots of parameter estimates across model fitting runs.

Steps:
- Load experimental and synthetic datasets for selected fish.
- Configure plot styling (layout, dimensions, colors).
- Run DDM simulations and visualize trajectories with decision thresholds.
- For repeatability analysis: load multiple fitting results,
  plot parameter evolution across iterations, and compare final values.
- Save the final figure as a PDF.

====================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.model.core.params import Parameter, ParameterList
from rg_behavior_model.model.ddm import DDMstable
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import ConfigurationExperiment
from rg_behavior_model.utils.constants import StimulusParameterLabel

# ================================================================
# Environment and paths
# ================================================================
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])     # input data directory
path_save = Path(env['PATH_SAVE'])   # directory where figures will be saved
path_data = path_dir / 'base_dataset'  # location of dataset files


# ================================================================
# Plot configuration (layout, sizes, padding, etc.)
# ================================================================
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
padding_plot = style.padding_in_plot
padding_short = style.padding / 2
padding_vertical = plot_height_small

lw = style.linewidth
number_bins_hist = 15  # number of bins for histograms


# ================================================================
# Flags: toggle which plots are generated
# ================================================================
show_trajectory_decision_variable = False
show_repeatability = True


# ================================================================
# Experiment configuration
# ================================================================
time_start_simulation = 0   # simulation start time (s)
time_end_simulation = 30    # simulation end time (s)
time_duration_simulation = time_end_simulation - time_start_simulation
analysed_parameter = StimulusParameterLabel.COHERENCE.value
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'


# ================================================================
# Load experimental and synthetic data for each fish
# ================================================================
df_dict = {}
fish_to_include_list = ConfigurationExperiment.example_fish_list
config_list = [
    {"label": "data", "line_dashes": None, "alpha": 0.5, "color": None},
    {"label": "fit", "line_dashes": (2, 4), "alpha": 1, "color": "k"}
]

for i_fish, fish in enumerate(fish_to_include_list):
    # Load experimental behavioral data
    df_data = pd.read_hdf(path_data / f"data_fish_{fish}.hdf5")
    # Load first available synthetic dataset (model fit)
    for path_fit in path_data.glob(f"data_synthetic_fish_{fish}_*.hdf5"):
        df_fit = pd.read_hdf(path_fit)
        break
    # Store datasets and assign a unique color per fish
    df_dict[fish] = {
        "fit": df_fit,
        "data": df_data,
        "color": style.palette["fish_code"][i_fish]
    }
    # Optional preprocessing step (commented out)


# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()
plot_height_row = plot_height_small * 2 + padding_vertical


# ================================================================
# Plot 1: DDM decision variable trajectories
# ================================================================
if show_trajectory_decision_variable:
    x_show_start = 0
    x_show_end = 100
    plot_width_here = style.plot_width * 8
    plot_height_here = style.plot_height * 0.7

    # Simulate only for selected coherence values
    index_parameter_to_simulate = [1]
    simulated_parameters = [ConfigurationExperiment.coherence_list[i] for i in index_parameter_to_simulate]

    # Define DDM model parameters
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=ConfigurationDDM.dt))
    parameters.add_parameter("noise_sigma", Parameter(value=1))
    parameters.add_parameter("scaling_factor", Parameter(value=0.6))
    parameters.add_parameter("leak", Parameter(value=-1))
    parameters.add_parameter("residual_after_bout", Parameter(value=0.03))
    parameters.add_parameter("inactive_time", Parameter(value=0.1))
    parameters.add_parameter("threshold", Parameter(value=ConfigurationDDM.threshold))

    # Instantiate DDM model
    ddm_model = DDMstable(parameters, trials_per_simulation=200,
                          time_experimental_trial=30, scaling_factor_input=1)
    ddm_model.define_stimulus(
        time_start_stimulus=ConfigurationExperiment.time_start_stimulus,
        time_end_stimulus=ConfigurationExperiment.time_end_stimulus
    )

    # Time vector for simulations
    time_trial_list = np.arange(x_show_start, x_show_end + ConfigurationDDM.dt, ConfigurationDDM.dt)

    for index_parameter, parameter in enumerate(simulated_parameters):
        # Create subplot frame
        plot_n = fig.create_plot(
            plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos,
            plot_height=plot_height_here, plot_width=plot_width_here,
            xl="Simulation time (s)",
            xmin=x_show_start, xmax=x_show_end, xticks=None,
            ymin=-1 * ConfigurationDDM.threshold - 0.5, ymax=ConfigurationDDM.threshold + 0.5,
            yticks=[-ConfigurationDDM.threshold, ConfigurationDDM.threshold],
            yticklabels=["-B", "B"], hlines=[0]
        )
        # Draw decision thresholds
        plot_n.draw_line(time_trial_list, np.zeros_like(time_trial_list) - ConfigurationDDM.threshold, lc="k")
        plot_n.draw_line(time_trial_list, np.zeros_like(time_trial_list) + ConfigurationDDM.threshold, lc="k")

        # Add time scalebar
        scalebar_time = 5
        scalebar = time_trial_list[-int(scalebar_time / ConfigurationDDM.dt):]
        plot_n.draw_line(scalebar, np.zeros_like(scalebar) - ConfigurationDDM.threshold - 0.3, lc="k")
        plot_n.draw_text(x_show_end - scalebar_time, -ConfigurationDDM.threshold - 0.4,
                         f"{scalebar_time}s", textlabel_ha="left", textlabel_va="top")

        # Label coherence value
        plot_n.draw_line(np.linspace(x_show_start, x_show_end, len(time_trial_list)),
                         np.zeros_like(time_trial_list) + ConfigurationDDM.threshold + 0.3,
                         lc=style.palette["stimulus"][-1 - index_parameter_to_simulate[index_parameter]])
        plot_n.draw_text(x_show_end + 1, ConfigurationDDM.threshold + 0.2,
                         f"Coh={parameter}%", textlabel_ha="left")

        # Input signal proportional to coherence
        input_signal = np.zeros(len(time_trial_list)) + parameter / 100

        # Run simulation
        response_time_list, bout_decision_list, decision_time_list, internal_state_trajectory = ddm_model.simulate_trial(
            input_signal=input_signal
        )

        # Plot internal trajectories and decisions
        for i_d, decision_time in enumerate(decision_time_list):
            if decision_time > x_show_end:
                break
            index_time_decision = np.argwhere(time_trial_list <= x_show_end).flatten()
            # Internal state trajectory
            plot_n.draw_line(time_trial_list[index_time_decision],
                             internal_state_trajectory[index_time_decision],
                             lc=style.palette["neutral"][0], lw=0.1)
            # Decision markers
            if bout_decision_list[i_d] > 0:
                y = ConfigurationDDM.threshold
                color_dot = style.palette["correct_incorrect"][0]  # correct
            else:
                y = -ConfigurationDDM.threshold
                color_dot = style.palette["correct_incorrect"][1]  # incorrect
            plot_n.draw_scatter(decision_time, y, pc=color_dot, elw=0)

        ypos -= plot_height_here + padding * 2


# ================================================================
# Plot 2: Repeatability of parameter estimates
# ================================================================
if show_repeatability:
    # Configuration for repeatability analysis
    path_dir_repeat = Path(fr"{path_dir}\benchmark\test_repeatability")
    test_id = "006"
    plot_width_here = 0.7  # style.plot_width_small
    padding_here = style.padding_small
    palette_here = style.palette["default"]

    # Iterate over repeatability test runs
    for path in path_dir_repeat.glob(f"model_test_{test_id}_*.hdf5"):
        df_target = pd.read_hdf(path)

        # Plot each parameter trajectory
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            p_target = df_target[p["label"]]

            # Create subplot per parameter
            plot_p = fig.create_plot(
                plot_label=style.get_plot_label() if i_p == 0 else None,
                plot_title=f"{p['label_show'].capitalize()}",
                xpos=xpos, ypos=ypos,
                plot_height=plot_height, plot_width=plot_width_here,
                errorbar_area=False,
                ymin=p["min"], ymax=p["max"],
                yticks=[p["min"], p["mean"], p["max"]],
                yl=None, xl="Iteration",
                xmin=0.5, xmax=2.5, xticks=[1, 2], xticklabels=["0", "1500"],
                hlines=[df_target[p["label"]][0]]  # horizontal reference at initial value
            )
            xpos += (plot_height + padding_short)

            # Collect end-of-fit parameter values
            p_fit_end_list = []
            for path_fit in path_dir_repeat.glob(f"error_test_{test_id}_*_fit.hdf5"):
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


# ================================================================
# Save final figure
# ================================================================
fig.save(path_save / "figure_2.pdf", open_file=False, tight=style.page_tight)

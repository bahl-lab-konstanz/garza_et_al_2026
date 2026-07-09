"""
The present script produces the following figure (or figure panels) in Garza et al 2026:
- Extended Data Fig. 1a-b

Overview:
This script quantifies trial-to-trial variability in zebrafish behavioral
responses to motion coherence stimuli. For each coherence level, it computes
the coefficient of variation (CV) — the standard deviation expressed as a
percentage of the mean — for two behavioral readouts: percentage of correct
swims and interbout interval (IBI). The resulting CV curves are plotted
against stimulus coherence to show how response reliability changes with
stimulus strength.
"""

from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from figures.style import BehavioralModelStyle
from service.behavioral_processing import BehavioralProcessing
from service.figure_helper import Figure
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel

# ----------------------------------------------------------------------------
# Load environment variables (data and save paths)
# ----------------------------------------------------------------------------
# dotenv_values reads the project's .env file into a dict-like object,
# giving access to machine-specific input/output directory paths without
# hardcoding them in the script.
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
# Path to the pooled, all-fish HDF5 dataset for the 5 days-post-fertilization
# wild-type cohort used in this analysis.
path_data = path_dir / "base_dataset_5dpfWT" / "data_fish_all.hdf5"
path_save = Path(env['PATH_SAVE'])


# ----------------------------------------------------------------------------
# Plot style and layout configuration
# ----------------------------------------------------------------------------
style = BehavioralModelStyle()

# Starting grid coordinates for placing subplots on the figure canvas.
xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

# Plot dimensions, scaled relative to the base style defaults for this
# particular figure (larger main plot, with a smaller half-height variant
# defined but not used directly in this script).
plot_height = style.plot_height * 1.5
plot_height_small = plot_height / 2.5
plot_width = style.plot_width * 1.5

# Spacing constants: padding between plots, padding within a plot's margins,
# and a vertical spacing unit derived from the small plot height.
padding = style.padding
padding_plot = style.padding_in_plot
padding_vertical = plot_height_small

# Color palette pulled from the shared style definitions.
palette = style.palette["default"]
color_neutral = style.palette["neutral"][0]

# ----------------------------------------------------------------------------
# Initialize main figure container
# ----------------------------------------------------------------------------
fig = Figure()

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
df = pd.read_hdf(path_data)

# ----------------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------------
# Restrict bouts to the stimulus presentation window only (exclude bouts that
# start before or end after the defined stimulus onset/offset times), so that
# only stimulus-driven responses are analysed.
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'
df_filtered_all = df.query(query_time)
# Further restrict to only the coherence levels explicitly defined in the
# experiment configuration (drops any coherence values not part of the
# intended stimulus set, e.g. calibration or unused conditions).
df_filtered_all = df_filtered_all[df_filtered_all[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]

# ----------------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------------
# For each coherence level, compute the across-fish mean and standard
# deviation of "percentage correct swims" (the default analysed quantity).
# parameter_list holds the coherence values; correct_bout_list and
# std_correct_bout_list hold the corresponding mean and std arrays.
parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=StimulusParameterLabel.COHERENCE.value)
# Coefficient of variation (%) = std / mean * 100, quantifying relative
# variability in response accuracy across fish, for each coherence level.
coefficient_variation_accuracy = std_correct_bout_list / correct_bout_list * 100

# Repeat the same mean/std computation, but this time on the interbout
# interval (IBI) column instead of the default accuracy column, by passing
# column_name=ConfigurationExperiment.ResponseTimeColumn. Note the returned
# variables are reused/overwritten here to hold IBI statistics instead of
# accuracy statistics.
parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=StimulusParameterLabel.COHERENCE.value, column_name=ConfigurationExperiment.ResponseTimeColumn)
# Coefficient of variation (%) for interbout interval, per coherence level.
coefficient_variation_ibi = std_correct_bout_list / correct_bout_list * 100

# ----------------------------------------------------------------------------
# Log
# ----------------------------------------------------------------------------
# Print summary statistics to the console for quick inspection/debugging.
# Note: at this point correct_bout_list/std_correct_bout_list still refer to
# the IBI statistics (overwritten above), so the "percentage_correct" labels
# in the first three print statements actually report IBI values, not accuracy.
print(f"mean percentage_correct: {correct_bout_list}")
print(f"std percentage_correct: {std_correct_bout_list}")
print(f"CV percentage_correct: {coefficient_variation_accuracy}\n")
print(f"mean IBI: {correct_bout_list}")
print(f"std IBI: {std_correct_bout_list}")
print(f"CV IBI: {coefficient_variation_ibi}\n")

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# Create a single subplot showing coefficient of variation (%) as a function
# of stimulus coherence, with shared x-axis range/ticks taken from the
# experiment's configured coherence levels.
plot_cv = fig.create_plot(xpos=xpos, ypos=ypos,
                         plot_height=plot_height,
                         plot_width=plot_width,
                         errorbar_area=True,
                         xl=ConfigurationExperiment.coherence_label, xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                         xticks=[int(p) for p in ConfigurationExperiment.coherence_list], yl="Coefficient variation (%)",
                         ymin=0, ymax=100,
                         yticks=[0, 50, 100])
# Draw the accuracy CV curve as a black dashed line (fine dash pattern).
plot_cv.draw_line(x=ConfigurationExperiment.coherence_list, y=coefficient_variation_accuracy, lc="k", lw=1, line_dashes=(1, 2), label="Percentage correct swims")
# Draw the IBI CV curve as a black dotted line (different dash pattern) on
# the same axes, for direct visual comparison of the two behavioral measures.
plot_cv.draw_line(x=ConfigurationExperiment.coherence_list, y=coefficient_variation_ibi, lc="k", lw=1, line_dashes=(0.1, 3), label="Interbout interval")


# ----------------------------------------------------------------------------
# Save figure
# ----------------------------------------------------------------------------
fig.save(path_save / "figure_extended_data_1_cv.pdf", open_file=True, tight=style.page_tight)
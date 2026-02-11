from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from all_stimuli.Panos.dot_motion_stimuli_versions.utils.stimulus_parameter import StimulusParameterLabel
from figures.style import BehavioralModelStyle
from service.behavioral_processing import BehavioralProcessing
from service.figure_helper import Figure
from utils.configuration_experiment import ConfigurationExperiment

# =====================================================================
# Load environment variables (data and save paths)
# =====================================================================
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_data = path_dir / "base_dataset_5dpfWT" / "data_fish_all.hdf5"
path_save = Path(env['PATH_SAVE'])


# =============================================================================
# Plot style and layout configuration
# =============================================================================
style = BehavioralModelStyle(plot_label_i=4)

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5
plot_width = style.plot_width * 3/2

padding = style.padding
padding_plot = style.padding_in_plot
padding_vertical = plot_height_small

palette = style.palette["default"]
color_neutral = style.palette["neutral"][0]

# =====================================================================
# Initialize main figure container
# =====================================================================
fig = Figure()

# =====================================================================
# Load data
# =====================================================================
df = pd.read_hdf(path_data)

# =====================================================================
# Filter
# =====================================================================
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'
df_filtered_all = df.query(query_time)
df_filtered_all = df_filtered_all[df_filtered_all[StimulusParameterLabel.COHERENCE.value].isin(ConfigurationExperiment.coherence_list)]

# =====================================================================
# Computation
# =====================================================================
parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=StimulusParameterLabel.COHERENCE.value)
coefficient_variation_accuracy = std_correct_bout_list / correct_bout_list * 100

parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=StimulusParameterLabel.COHERENCE.value, column_name=ConfigurationExperiment.ResponseTimeColumn)
coefficient_variation_ibi = std_correct_bout_list / correct_bout_list * 100

# =====================================================================
# Log
# =====================================================================
print(f"mean percentage_correct: {correct_bout_list}")
print(f"std percentage_correct: {std_correct_bout_list}")
print(f"CV percentage_correct: {coefficient_variation_accuracy}\n")
print(f"mean IBI: {correct_bout_list}")
print(f"std IBI: {std_correct_bout_list}")
print(f"CV IBI: {coefficient_variation_ibi}\n")

# =====================================================================
# Plotting
# =====================================================================
plot_cv = fig.create_plot(xpos=xpos, ypos=ypos,
                         plot_height=plot_height,
                         plot_width=plot_width,
                         errorbar_area=True,
                         xl=ConfigurationExperiment.coherence_label, xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                         xticks=[int(p) for p in ConfigurationExperiment.coherence_list], yl="Coefficient variation (%)",
                         ymin=0, ymax=100,
                         yticks=[0, 50, 100])
plot_cv.draw_line(x=ConfigurationExperiment.coherence_list, y=coefficient_variation_accuracy, lc="k", lw=1, line_dashes=(1, 2), label="Percentage correct swims")
plot_cv.draw_line(x=ConfigurationExperiment.coherence_list, y=coefficient_variation_ibi, lc="k", lw=1, line_dashes=(0.1, 3), label="Interbout interval")


# =====================================================================
# Finalize and save figure
# =====================================================================
fig.save(path_save / "figure_extended_data_1_cv.pdf", open_file=True, tight=style.page_tight)

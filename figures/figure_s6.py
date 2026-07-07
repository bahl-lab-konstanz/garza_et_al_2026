"""
Psychometric curve analysis for synthetic DDM models across stimulus coherence.

This script generates (or loads) a synthetic behavioral dataset from DDM model
simulations filtered by leak sign ("leakneg-all" vs "leakpos-all"), then plots
the psychometric curve of accuracy (percentage correct swims) as a function of
stimulus coherence, for individual synthetic fish and their population average.

Workflow:
1. Load environment paths and select dataset variant based on model leak sign
   (select_dataset["sign"]/"label").
2. If save_dataset is True: aggregate per-fish synthetic HDF5 files (model + data)
   matching the selected leak sign into a single dataset and save it to disk;
   otherwise load the previously saved aggregated dataset.
3. Filter bouts to the stimulus time window and valid coherence levels
   (ConfigurationExperiment.coherence_list).
4. If show_psychometric_curve is True:
   a. Compute and plot per-fish accuracy vs. coherence curves (individual lines),
      highlighting a subset of "example" fish in a distinct color/label.
   b. Compute and plot the across-fish mean psychometric curve (with std and
      coefficient of variation logged to console).
   c. Log the average curve slope in the 50-100% coherence range as a summary
      metric of discrimination sensitivity.
5. Save the resulting figure as a PDF.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from figures.style import BehavioralModelStyle
from service.behavioral_processing import BehavioralProcessing
from service.figure_helper import Figure
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel

# =============================================================================
# Script configurations
# =============================================================================
save_dataset = False
show_psychometric_curve = True

# =============================================================================
# Environment and data paths
# =============================================================================
env_path = Path(__file__).parent.parent / ".env"
env = dotenv_values(env_path)
path_dir = Path(env['PATH_DIR'])
path_save = Path(env['PATH_SAVE'])
path_data = path_dir
select_dataset = {"sign": -1, "label": "leakneg-all"}  # ALTERNATIVE {"sign": 1, "label": "leakpos-all"}

# =============================================================================
# Plot style and layout configuration
# =============================================================================
style = BehavioralModelStyle(plot_label_i=4)

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_row = style.plot_height
plot_height_small = plot_height / 2.5
plot_width = style.plot_width * 3/2

padding = style.padding
padding_plot = style.padding_in_plot
padding_vertical = plot_height_small

palette = style.palette["default"]
color_neutral = style.palette["neutral"][0]

# =============================================================================
# Experimental parameters
# =============================================================================
coherence_label = StimulusParameterLabel.COHERENCE.value
analysed_parameter_label = "Coh (%)"
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

# =============================================================================
# Load and/or save dataset
# =============================================================================
if save_dataset:
    df_list = []
    for path_model in path_data.glob("model_test_*.hdf5"):
        if "fit" in path_model.name: continue
        id_fish = path_model.name.split("_")[2].replace(".hdf5", "")
        df_model = pd.read_hdf(path_model)
        if df_model["leak"][0] * select_dataset["sign"] < 0: continue
        for path_fish in path_data.glob(f"data_synthetic_test_{id_fish}_*.hdf5"):
            break
        df_list.append(pd.read_hdf(path_fish))
    df = pd.concat(df_list)
    df.to_hdf(str(path_save / f"data_synthetic_test_{select_dataset['label']}.hdf5"), key="all_events", complevel=9)
else:
    df = pd.read_hdf(path_data / f"data_synthetic_test_{select_dataset['label']}.hdf5")

# =============================================================================
# Initialize main figure container
# =============================================================================
fig = Figure()

# =============================================================================
# Computation and plotting
# =============================================================================
if show_psychometric_curve:
    plot_height = plot_height_row
    plot_width = 1
    m_list = []

    # --- Data preparation ---
    df_filtered_all = df.query(query_time)
    df_filtered_all = df_filtered_all[df_filtered_all[coherence_label].isin(ConfigurationExperiment.coherence_list)]
    id_fish_list = list(df_filtered_all["fish_ID"].unique())

    # Compute mean accuracy per coherence level
    parameter_list_all, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
        df_filtered_all, analysed_parameter=coherence_label)
    parameter_list_all = np.array([int(p) for p in parameter_list_all])

    # --- Plot setup ---
    line_dashes = None
    plot_0 = fig.create_plot(plot_label=style.get_plot_label(),
                             xpos=xpos, ypos=ypos,
                             plot_height=plot_height,
                             plot_width=plot_width,
                             xmin=min(parameter_list_all), xmax=max(parameter_list_all),
                             xticks=parameter_list_all, xl=ConfigurationExperiment.coherence_label,
                             yl="Percentage\ncorrect swims (%)",
                             ymin=0, ymax=100,
                             yticks=[0, 50, 100], hlines=[50])

    # Plot individual fish curves
    for i_id, id in enumerate(id_fish_list):
        df_fish = df_filtered_all[df_filtered_all["fish_ID"] == id]
        df_fish_filtered = df_fish[df_fish[coherence_label].isin(ConfigurationExperiment.coherence_list)]
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=coherence_label)
        correct_bout_list *= 100

        # Example fish highlighted, others in gray
        if id in ConfigurationExperiment.example_fish_list:
            color_line = "gray"
            lw = 1
            show_label = True
        else:
            color_line = color_neutral
            lw = 0.05
            show_label = False
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_line, lw=lw, alpha=0.8)
        m_list.append((correct_bout_list[-1]-correct_bout_list[-2])/(parameter_list[-1]-parameter_list[-2]))
        if show_label:
            plot_0.draw_text(max(parameter_list_all) + 0.1, correct_bout_list[-1], f"fish {id}",
                             textlabel_rotation='horizontal', textlabel_ha='left')

    # Mean psychometric curve across fish
    parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=coherence_label)
    coefficient_variation_accuracy = std_correct_bout_list / correct_bout_list * 100
    correct_bout_list *= 100
    print(f"mean percentage_correct: {correct_bout_list}")
    print(f"std percentage_correct: {std_correct_bout_list}")
    print(f"CV percentage_correct: {coefficient_variation_accuracy}")

    plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc="k", lw=1, line_dashes=line_dashes)

    ypos = ypos
    xpos = xpos + padding + plot_width

    print(f"Average slope in range 50-100%: {np.mean(np.abs(m_list))}")

# =============================================================================
# Save final figure
# =============================================================================
fig.save(path_save / f"figure_s6_{select_dataset['label']}.pdf", open_file=False, tight=style.page_tight)
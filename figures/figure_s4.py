"""
Script Overview
---------------
This script generates behavioral analysis figures comparing experimental fish data
to synthetic (model-generated) fish data. It processes raw datasets, computes
statistics (e.g., accuracy over time, bout number vs. accuracy, reaction time
effects, and direction consistency), and visualizes results in multiple plots.
Statistical tests (e.g., Wilcoxon signed-rank test, Cohen's d effect size) are
performed where relevant. The final figure is exported as a PDF.

Key steps:
1. Load experimental and synthetic datasets.
2. Preprocess data (remove invalid/fast bouts, compute derived metrics).
3. Generate figures for:
   - Accuracy over time
   - Accuracy vs. bout number
   - First-bout reaction time vs. accuracy
   - Bout direction consistency vs. interbout interval
4. Save the composite figure as "figure_s2.pdf".
"""

import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import dotenv_values
from scipy.stats import wilcoxon

from analysis.utils.figure_helper import Figure
from garza_et_al_2026.figures.style import BehavioralModelStyle
from garza_et_al_2026.service.behavioral_processing import BehavioralProcessing
from garza_et_al_2026.service.statistics_service import StatisticsService
from garza_et_al_2026.utils.configuration_experiment import ConfigurationExperiment
from garza_et_al_2026.utils.constants import StimulusParameterLabel

# ==============================
# Environment and data paths
# ==============================

# Load environment variables from .env file
env = dotenv_values()

# Base directories (input, output, dataset)
path_dir = Path(env['PATH_DIR'])       # input data directory
path_save = Path(env['PATH_SAVE'])     # directory where figures will be saved
path_data = path_dir / 'base_dataset_5dpfWT'  # dataset folder

# Input datasets
path_data_experimental = fr"{path_data}\data_fish_all.hdf5"         # experimental fish data
path_data_model = fr"{path_data}\data_synthetic_fish_all.hdf5"      # synthetic fish data (model-generated)

# ==============================
# Figure style and layout
# ==============================

# Load default style (colors, layout, etc.)
style = BehavioralModelStyle()

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_width = style.plot_width
plot_width_big = style.plot_width * 2
padding = style.padding

# ==============================
# Plot toggles (enable/disable panels)
# ==============================

show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_time_vs_accuracy = True
show_bout_number_vs_percentage_correct = True
show_first_bout_rt_vs_accuracy = True
show_same_bout_dir_vs_interbout_interval = True

# ==============================
# Experimental parameters
# ==============================

# Stimulus parameter to analyze (here: coherence)
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = ConfigurationExperiment.coherence_list

# ==============================
# Data loading and preprocessing
# ==============================

# Load experimental data
df_experimental = pd.read_hdf(path_data_experimental)

# Remove "fast straight bouts" (responses < 100ms)
df_experimental = BehavioralProcessing.remove_fast_straight_bout(df_experimental, threshold_response_time=100)

# Load synthetic/model data
df_synthetic = pd.read_hdf(path_data_model)

# Query string for selecting time window during stimulus presentation
query_time = (f'start_time > {ConfigurationExperiment.time_start_stimulus} and '
              f'end_time < {ConfigurationExperiment.time_end_stimulus}')

# ==============================
# Figure setup
# ==============================

# Create a figure container for all subplots
fig = Figure()

# ------------------------------
# Panel 1: Time vs Accuracy
# ------------------------------
if show_time_vs_accuracy:
    for i_df, df in enumerate((df_experimental, df_synthetic)):
        # Create subplot for accuracy over trial time
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_df == 0 else None,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height, plot_width=plot_width_big,
            errorbar_area=False,
            xl="Time (s)", xmin=0, xmax=ConfigurationExperiment.time_experimental_trial,
            xticks=[0, ConfigurationExperiment.time_start_stimulus,
                    ConfigurationExperiment.time_end_stimulus,
                    ConfigurationExperiment.time_experimental_trial] if i_df == 1 else None,
            yl="Percentage\ncorrect swims (%)", ymin=45, ymax=100,
            yticks=[50, 100], hlines=[50],
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        # Remove straight bouts (correct_bout = -1)
        df = df[df['correct_bout'] != -1]
        coherence_list = np.sort(df_synthetic[analysed_parameter].unique())

        # Compute and plot moving-window accuracy for each coherence level
        for i_coh, coherence in enumerate(coherence_list):
            windowed_data, time_stamp_list, std_list = BehavioralProcessing.windowing_column(
                df[df[analysed_parameter] == coherence],
                ConfigurationExperiment.CorrectBoutColumn,
                window_size=2.5,
                window_step_size=2.5,
                window_operation='mean_multiple_fish',
            )
            index_subset = range(0, len(time_stamp_list))

            # Plot mean ± std of accuracy
            plot_0.draw_line(
                x=time_stamp_list[index_subset] + 2.5,
                y=windowed_data[index_subset] * 100,
                yerr=np.array(std_list[index_subset]) * 100,
                lc=style.palette["stimulus"][-i_coh - 1],
                lw=0.75
            )

        ypos = ypos_start - padding - plot_height

    # Shift to next column for following panels
    ypos = ypos_start
    xpos = xpos + plot_width_big + padding

# ------------------------------
# Panel 2: Bout Number vs Accuracy
# ------------------------------
if show_bout_number_vs_percentage_correct:
    number_of_bouts_to_plot = 3
    bout_index_list = [int(i) for i in list(range(number_of_bouts_to_plot))]
    x_ticks = [i + 1 for i in bout_index_list]

    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df_filtered = df[df['correct_bout'] != -1]

        # Two subplots: accuracy after stim start vs after stim end
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_df == 0 else None,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height,
            plot_width=plot_width,
            errorbar_area=False,
            xl="Bout number\nafter stimulus\nstart" if i_df == 1 else None,
            xmin=min(x_ticks), xmax=max(x_ticks),
            xticks=x_ticks if i_df == 1 else None,
            yl="Percentage\ncorrect swims (%)",
            ymin=45, ymax=100, yticks=[50, 100],
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        plot_1 = fig.create_plot(
            xpos=xpos + padding, ypos=ypos,
            plot_height=plot_height,
            plot_width=plot_width * (number_of_bouts_to_plot - 1) / number_of_bouts_to_plot,
            errorbar_area=False,
            xl="Bout number\nafter stimulus\nend" if i_df == 1 else None,
            xmin=min(x_ticks), xmax=max(x_ticks) - 1,
            xticks=x_ticks[:-1] if i_df == 1 else None,
            ymin=45, ymax=100,
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        # Loop over stimulus parameters (e.g. coherence levels)
        y_star = 80
        for i_p, parameter in enumerate(analysed_parameter_list):

            # Helper: count number of unique trials
            def count_trials(df):
                return len(df.index.unique("trial_count_since_experiment_start"))

            if i_df == 0:  # print trial counts for experimental data
                number_trials_per_fish = df_filtered[df_filtered[analysed_parameter] == parameter] \
                    .groupby(by=["folder_name"]).apply(count_trials)
                print(fr"COH: {parameter} | # trials: {np.mean(number_trials_per_fish)}$\pm${np.std(number_trials_per_fish)}")

            # --- Accuracy relative to stimulus start ---
            accuracy_bout_dict_start = BehavioralProcessing.accuracy_bout_in_trial(
                df_filtered[df_filtered[analysed_parameter] == parameter],
                bout_index_list,
                time_window_start=ConfigurationExperiment.time_start_stimulus
            )
            accuracy_bout_list_start_mean = np.zeros(number_of_bouts_to_plot)
            accuracy_bout_list_start_std = np.zeros(number_of_bouts_to_plot)

            # Compare accuracy between successive bouts using Wilcoxon + Cohen's d
            for index, bout_list in accuracy_bout_dict_start.items():
                if index != 0:
                    label = "DATA" if i_df == 0 else "SIMULATION"
                    stat, p = wilcoxon(bout_list, bout_list_pre)
                    cohen_d = StatisticsService.cohen_d(bout_list, bout_list_pre)

                    print(f"{label} | after stim START | coherence: {parameter} | "
                          f"bout {x_ticks[index-1]} VS {x_ticks[index]} | "
                          f"p-value: {p:.04f}, Cohen's d: {cohen_d:.04f}")

                    # Add significance marker if p < 0.05
                    if p is not None and p < 0.05:
                        plot_0.draw_scatter(
                            x=[np.mean(x_ticks[index-1:index+1])],
                            y=[y_star], pt="*",
                            ec=style.palette["stimulus"][-i_p - 1],
                            pc=style.palette["stimulus"][-i_p - 1]
                        )
                bout_list_pre = bout_list

                # Mean ± std accuracy per bout
                accuracy_bout_list_start_mean[index] = np.nanmean(bout_list) * 100
                accuracy_bout_list_start_std[index] = np.nanstd(bout_list) * 100 / len(bout_list)

            # --- Accuracy relative to stimulus end ---
            accuracy_bout_dict_end = BehavioralProcessing.accuracy_bout_in_trial(
                df_filtered[df_filtered[analysed_parameter] == parameter],
                bout_index_list[:-1],
                time_window_start=ConfigurationExperiment.time_end_stimulus
            )
            accuracy_bout_list_end_mean = np.zeros(number_of_bouts_to_plot - 1)
            accuracy_bout_list_end_std = np.zeros(number_of_bouts_to_plot - 1)

            for index, bout_list in accuracy_bout_dict_end.items():
                if index != 0:
                    stat, p = wilcoxon(bout_list, bout_list_pre)
                    cohen_d = StatisticsService.cohen_d(bout_list, bout_list_pre)

                    label = "DATA" if i_df == 0 else "SIMULATION"
                    print(f"{label} | after stim END | coherence: {parameter} | "
                          f"bout {x_ticks[index-1]} VS {x_ticks[index]} | "
                          f"p-value: {p:.04f}, Cohen's d: {cohen_d:.04f}")

                    # Add significance marker if p < 0.05
                    if p is not None and p < 0.05:
                        plot_1.draw_scatter(
                            x=[np.mean(x_ticks[index-1:index+1])],
                            y=[y_star], pt="*",
                            ec=style.palette["stimulus"][-i_p - 1],
                            pc=style.palette["stimulus"][-i_p - 1]
                        )
                bout_list_pre = bout_list

                accuracy_bout_list_end_mean[index] = np.nanmean(bout_list) * 100
                accuracy_bout_list_end_std[index] = np.nanstd(bout_list) * 100 / len(bout_list)

            # Plot accuracy trajectories
            plot_0.draw_line(x=x_ticks, y=accuracy_bout_list_start_mean,
                             lc=style.palette["stimulus"][-i_p - 1], lw=0.75,
                             yerr=accuracy_bout_list_start_std)
            plot_1.draw_line(x=x_ticks[:-1], y=accuracy_bout_list_end_mean,
                             lc=style.palette["stimulus"][-i_p - 1], lw=0.75,
                             yerr=accuracy_bout_list_end_std)

            y_star += 5  # shift for significance markers

        ypos = ypos_start - padding - plot_height

    ypos = ypos_start
    xpos = xpos + 2 * padding + 2 * plot_width

# ------------------------------
# Panel 3: First-Bout RT vs Accuracy
# ------------------------------
if show_first_bout_rt_vs_accuracy:
    number_points = 3
    dt = 0.4  # time bin size
    time_list_start = [ConfigurationExperiment.time_start_stimulus + dt * i for i in range(number_points + 1)]
    time_list_end = [ConfigurationExperiment.time_end_stimulus + dt * i for i in range(number_points + 1)]

    plot_height = 1
    plot_width = 1

    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df = df[df[ConfigurationExperiment.CorrectBoutColumn] != -1]

        # Subplot: accuracy vs RT (after stim start)
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_df == 0 else None,
            xpos=xpos, ypos=ypos,
            plot_height=plot_height, plot_width=plot_width,
            errorbar_area=False,
            xl="Time after\nstimulus start (s)", xmin=min(time_list_start),
            xmax=max(time_list_start) + dt / 2,
            yl="Percentage\ncorrect swims (%)", ymin=45, ymax=100,
            yticks=[50, 100], hlines=[50],
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        # Subplot: accuracy vs RT (after stim end)
        plot_1 = fig.create_plot(
            xpos=xpos + padding, ypos=ypos,
            plot_height=plot_height, plot_width=plot_width,
            errorbar_area=False,
            xl="Time after\nstimulus end (s)", xmin=min(time_list_end),
            xmax=max(time_list_end) + dt / 2,
            ymin=45, ymax=100, hlines=[50],
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        # Compute accuracy per fish per time bin
        for i_p, parameter in enumerate(analysed_parameter_list):
            df_p = df[df[analysed_parameter] == parameter]
            acc_start = np.zeros(number_points)
            acc_start_std = np.zeros(number_points)
            acc_end = np.zeros(number_points)
            acc_end_std = np.zeros(number_points)

            # Fish identifiers differ between data and simulation
            if i_df == 0:
                fish_id_list = df_p.index.unique('folder_name')
            else:
                fish_id_list = df_p['fish_ID'].unique()

            for i_t in range(len(time_list_start) - 1):
                df_p_t_start = df_p.query(f"start_time > {time_list_start[i_t]} and start_time < {time_list_start[i_t+1]}")
                df_p_t_end = df_p.query(f"start_time > {time_list_end[i_t]} and start_time < {time_list_end[i_t+1]}")

                fish_accuracy_start_list = []
                fish_accuracy_end_list = []
                for fish_id in fish_id_list:
                    try:
                        if i_df == 0:  # experimental data grouped by folder/trial
                            df_p_t_start_fish = df_p_t_start.xs(fish_id, level='folder_name') \
                                .groupby(level="trial_count_since_experiment_start").first()
                            df_p_t_end_fish = df_p_t_end.xs(fish_id, level='folder_name') \
                                .groupby(level="trial_count_since_experiment_start").first()
                        else:  # synthetic data grouped by fish_ID/trial
                            df_p_t_start_fish = df_p_t_start[df_p_t_start['fish_ID'] == fish_id] \
                                .groupby(by="trial").first()
                            df_p_t_end_fish = df_p_t_end[df_p_t_end['fish_ID'] == fish_id] \
                                .groupby(by="trial").first()

                        fish_accuracy_start_list.append(np.nanmean(df_p_t_start_fish[ConfigurationExperiment.CorrectBoutColumn]))
                        fish_accuracy_end_list.append(np.nanmean(df_p_t_end_fish[ConfigurationExperiment.CorrectBoutColumn]))
                    except KeyError:
                        pass

                acc_start[i_t] = np.mean(fish_accuracy_start_list) * 100
                acc_start_std[i_t] = np.std(fish_accuracy_start_list) / len(fish_accuracy_start_list)
                acc_end[i_t] = np.mean(fish_accuracy_end_list) * 100
                acc_end_std[i_t] = np.std(fish_accuracy_end_list) / len(fish_accuracy_end_list)

            # Plot results
            plot_1.draw_line(x=time_list_end[-2:], y=np.ones(2) * 47, lc="k")
            plot_1.draw_text(time_list_end[-2], 35, f"{dt}s")

            plot_0.draw_line(x=time_list_start[1:], y=acc_start, yerr=acc_start_std,
                             lc=style.palette["stimulus"][-i_p - 1], lw=0.75)
            plot_1.draw_line(x=time_list_end[1:], y=acc_end, yerr=acc_end_std,
                             lc=style.palette["stimulus"][-i_p - 1], lw=0.75)

        ypos = ypos_start - padding - plot_height

    ypos = ypos_start
    xpos = xpos + 2 * padding + 2 * plot_width

# ------------------------------
# Panel 4: Same Bout Direction vs Interbout Interval
# ------------------------------
if show_same_bout_dir_vs_interbout_interval:
    dt = 0.4  # time bin size

    # Compute "same_direction_as_previous_bout" for synthetic data
    def compute_same_as_previous(df):
        same_as_previous = [np.nan]
        df_temp = df.reset_index(allow_duplicates=True)
        same_as_previous.extend([
            1 if df_temp.loc[i_row, ConfigurationExperiment.CorrectBoutColumn] ==
                 df_temp.loc[i_row - 1, ConfigurationExperiment.CorrectBoutColumn]
            else 0
            for i_row in range(1, len(df_temp))
        ])
        df["same_direction_as_previous_bout"] = same_as_previous
        return df

    df_synthetic = df_synthetic.groupby(by=["fish_ID", "trial"]).apply(
        compute_same_as_previous, include_groups=False
    )

    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df_filtered = df.query(query_time)

        # Subplot: probability of repeating same direction vs interbout interval
        plot_0 = fig.create_plot(
            plot_label=style.get_plot_label() if i_df == 0 else None,
            xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
            xmin=0, xmax=1.2,
            xl="Time from\nlast bout (s)", yl="Probability to swim\nin same direction (%)",
            ymin=45, ymax=100, yticks=[50, 100], hlines=[50],
            vspans=[[ConfigurationExperiment.time_start_stimulus,
                     ConfigurationExperiment.time_end_stimulus, "gray", 0.1]]
        )

        try:
            for i_p, parameter in enumerate(analysed_parameter_list):
                # Compute probability in time bins
                windowed_data, time_stamp_list, std_list = BehavioralProcessing.windowing_column(
                    df_filtered[df_filtered[analysed_parameter] == parameter],
                    'same_direction_as_previous_bout',
                    window_size=dt, window_step_size=dt,
                    window_operation='mean_multiple_fish',
                    time_column=ConfigurationExperiment.ResponseTimeColumn,
                    time_start=0, time_end=1.2
                )

                # Plot mean ± std
                plot_0.draw_line(x=time_stamp_list, y=windowed_data * 100,
                                 yerr=std_list * 100,
                                 lc=style.palette["stimulus"][-i_p - 1], lw=0.75)

            # Time bin marker
            plot_0.draw_line(x=time_stamp_list[-2:], y=np.ones(2) * 47, lc="k")
            plot_0.draw_text(time_stamp_list[-2], 35, f"{dt}s")
        except KeyError:
            pass

        ypos = ypos_start - padding - plot_height

    ypos = ypos_start
    xpos = xpos + padding + plot_width

# ==============================
# Save final figure
# ==============================
fig.save(path_save / "figure_s4.pdf", open_file=False, tight=style.page_tight)

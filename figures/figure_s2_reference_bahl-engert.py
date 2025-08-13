import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values
from scipy.stats import mannwhitneyu, normaltest, ttest_ind, ttest_rel, wilcoxon

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.constants import alphabet
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, \
    palette_1, ResponseTimeColumn
from analysis.utils.figure_helper import Figure

# env
env = dotenv_values()
path_data = Path(env['PATH_DATA'])
path_data_model = Path(env['PATH_DATA_MODEL'])
path_dir = Path()

# parameters plot
style = BehavioralModelStyle()
xpos_start = 0.5
ypos_start = 0.5
xpos = xpos_start
ypos = ypos_start
padding=2
i_plot_label = 0
plot_label_list = alphabet
plot_height = 1
plot_width = 2

show_psychometric_curve = False
show_coherence_vs_interbout_interval = False
show_time_vs_accuracy = True
show_bout_number_vs_percentage_correct = True
show_first_bout_rt_vs_accuracy = True
show_same_bout_dir_vs_interbout_interval = True

# parameters_experiment
time_start_stimulus = 10  # 10  # seconds
time_end_stimulus = 40  #40  # seconds
time_experimental_trial = 50  # seconds
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = (0, 25, 50, 100)

df_experimental = pd.read_hdf(path_data)
df_experimental = BehavioralProcessing.remove_fast_straight_bout(df_experimental, threshold_response_time=100)
df_synthetic = pd.read_hdf(path_data_model)
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

# Make a standard figure
fig = Figure()

if show_time_vs_accuracy:
    plot_height = 1
    plot_width = 2
    for i_df, df in enumerate((df_experimental, df_synthetic)):
        plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_df == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                 errorbar_area=False,
                                 xl="Time (s)", xmin=0, xmax=time_experimental_trial,
                                 xticks=[0, time_start_stimulus, time_end_stimulus, time_experimental_trial] if i_df == 1 else None,
                                 yl="Percentage\ncorrect swims (%)", ymin=45, ymax=100,
                                 yticks=[50, 100], hlines=[50],
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])

        # computation
        # df['correct_bout'] = BehavioralProcessing.compute_correct_bout_list(df, side_threshold=3, inverted_stimulus=False)
        df = df[df['correct_bout'] != -1]
        df_synthetic_filter = df_synthetic
        coherence_list = np.sort(df_synthetic[analysed_parameter].unique())
        for i_coh, coherence in enumerate(coherence_list):
            windowed_data, time_stamp_list, std_list = BehavioralProcessing.windowing_column(
                df[df[analysed_parameter] == coherence],
                CorrectBoutColumn,
                window_size=2.5,
                window_step_size=2.5,
                window_operation='mean_multiple_fish',
            )
            index_subset = range(0, len(time_stamp_list))

            # plot
            plot_0.draw_line(x=time_stamp_list[index_subset]+2.5, y=windowed_data[index_subset]*100,
                             yerr=np.array(std_list[index_subset])*100, lc=style.palette["stimulus"][-i_coh-1], lw=0.75)

        ypos = ypos_start - padding - plot_height

    i_plot_label += 1
    ypos = ypos_start
    xpos = xpos + padding + plot_width

if show_bout_number_vs_percentage_correct:
    plot_height = 1
    plot_width = 1
    number_of_bouts_to_plot = 3
    bout_index_list = [int(i) for i in list(range(number_of_bouts_to_plot))]
    x_ticks = [i + 1 for i in bout_index_list]
    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df_filtered = df[df['correct_bout'] != -1]

        # plot
        plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_df == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=False,
                                 xl="Bout number\nafter stimulus\nstart" if i_df == 1 else None, xmin=min(x_ticks), xmax=max(x_ticks),
                                 xticks=x_ticks if i_df == 1 else None,
                                 yl="Percentage\ncorrect swims (%)",
                                 ymin=45, ymax=100, yticks=[50, 100],
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])

        plot_1 = fig.create_plot(xpos=xpos + padding, ypos=ypos,
                                 plot_height=plot_height,
                                 plot_width=plot_width*(number_of_bouts_to_plot-1)/number_of_bouts_to_plot,
                                 errorbar_area=False,
                                 xl="Bout number\nafter stimulus\nend" if i_df == 1 else None, xmin=min(x_ticks), xmax=max(x_ticks)-1,
                                 xticks=x_ticks[:-1] if i_df == 1 else None,
                                 ymin=45, ymax=100,
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])

        # computation
        y_star = 80
        for i_p, parameter in enumerate(analysed_parameter_list):

            def count_trials(df):
                return len(df.index.unique("trial_count_since_experiment_start"))
            if i_df == 0:
                number_trials_per_fish = df_filtered[df_filtered[analysed_parameter] == parameter].groupby(by=["folder_name"]).apply(count_trials)
                print(fr"COH: {parameter} | # trials: {np.mean(number_trials_per_fish)}$\pm${np.std(number_trials_per_fish)}")

            accuracy_bout_dict_start = BehavioralProcessing.accuracy_bout_in_trial(
                df_filtered[df_filtered[analysed_parameter] == parameter],
                bout_index_list,
                time_window_start=time_start_stimulus
            )
            accuracy_bout_list_start_mean = np.zeros(number_of_bouts_to_plot)
            accuracy_bout_list_start_std = np.zeros(number_of_bouts_to_plot)
            for index, bout_list in accuracy_bout_dict_start.items():
                if index != 0:
                    label = "DATA" if i_df == 0 else "SIMULATION"
                    # _, p_norm_0 = normaltest(bout_list)
                    # _, p_norm_1 = normaltest(bout_list_pre)
                    # if p_norm_0 > 0.05 and p_norm_1 > 0.05:
                    #     print(f"{label} | after stim START | coherence: {parameter} | bout {x_ticks[index-1]} VS {x_ticks[index]} | NORMAL")
                    #     stat, p = ttest_ind(bout_list, bout_list_pre)
                    # else:
                    #     stat, p = mannwhitneyu(bout_list, bout_list_pre)
                    stat, p = wilcoxon(bout_list, bout_list_pre)
                    cohen_d = StatisticsService.cohen_d(bout_list, bout_list_pre)

                    print(f"{label} | after stim START | coherence: {parameter} | bout {x_ticks[index-1]} VS {x_ticks[index]} | p-value: {p:.04f}, Cohen's d: {cohen_d:.04f}")
                    if p is not None and p < 0.05:
                        plot_0.draw_scatter(x=[np.mean(x_ticks[index-1:index+1])], y=[y_star], pt="*", ec=style.palette["stimulus"][-i_p-1], pc=style.palette["stimulus"][-i_p-1])
                else:
                    p = None
                bout_list_pre = bout_list

                # bout_list = [np.sign(bout - 1) + 1 for bout in bout_list]  # map from {-1, 1} to {0, 1}
                accuracy_bout_list_start_mean[index] = np.nanmean(bout_list) * 100
                accuracy_bout_list_start_std[index] = np.nanstd(bout_list) * 100 / len(bout_list)

            accuracy_bout_dict_end = BehavioralProcessing.accuracy_bout_in_trial(
                df_filtered[df_filtered[analysed_parameter] == parameter],
                bout_index_list[:-1],
                time_window_start=time_end_stimulus
            )
            accuracy_bout_list_end_mean = np.zeros(number_of_bouts_to_plot-1)
            accuracy_bout_list_end_std = np.zeros(number_of_bouts_to_plot-1)
            for index, bout_list in accuracy_bout_dict_end.items():
                if index != 0:
                    stat, p = wilcoxon(bout_list, bout_list_pre)
                    cohen_d = StatisticsService.cohen_d(bout_list, bout_list_pre)

                    label = "DATA" if i_df == 0 else "SIMULATION"
                    print(f"{label} | after stim END | coherence: {parameter} | bout {x_ticks[index-1]} VS {x_ticks[index]} | p-value: {p:.04f}, Cohen's d: {cohen_d:.04f}")
                    if p is not None and p < 0.05:
                        plot_1.draw_scatter(x=[np.mean(x_ticks[index-1:index+1])], y=[y_star], pt="*", ec=style.palette["stimulus"][-i_p-1], pc=style.palette["stimulus"][-i_p-1])
                else:
                    p = None
                bout_list_pre = bout_list

                # bout_list = [np.sign(bout - 1) + 1 for bout in bout_list]  # map from {-1, 1} to {0, 1}
                accuracy_bout_list_end_mean[index] = np.nanmean(bout_list) * 100
                accuracy_bout_list_end_std[index] = np.nanstd(bout_list) * 100 / len(bout_list)

            plot_0.draw_line(x=x_ticks, y=accuracy_bout_list_start_mean, lc=style.palette["stimulus"][-i_p-1], lw=0.75, yerr=accuracy_bout_list_start_std)
            plot_1.draw_line(x=x_ticks[:-1], y=accuracy_bout_list_end_mean, lc=style.palette["stimulus"][-i_p-1], lw=0.75, yerr=accuracy_bout_list_end_std)

            y_star += 5

        ypos = ypos_start - padding - plot_height

    i_plot_label += 1
    ypos = ypos_start
    xpos = xpos + 2 * padding + 2 * plot_width

if show_first_bout_rt_vs_accuracy:
    number_points = 3
    dt = 0.4
    time_list_start = [time_start_stimulus + dt * i for i in range(number_points + 1)]
    time_list_end = [time_end_stimulus + dt * i for i in range(number_points + 1)]

    plot_height = 1
    plot_width = 1
    properties_to_group_by_in_df_data = ("folder_name", "trial_count_since_experiment_start", "stimulus_name")
    properties_to_group_by_in_df_simulation = ("fish_ID", "trial")
    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df = df[df[CorrectBoutColumn] != -1]
        plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_df == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width,
                                 errorbar_area=False,
                                 xl="Time after\nstimulus start (s)", xmin=min(time_list_start), xmax=max(time_list_start)+dt/2, xticks=None,  # np.arange(min(time_list_start)+dt/2, max(time_list_start), dt),
                                 # xticklabels=[f"{i}" for i in range(1, number_points + 1)],
                                 yl="Percentage\ncorrect swims (%)", ymin=45, ymax=100, yticks=[50, 100], hlines=[50],
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])

        plot_1 = fig.create_plot(xpos=xpos+padding, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width,
                                 errorbar_area=False,
                                 xl="Time after\nstimulus end (s)", xmin=min(time_list_end), xmax=max(time_list_end)+dt/2, xticks=None,  # np.arange(min(time_list_end)+dt/2, max(time_list_end), dt),
                                 # xticklabels=[f"{i}" for i in range(1, number_points + 1)],
                                 ymin=45, ymax=100, hlines=[50],
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])

        # computation
        # parameter_list = np.sort(df_synthetic[analysed_parameter].unique())
        for i_p, parameter in enumerate(analysed_parameter_list):
            df_p = df[df[analysed_parameter] == parameter]
            acc_start = np.zeros(number_points)
            acc_start_std = np.zeros(number_points)
            acc_end = np.zeros(number_points)
            acc_end_std = np.zeros(number_points)

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
                        if i_df == 0:
                            df_p_t_start_fish = df_p_t_start.xs(fish_id, level='folder_name')
                            df_p_t_start_fish = df_p_t_start_fish.groupby(level="trial_count_since_experiment_start").first()

                            df_p_t_end_fish = df_p_t_end.xs(fish_id, level='folder_name')
                            df_p_t_end_fish = df_p_t_end_fish.groupby(level="trial_count_since_experiment_start").first()
                        else:
                            df_p_t_start_fish = df_p_t_start[df_p_t_start['fish_ID'] == fish_id]
                            df_p_t_start_fish = df_p_t_start_fish.groupby(by="trial").first()

                            df_p_t_end_fish = df_p_t_end[df_p_t_end['fish_ID'] == fish_id]
                            df_p_t_end_fish = df_p_t_end_fish.groupby(by="trial").first()

                        fish_accuracy_start_list.append(np.nanmean(df_p_t_start_fish[CorrectBoutColumn]))
                        fish_accuracy_end_list.append(np.nanmean(df_p_t_end_fish[CorrectBoutColumn]))
                    except KeyError:
                        pass

                acc_start[i_t] = np.mean(fish_accuracy_start_list) * 100
                acc_start_std[i_t] = np.std(fish_accuracy_start_list) / len(fish_accuracy_start_list)
                acc_end[i_t] = np.mean(fish_accuracy_end_list) * 100
                acc_end_std[i_t] = np.std(fish_accuracy_end_list) / len(fish_accuracy_end_list)

                # if i_df == 0:
                #     acc_start[i_t] = np.mean(df_p_t_start.groupby(level=properties_to_group_by_in_df_data).agg({CorrectBoutColumn: "mean"})[CorrectBoutColumn]) * 100
                #     acc_start_std[i_t] = np.mean(df_p_t_start.groupby(level=properties_to_group_by_in_df_data).agg({CorrectBoutColumn: "std"})[CorrectBoutColumn]) * 100
                #     acc_end[i_t] = np.mean(df_p_t_end.groupby(level=properties_to_group_by_in_df_data).agg({CorrectBoutColumn: "mean"})[CorrectBoutColumn]) * 100
                #     acc_end_std[i_t] = np.mean(df_p_t_end.groupby(level=properties_to_group_by_in_df_data).agg({CorrectBoutColumn: "std"})[CorrectBoutColumn]) * 100
                #
                # else:
                #     correct_bout_list = []
                #     for t in df_p_t_start["trial"].unique():
                #         correct_bout_list.append(np.mean(df_p_t_start[df_p_t_start["trial"] == t][CorrectBoutColumn]))
                #     acc_start[i_t] = np.mean(correct_bout_list) * 100
                #     acc_start_std[i_t] = np.std(correct_bout_list) * 100
                #
                #     correct_bout_list = []
                #     for t in df_p_t_end["trial"].unique():
                #         correct_bout_list.append(np.mean(df_p_t_end[df_p_t_end["trial"] == t][CorrectBoutColumn]))
                #     acc_end[i_t] = np.mean(correct_bout_list) * 100
                #     acc_end_std[i_t] = np.std(correct_bout_list) * 100

            # plot
            # plot_0.draw_line(x=time_list_start[-2:], y=np.ones(2)*90, lc="k")
            # plot_0.draw_text(time_list_start[-2], 85, f"{dt}s")
            plot_1.draw_line(x=time_list_end[-2:], y=np.ones(2)*47, lc="k")
            plot_1.draw_text(time_list_end[-2], 35, f"{dt}s")

            plot_0.draw_line(x=time_list_start[1:], y=acc_start, yerr=acc_start_std,
                             lc=style.palette["stimulus"][-i_p-1], lw=0.75)

            plot_1.draw_line(x=time_list_end[1:], y=acc_end, yerr=acc_end_std,
                             lc=style.palette["stimulus"][-i_p-1], lw=0.75)

        ypos = ypos_start - padding - plot_height

    i_plot_label += 1
    ypos = ypos_start
    xpos = xpos + 2 * padding + 2 * plot_width

if show_same_bout_dir_vs_interbout_interval:
    plot_height = 1
    plot_width = 1
    dt = 0.4

    # compute column same_direction_as_previous_bout for synthetic data
    def compute_same_as_previous(df):
        same_as_previous = [np.nan]
        df_temp = df.reset_index(allow_duplicates=True)
        same_as_previous.extend([1 if df_temp.loc[i_row, CorrectBoutColumn] == df_temp.loc[i_row - 1, CorrectBoutColumn] else 0
                                 for i_row in range(1, len(df_temp))])
        df["same_direction_as_previous_bout"] = same_as_previous
        return df
    df_synthetic = df_synthetic.groupby(by=["fish_ID", "trial"]).apply(compute_same_as_previous, include_groups=False)

    for i_df, df in enumerate((df_experimental, df_synthetic)):
        df_filtered = df.query(query_time)

        # plot
        plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_df == 0 else None,
                                 xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                 xmin=0, xmax=1.2,
                                 xl="Time from\nlast bout (s)", yl="Probability to swim\nin same direction (%)",
                                 ymin=45, ymax=100, yticks=[50, 100], hlines=[50],
                                 vspans=[[time_start_stimulus, time_end_stimulus, "gray", 0.1]])
        try:
            for i_p, parameter in enumerate(analysed_parameter_list):
                # computation
                windowed_data, time_stamp_list, std_list = BehavioralProcessing.windowing_column(
                    df_filtered[df_filtered[analysed_parameter] == parameter],
                    'same_direction_as_previous_bout',
                    window_size=dt, window_step_size=dt,
                    window_operation='mean_multiple_fish',
                    time_column=ResponseTimeColumn, time_start=0, time_end=1.2)

                plot_0.draw_line(x=time_stamp_list, y=windowed_data * 100, yerr=std_list*100, lc=style.palette["stimulus"][-i_p-1], lw=0.75)

            plot_0.draw_line(x=time_stamp_list[-2:], y=np.ones(2) * 47, lc="k")
            plot_0.draw_text(time_stamp_list[-2], 35, f"{dt}s")
        except KeyError:
            pass

        i_plot_label += 1
        ypos = ypos_start - padding - plot_height

    ypos = ypos_start
    xpos = xpos + padding + plot_width

fig.save(Path.home() / 'Desktop' / f"figure_s2_reference_bahl-engert_reset_nonzero.pdf", open_file=True, tight=True)
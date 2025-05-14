import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, \
    ResponseTimeColumn, alphabet, Keyword
from analysis_helpers.analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis_helpers.analysis.utils.figure_helper import Figure

# env
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])

# parameters plot
plot_height = 1
plot_height_small = plot_height / 2.5
padding_plot = 0.5
padding_vertical = plot_height_small
i_plot_label = 1  # 6  # 0  #
plot_label_list = alphabet
color_neutral = "#bfbfbfff"

style = BehavioralModelStyle()
xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start
padding = 1.5  # style.padding
plot_height = style.plot_height
plot_width = style.plot_width * 3/2
letter_counter = 1
style.add_palette("green", Palette.green_short)
style.add_palette("neutral", [Palette.color_neutral])
palette = style.palette["default"]
style.add_palette("fish_code", ["#73489C", "#753B51", "#103882", "#7F0C0C"])


# start
show_trial_structure = True
show_distribution_change_angles = True
show_fish_trajectory = False

# single-fish analysis is necessary
show_rt_distributions = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_interbout_interval_vs_accuracy = False
show_time_vs_accuracy = False
show_bout_number_vs_percentage_correct = False
show_frequency_vs_amplitude_accuracy = False
show_time_vs_accuracy_dynamics = False


# parameters_experiment
analysed_parameter_list = [0, 25, 50, 100]
time_start_stimulus = 10  # 10  # seconds
time_end_stimulus = 40  # seconds
time_experimental_trial = 50  # seconds
number_individuals = 16
analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
analysed_parameter_label = "coherence [%]"  # "period [s]"  #
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

df_dict = {}
all_label = "all"
fish_0_label = "201"
fish_1_label = "205"
fish_2_label = "506"
# fish_3_label = "504"
fish_to_include_list = [fish_0_label, fish_1_label, fish_2_label, all_label]
for fish in fish_to_include_list:
    df = pd.read_hdf(path_dir / f"data_fish_{fish}.hdf5")
    df_dict[fish] = df  # BehavioralProcessing.remove_fast_straight_bout(df, threshold_response_time=100)

# Make a standard figure
fig = Figure()

if show_trial_structure:
    color_exp_1 = style.palette["stimulus"]
    experiment_list = [
        {"analysed_parameter": StimulusParameterLabel.COHERENCE.value,
         "analysed_parameter_list": [0, 25, 50, 100],
         "analysed_parameter_label": "C",
         "unit": "%",
         "input_function": lambda t, coh: coh/100,
         "palette": color_exp_1},
        # {"analysed_parameter": StimulusParameterLabel.PERIOD.value,
        #  "analysed_parameter_list": [10, 7.5, 5, 6],
        #  "analysed_parameter_label": "T",
        #  "unit": "s",
        #  "input_function": lambda t, T: (0.5 * (np.sin((2.0 * np.pi / T) * (t-time_start_stimulus) - np.pi / 2.0) + 1)),
        #  "palette": Palette.red_short},
    ]

    # parameters
    dt = 0.01
    time_stimulus = np.arange(time_start_stimulus, time_end_stimulus+2*dt, dt)
    time_rest = np.arange(dt, time_start_stimulus, dt)
    x_limits = [0, 3]  # None  #
    y_limits = [0, 0.2]  #
    number_individuals = 1

    plot_height_here = 0.2
    padding_vert_here = 0.05
    plot_width_here = 2
    ypos_here = ypos + 0.5

    number_plots = 0
    height_drift = 0
    for i_ex, ex in enumerate(experiment_list):

        for i_param, parameter in enumerate(ex["analysed_parameter_list"]):
            plot_section = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_param == 0 else None,
                                           xpos=xpos, ypos=ypos_here - (padding_vert_here + plot_height_here) * number_plots - height_drift,
                                           plot_height=plot_height_here,
                                           plot_width=plot_width_here,
                                           xmin=0, xmax=time_experimental_trial,
                                           ymin=-0.1, ymax=1.1,
                                           legend_xpos=xpos + plot_width_here,
                                           legend_ypos=ypos_here + 0.3 - (padding_vert_here + plot_height_here) * number_plots - height_drift)
            # if i_ex == len(experiment_list)-1 and i_param == len(ex["analysed_parameter_list"])-1:
            #     plot_section = fig.create_plot(xpos=xpos, ypos=ypos - (padding_vert + plot_height) * number_plots,
            #                                    plot_height=plot_height,
            #                                    plot_width=plot_width,
            #                                    xl="time [s]", xmin=x_limits[0], xmax=x_limits[-1],
            #                                    xticks=[0, time_start_stimulus, time_end_stimulus,
            #                                            time_experimental_trial],
            #                                    yl="coherence [-]", yticks=[0, 100],
            #                                    # ymin=y_limits[0], ymax=y_limits[-1],
            #                                    legend_xpos=xpos + padding + plot_width,
            #                                    legend_ypos=-(padding_vert + plot_height) * number_plots)
            # else:

            number_plots += 1
            trial_stimulus = np.zeros(len(time_stimulus))
            for i_time, time in enumerate(time_stimulus):
                if time > time_start_stimulus and time < time_end_stimulus:
                    trial_stimulus[i_time] = ex["input_function"](time, parameter)
            color = ex["palette"][-1-i_param]
            plot_section.draw_line(time_rest, np.zeros(len(time_rest)), lc="black")
            plot_section.draw_line(time_rest + time_end_stimulus, np.zeros(len(time_rest)), lc="black")
            plot_section.draw_line(time_stimulus, trial_stimulus, lc=color,
                                   label=f"{ex['analysed_parameter_label']} = {int(parameter)}{ex['unit']}")
        height_drift += 0.2
    # plot_section = fig.create_plot(xpos=xpos, ypos=ypos - (padding_vert_here/2 + plot_height_here) * number_plots - height_drift,
    #                                plot_height=plot_height_here,
    #                                plot_width=plot_width_here,
    #                                xl="time [s]", xticks=[0, time_start_stimulus, time_end_stimulus, time_experimental_trial],
    #                                xmin=0, xmax=time_experimental_trial,
    #                                ymin=0.1, ymax=1.1)
    # plot_section.draw_line(time_rest, np.zeros(len(time_rest)), lc="black")

    i_plot_label += 1
    ypos = ypos
    xpos = xpos + 1.5*plot_width + padding

if show_distribution_change_angles:
    # parameters
    palette = Palette.green_short
    angle_list_length = 40  # 100
    max_angle = 90  # 200
    min_angle = -90  # -200
    threshold_side = 3
    inverted_direction = True
    show_gaussian_mixture_model = False
    wrong_direction_label = True
    df = df_dict[all_label]
    color_list = style.palette["stimulus"]
    analysed_parameter_list = [0, 25, 50, 100]

    plot_height_here = 1
    plot_width_here = 1

    # plotting
    ymax = 20
    plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height_here,
                             plot_width=plot_width_here, errorbar_area=False, xmin=min_angle, xmax=max_angle, ymin=0, ymax=ymax,
                             xticks=[min_angle, 0, max_angle], xl='orientation change [°]',
                             yticks=[0, ymax], yl='percentage swims [%]',
                             vlines=[0],
                             vspans=[[min_angle, -threshold_side, Palette.correct_incorrect[1], 0.25], [threshold_side, max_angle, Palette.correct_incorrect[0], 0.25]])
    i_plot_label += 1

    if wrong_direction_label:
        for i_row, row in df.iterrows():
            if 'right' == row[StimulusParameterLabel.DIRECTION.value]:
                df.loc[i_row, StimulusParameterLabel.DIRECTION.value] = Keyword.RIGHT.value  # initally inverted
            elif 'left' == row[StimulusParameterLabel.DIRECTION.value]:
                df.loc[i_row, StimulusParameterLabel.DIRECTION.value] = Keyword.LEFT.value  # initally inverted
    query_time = f"start_time > {time_start_stimulus} and end_time < {time_end_stimulus} " \
                 f"and estimated_orientation_change < {max_angle} and estimated_orientation_change > {min_angle}"  # slice over time and angle
    df_filtered = df.query(query_time)
    coherence_list = analysed_parameter_list  # np.sort(df[analysed_parameter].unique())
    legends = []
    for i_coherence, coherence in enumerate(coherence_list):
        coherence_condition_is_met = list(df_filtered[StimulusParameterLabel.COHERENCE.value] == coherence)
        df_coh = df_filtered[coherence_condition_is_met]
        df_coh['flipped_estimated_orientation_change'] = BehavioralProcessing.flip_column(df_coh, 'estimated_orientation_change', inverted_direction=inverted_direction)
        if len(df_coh['flipped_estimated_orientation_change'].dropna()) == 0:
            print(f"WARNING | plotting | distribution of angle changes | no values found for coherence {coherence}")
        else:
            angle_list = np.array(df_coh['flipped_estimated_orientation_change'].dropna())

            angle_start_list, angle_center_list, angle_end_list = StatisticsService.get_window_list_from_data(
                angle_list,
                window_number=angle_list_length,
                center=0
            )
            angle_distribution = np.zeros(len(angle_center_list))
            for angle_index in range(len(angle_center_list)):
                angle_value = len([angle for angle in df_coh['flipped_estimated_orientation_change'] if angle_start_list[angle_index] <= angle <= angle_end_list[angle_index]]) / len(df_coh['flipped_estimated_orientation_change'])
                angle_distribution[angle_index] = angle_value
            plot_0.draw_line(angle_center_list, angle_distribution * 100, lc=color_list[-1-i_coherence],
                             label=f"{coherence}%" if coherence != -1 else 'rest')

    ypos = ypos
    xpos = xpos + plot_width_here + padding

if show_fish_trajectory:
    plot_height_here = 2
    plot_width_here = 2

    # circle
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - 0.6

    # parameters
    index_trial = np.arange(4, 5)
    df = df_dict["08"]
    try:
        fish_id = df.index.unique('experiment_ID')[0]
    except KeyError:
        fish_id = np.array(df['experiment_ID'])[0]
    try:
        trial_id_list = df.index.unique('trial')[index_trial]
    except KeyError:
        trial_id_list = np.array(df['trial'])[index_trial]
    try:
        stimulus_id = df.index.unique('stimulus_name')[-3]
    except KeyError:
        stimulus_id = np.array(df[StimulusParameterLabel.COHERENCE.value])[-1]
    x_position_column = 'end_x_position'
    y_position_column = 'end_y_position'
    time_column = 'end_time_absolute'

    plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height_here,
                             plot_width=plot_width_here, errorbar_area=False, xmin=-1, xmax=1, ymin=-1, ymax=1)
    plot_0.ax.contour(X, Y, F, [0], colors="k")
    i_plot_label += 1

    # define temporary dataframe
    for trial_id in trial_id_list:
        df_filtered = df[df[CorrectBoutColumn] != -1]
        try:
            df_filtered = df_filtered.xs(fish_id, level='experiment_ID')
        except KeyError:
            df_filtered = df_filtered[df_filtered['experiment_ID'] == fish_id]
        try:
            df_filtered = df_filtered.xs(trial_id, level='trial')
        except KeyError:
            df_filtered = df_filtered[df_filtered['trial'] == trial_id]
        try:
            df_filtered = df_filtered.xs(stimulus_id, level='stimulus_name')
        except KeyError:
            df_filtered = df_filtered[df_filtered[StimulusParameterLabel.COHERENCE.value] == stimulus_id]

        # plotting
        plot_0.draw_line(df_filtered[x_position_column], df_filtered[y_position_column], lc="gray", alpha=0.50)
        color_list = []
        for i, row in df_filtered.iterrows():
            if row[CorrectBoutColumn] == 1:
                color_list.append(Palette.correct_incorrect[0])
                # color_list.append("#66F7FF")
            else:
                color_list.append(Palette.correct_incorrect[1])
                # color_list.append("#FF03C8")
        plot_0.draw_scatter(df_filtered[x_position_column], df_filtered[y_position_column], pc=color_list, ec=color_list, alpha=0.5)
    # plt.scatter([-2], [-2], s=150, c="#66F7FF", cmap="Spectral_r", alpha=0.5, label="correct swims")
    # plt.scatter([-2], [-2], s=150, c="#FF03C8", cmap="Spectral_r", alpha=0.5, label="incorrect swims")
    # plt.title(f"trajectory fish {fish_id} trial {trial_id}")
    # plt.legend(loc='lower right')
    # plt.show()

    ypos = ypos - plot_height_here - padding
    xpos = xpos_start

plot_height_row = plot_height_small * 2 + padding_vertical

xpos = xpos_start
ypos = ypos - padding - plot_height
if show_psychometric_curve:
    plot_height = plot_height_row
    plot_width = 1

    # fetch
    df = df_dict[all_label]
    df_filtered_all = df.query(query_time)
    df_filtered_all = df_filtered_all[df_filtered_all[analysed_parameter].isin(analysed_parameter_list)]
    id_fish_list = list(df_filtered_all.index.unique("experiment_ID"))

    # computation
    parameter_list_all, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
        df_filtered_all, analysed_parameter=analysed_parameter)
    parameter_list_all = np.array([int(p) for p in parameter_list_all])

    # plot
    line_dashes = None
    plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=plot_width,
                             xmin=min(parameter_list_all), xmax=max(parameter_list_all),
                             # xl=analysed_parameter_label,
                             xticks=None,  # xticks=[int(p) for p in parameter_list_all],
                             yl="accuracy [-]",
                             ymin=0, ymax=1,
                             yticks=[0, 0.5, 1], hlines=[0.5])
    i_plot_label += 1


    for i_id, id in enumerate(id_fish_list):
        df_fish = df_filtered_all.xs(id, level="experiment_ID")
        df_fish_filtered = df_fish[df_fish[analysed_parameter].isin(analysed_parameter_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=analysed_parameter)

        # plot
        if id == int(fish_0_label):
            color_line = style.palette["fish_code"][0]
            lw = 0.3
        elif id == int(fish_1_label):
            color_line = style.palette["fish_code"][1]
            lw = 0.3
        elif id == int(fish_2_label):
            color_line = style.palette["fish_code"][2]
            lw = 0.3
        # elif id == int(fish_3_label):
        #     color_line = style.palette["fish_code"][3]
        #     lw = 0.3
        else:
            color_line = color_neutral
            lw = 0.05
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_line, lw=lw, alpha=0.8)
        # try:
        #     plot_0.draw_line(x=parameter_list, y=correct_bout_list, lw=0.3, alpha=0.8, lc=Palette.arlecchino[i_id], label=id)
        # except IndexError:
        #     pass

    # mean curve
    parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
        df_filtered_all, analysed_parameter=analysed_parameter)
    plot_0.draw_line(x=parameter_list, y=correct_bout_list,
                     # errorbar_area=True, yerr=np.array(std_correct_bout_list) / number_individuals,
                     lc="k", lw=1, line_dashes=line_dashes)

    # ypos = ypos - padding - plot_height
    # xpos = xpos_start
    ypos = ypos
    xpos = xpos + padding + plot_width

xpos = xpos_start
ypos = ypos - padding - plot_height
if show_coherence_vs_interbout_interval:
    plot_height = plot_height_row
    plot_width = 1

    df = df_dict["all"]
    df_filtered = df.query(query_time)
    df_filtered = df_filtered[df_filtered[analysed_parameter].isin(analysed_parameter_list)]
    # computation
    parameter_list, interbout_interval_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(df_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)
    parameter_list = np.array([int(p) for p in parameter_list])

    # plot
    line_dashes = None
    plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos,
                             plot_height=plot_height,
                             plot_width=plot_width,
                             errorbar_area=True,
                             xl=analysed_parameter_label, xmin=min(parameter_list), xmax=max(parameter_list),
                             xticks=[int(p) for p in parameter_list], yl="interbout interval [s]",
                             ymin=0, ymax=2,
                             yticks=[0, 1, 2])
    i_plot_label += 1

    for i_id, id in enumerate(id_fish_list):
        df_fish = df_filtered_all.xs(id, level="experiment_ID")
        df_fish_filtered = df_fish[df_fish[analysed_parameter].isin(analysed_parameter_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)

        # plot
        if id == int(fish_0_label):
            color_line = style.palette["fish_code"][0]
            lw = 0.3
        elif id == int(fish_1_label):
            color_line = style.palette["fish_code"][1]
            lw = 0.3
        elif id == int(fish_2_label):
            color_line = style.palette["fish_code"][2]
            lw = 0.3
        # elif id == int(fish_3_label):
        #     color_line = style.palette["fish_code"][3]
        #     lw = 0.3
        else:
            color_line = color_neutral
            lw = 0.05
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_line, lw=lw, alpha=0.8)
        # try:
        #     plot_0.draw_line(x=parameter_list, y=correct_bout_list, lw=0.3, alpha=0.8, lc=Palette.arlecchino[i_id], label=id)
        # except IndexError:
        #     pass

    # plot
    parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
        df_filtered_all, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)
    plot_0.draw_line(x=parameter_list, y=interbout_interval_list,
                     # errorbar_area=True, yerr=std_correct_bout_list / number_individuals,
                     lc="k", lw=1, line_dashes=line_dashes)

    ypos = ypos
    xpos = xpos + padding + plot_width


xpos = xpos_start
ypos = ypos - padding - plot_height
if show_rt_distributions:
    # parameters
    padding_here = plot_height_small * 2
    palette = style.palette["stimulus"]
    x_limits = [0, 2]  # None  #
    y_limits = [0, 0.17]

    plot_width_here = 0.7

    plot_section = {"corr": {}, "err": {}}

    df_dict.pop(all_label, None)  # get rid of the df with all data merged

    for i_k, k in enumerate(df_dict.keys()):
        # if k == fish_0_label:
        #     line_dashes = None
        #     number_individuals = 1
        # else:
        #     line_dashes = (1, 2)
        #     number_individuals = 1

        df = df_dict[k]
        # id_list = np.sort(np.array(df.index.unique(level='experiment_ID')))
        # number_individuals = len(id_list)
        for i_param, parameter in enumerate(analysed_parameter_list):
            # plot
            line_dashes = None
            plot_title = f"coh={parameter}%" if i_k == 0 else None
            plot_section["corr"][i_param] = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_k == 0 and i_param == 0 else None,
                                                            plot_title=plot_title,
                                                            xpos=xpos + i_param * (plot_width_here + padding_plot),
                                                            ypos=ypos + plot_height_small + padding_vertical,
                                                            plot_height=plot_height_small,
                                                            plot_width=plot_width_here,
                                                            xmin=x_limits[0], xmax=x_limits[-1],
                                                            xticks=None, yticks=None,
                                                            ymin=-y_limits[-1], ymax=y_limits[-1],
                                                            hlines=[0])
            plot_section["err"][i_param] = plot_section["corr"][i_param]

            if i_param == len(analysed_parameter_list) - 1 and i_k == len(df_dict)-1:
                location_scalebar = y_limits[-1] / 6
                plot_section["corr"][i_param].draw_line((2, 2), (location_scalebar, location_scalebar + 0.1), lc="k")
                plot_section["corr"][i_param].draw_text(2 + 0.2, location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

            # if i_k == 0:
                # if i_param == 0:
                #     plot_section["corr"][i_param] = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos + plot_height_small + padding_vertical,
                #                                         plot_height=plot_height_small,
                #                                         plot_width=plot_width_here,
                #                                         xmin=x_limits[0], xmax=x_limits[-1], xticks=[],
                #                                         yl="correct",
                #                                         ymin=y_limits[0], ymax=y_limits[-1], yticks=[0, 0.15])
                #                      # ,legend_xpos=xpos_start-1.5 * plot_width, legend_ypos=ypos_start)
                #     plot_section["err"][i_param] = fig.create_plot(xpos=xpos, ypos=ypos,
                #                                        plot_height=plot_height_small,
                #                                        plot_width=plot_width_here,
                #                                        xl="interbout interval [s]",
                #                                        xmin=x_limits[0], xmax=x_limits[-1], xticks=[0, 1, 2],
                #                                        yl="incorrect",
                #                                        ymin=y_limits[0], ymax=y_limits[-1], yticks=[0, 0.15])
                #     plot_section["err"][i_param].draw_text(-3, 0.2, "activity [events/s]", textlabel_rotation='vertical', textlabel_ha='center')
                # else:
                #     plot_section["corr"][i_param] = fig.create_plot(xpos=xpos + i_param * (plot_width_here + padding_plot), ypos=ypos + plot_height_small + padding_vertical,
                #                                         plot_height=plot_height_small,
                #                                         plot_width=plot_width_here,
                #                                         xmin=x_limits[0], xmax=x_limits[-1], xticks=[],
                #                                         ymin=y_limits[0], ymax=y_limits[-1])
                #     plot_section["err"][i_param] = fig.create_plot(xpos=xpos + i_param * (plot_width_here + padding_plot), ypos=ypos,
                #                                        plot_height=plot_height_small,
                #                                        plot_width=plot_width_here,
                #                                        xmin=x_limits[0], xmax=x_limits[-1], xticks=[0, 1, 2],
                #                                        ymin=y_limits[0], ymax=y_limits[-1])
                #

            plot_section_corr = plot_section["corr"][i_param]
            plot_section_err = plot_section["err"][i_param]

            df_filtered = df[df[analysed_parameter] == parameter]
            df_filtered = df_filtered.query(query_time)

            duration = np.sum(
                BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=time_end_stimulus-time_start_stimulus)
            ) * number_individuals

            # plot distribution of data over coherence levels
            data_corr = df_filtered[df_filtered[CorrectBoutColumn] == 1][ResponseTimeColumn]
            data_err = df_filtered[df_filtered[CorrectBoutColumn] == 0][ResponseTimeColumn]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   # bins=100,
                                                                                   bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                   duration=duration,
                                                                                   center_bin=True)
            index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
            data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
            data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 # bins=100,
                                                                                 bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                 duration=duration,
                                                                                 center_bin=True)
            index_in_limits = np.argwhere(
                    np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
            data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
            data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

            plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                        lw=0.75, line_dashes=line_dashes)
            plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                        lw=0.75, line_dashes=line_dashes)

        ypos = ypos - (padding_here + plot_height_small)
    i_plot_label += 1
    plot_height = 1
    ypos = ypos
    xpos = xpos + i_param * (plot_width_here + padding_plot) + plot_width_here + padding

# fig.save(Path.home() / 'Academics' / 'graphics' / 'pictures' / 'figures_for_papers' / 'behavior_model' / "figure_1_experiment_coh_temp.pdf", open_file=True, tight=True)
fig.save(Path.home() / 'Desktop' / "figure_1.pdf", open_file=True, tight=style.page_tight)

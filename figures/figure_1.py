import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values


from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.service.behavioral_processing import BehavioralProcessing
from rg_behavior_model.utils.configuration_ddm import dt
from rg_behavior_model.utils.configuration_experiment import time_start_stimulus, time_end_stimulus, \
    time_experimental_trial, StimulusParameterLabel, Keyword, ResponseTimeColumn, CorrectBoutColumn
from rg_modeling_framework.src.service.statistics_service import StatisticsService

# env
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_save = Path(env['PATH_SAVE'])

# configuration plot
style = BehavioralModelStyle()

xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5
plot_width = style.plot_width * 3/2

padding = style.padding * 3/4
padding_plot = 0.5
padding_vertical = plot_height_small

palette = style.palette["default"]
color_neutral = style.palette["neutral"][0]

# show plots
show_trial_structure = True
show_distribution_change_angles = True
show_fish_trajectory = False
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_interbout_interval_vs_accuracy = False
show_rt_distributions = True
show_cv = False


# parameters_experiment
coherence_list = [0, 25, 50, 100]
coherence_label = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
analysed_parameter_label = "Coh (%)"
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

df_dict = {}
all_label = "all"
fish_0_label = "001"
fish_1_label = "002"
fish_2_label = "003"
# fish_3_label = "504"
fish_to_include_list = [fish_0_label, fish_1_label, fish_2_label, all_label]
for fish in fish_to_include_list:
    df = pd.read_hdf(path_dir / f"data_fish_{fish}.hdf5")
    df_dict[fish] = df  # BehavioralProcessing.remove_fast_straight_bout(df, threshold_response_time=100)

# Make a standard figure
fig = Figure()

if show_trial_structure:
    experiment_list = [
        {"analysed_parameter": StimulusParameterLabel.COHERENCE.value,
         "analysed_parameter_list": [0, 25, 50, 100],
         "analysed_parameter_label": "Coh",
         "unit": "%",
         "input_function": lambda t, coh: coh/100,
         "palette": style.palette["stimulus"],
         "trial_number": 6
         },
    ]

    # parameters
    df_sample = df_dict[fish_0_label]
    time_stimulus = np.arange(time_start_stimulus, time_end_stimulus+2*dt, dt)
    time_rest = np.arange(dt, time_start_stimulus, dt)

    plot_height_here = plot_height * 0.2
    padding_vert_here = plot_height * 0.05
    plot_width_here = plot_width * 2
    ypos_here = ypos + 0.5

    number_plots = 0
    height_drift = 0
    for i_ex, ex in enumerate(experiment_list):

        for i_param, parameter in enumerate(ex["analysed_parameter_list"]):
            plot_section = fig.create_plot(plot_label=style.get_plot_label() if i_param == 0 else None,
                                           xpos=xpos, ypos=ypos_here - (padding_vert_here + plot_height_here) * number_plots - height_drift,
                                           plot_height=plot_height_here,
                                           plot_width=plot_width_here,
                                           xmin=0, xmax=time_experimental_trial,
                                           ymin=-0.1, ymax=1.1,
                                           legend_xpos=xpos + plot_width_here,
                                           legend_ypos=ypos_here + 0.3 - (padding_vert_here + plot_height_here) * number_plots - height_drift)

            df_filtered = df_sample[df_sample[coherence_label] == parameter]
            df_filtered = df_filtered[df_filtered["dir"] == 90].xs(ex["trial_number"], level="trial")
            decision_time_list = np.array(df_filtered["start_time"])

            trial_stimulus = np.zeros(len(time_stimulus))
            for i_time, time in enumerate(time_stimulus):
                if time > time_start_stimulus and time < time_end_stimulus:
                    trial_stimulus[i_time] = ex["input_function"](time, parameter)
            color = ex["palette"][i_param]

            plot_section.draw_line(time_rest, np.zeros(len(time_rest)), lc="black")
            plot_section.draw_line(time_rest + time_end_stimulus, np.zeros(len(time_rest)), lc="black")
            plot_section.draw_line(time_stimulus, trial_stimulus, lc=color,
                                   label=f"{ex['analysed_parameter_label']} = {int(parameter)}{ex['unit']}")

            plot_section = fig.create_plot(xpos=xpos, ypos=ypos_here - (padding_vert_here + plot_height_here) * number_plots - height_drift,
                                           plot_height=plot_height_here,
                                           plot_width=plot_width_here,
                                           xmin=0, xmax=time_experimental_trial,
                                           ymin=0, ymax=1)

            def color_timestamp(timestamp):
                if timestamp < time_stimulus[0] or timestamp > time_stimulus[-1]:
                    return "black"
                else:
                    return color
            color_scatter = [color_timestamp(t) for t in decision_time_list]
            plot_section.draw_scatter(decision_time_list, 0.5+np.zeros(len(decision_time_list)), pt="|", ec=color_scatter, pc=color_scatter)

        height_drift += 0.2

    ypos = ypos
    xpos = xpos + 2 * plot_width_here + padding

if show_distribution_change_angles:
    # parameters
    angle_list_length = 40
    max_angle = 90
    min_angle = -90
    threshold_side = 3
    inverted_direction = True
    df = df_dict[all_label]

    plot_height_here = style.plot_height * 1.5
    plot_width_here = style.plot_width * 1.5

    # plotting
    ymax = 20
    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=plot_height_here,
                             plot_width=plot_width_here, errorbar_area=False, xmin=min_angle, xmax=max_angle, ymin=0, ymax=ymax,
                             xticks=[min_angle, 0, max_angle], xl='Orientation change (°)',
                             yticks=[0, ymax], yl='Percentage swims (%)',
                             vlines=[0],
                             vspans=[[min_angle, -threshold_side, style.palette["correct_incorrect"][1], 1],
                                     [threshold_side, max_angle, style.palette["correct_incorrect"][0], 1]])

    query_time = f"start_time > {time_start_stimulus} and end_time < {time_end_stimulus} " \
                 f"and estimated_orientation_change < {max_angle} and estimated_orientation_change > {min_angle}"  # slice over time and angle
    df_filtered = df.query(query_time)
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
            plot_0.draw_line(angle_center_list, angle_distribution * 100, lc=style.palette["stimulus"][i_coherence],
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

    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=plot_height_here,
                             plot_width=plot_width_here, errorbar_area=False, xmin=-1, xmax=1, ymin=-1, ymax=1)
    plot_0.ax.contour(X, Y, F, [0], colors="k")

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
                color_list.append(style.palette["correct_incorrect"][0])
            else:
                color_list.append(style.palette["correct_incorrect"][1])
        plot_0.draw_scatter(df_filtered[x_position_column], df_filtered[y_position_column], pc=color_list, ec=color_list, alpha=0.5)

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
    df_filtered_all = df_filtered_all[df_filtered_all[coherence_label].isin(coherence_list)]
    id_fish_list = list(df_filtered_all.index.unique("experiment_ID"))

    # computation
    parameter_list_all, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
        df_filtered_all, analysed_parameter=coherence_label)
    parameter_list_all = np.array([int(p) for p in parameter_list_all])

    # plot
    line_dashes = None
    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=plot_width,
                             xmin=min(parameter_list_all), xmax=max(parameter_list_all),
                             xticks=None,
                             yl="Percentage\ncorrect swims (%)",
                             ymin=45, ymax=100,
                             yticks=[50, 75, 100], hlines=[50])

    for i_id, id in enumerate(id_fish_list):
        df_fish = df_filtered_all.xs(id, level="experiment_ID")
        df_fish_filtered = df_fish[df_fish[coherence_label].isin(coherence_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=coherence_label)
        correct_bout_list *= 100

        # plot
        if id == int(fish_0_label):
            color_line = "gray"
            lw = 1
            show_label = True
        elif id == int(fish_1_label):
            color_line = "gray"
            lw = 1
            show_label = True
        elif id == int(fish_2_label):
            color_line = "gray"
            lw = 1
            show_label = True
        else:
            color_line = color_neutral
            lw = 0.05
            show_label = False
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_line, lw=lw, alpha=0.8)
        if show_label:
            plot_0.draw_text(max(parameter_list_all) + 0.1, correct_bout_list[-1], f"fish {id}",
                             textlabel_rotation='horizontal', textlabel_ha='left')

    # mean curve
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

xpos = xpos_start
ypos = ypos - padding - plot_height

if show_coherence_vs_interbout_interval:
    plot_height = plot_height_row
    plot_width = 1

    df = df_dict["all"]
    df_filtered = df.query(query_time)
    df_filtered = df_filtered[df_filtered[coherence_label].isin(coherence_list)]
    # computation
    parameter_list, interbout_interval_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(df_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)
    parameter_list = np.array([int(p) for p in parameter_list])

    # plot
    line_dashes = None
    plot_0 = fig.create_plot(xpos=xpos, ypos=ypos,
                             plot_height=plot_height,
                             plot_width=plot_width,
                             errorbar_area=True,
                             xl=analysed_parameter_label, xmin=min(parameter_list), xmax=max(parameter_list),
                             xticks=[int(p) for p in parameter_list], yl="Interbout interval (s)",
                             ymin=0, ymax=2,
                             yticks=[0, 1, 2])

    for i_id, id in enumerate(id_fish_list):
        df_fish = df_filtered_all.xs(id, level="experiment_ID")
        df_fish_filtered = df_fish[df_fish[coherence_label].isin(coherence_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=coherence_label, column_name=ResponseTimeColumn)

        # plot
        if id == int(fish_0_label):
            color_line = "gray"
            lw = 1
            show_label = True
        elif id == int(fish_1_label):
            color_line = "gray"
            lw = 1
            show_label = True
        elif id == int(fish_2_label):
            color_line = "gray"
            lw = 1
            show_label = True
        else:
            color_line = color_neutral
            lw = 0.05
            show_label = False
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_line, lw=lw, alpha=0.8)
        if show_label:
            plot_0.draw_text(max(parameter_list_all) + 0.1, correct_bout_list[-1], f"fish {id}",
                             textlabel_rotation='horizontal', textlabel_ha='left')

    # plot
    parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
        df_filtered_all, analysed_parameter=coherence_label, column_name=ResponseTimeColumn)
    plot_0.draw_line(x=parameter_list, y=interbout_interval_list,
                     # errorbar_area=True, yerr=std_correct_bout_list / number_individuals,
                     lc="k", lw=1, line_dashes=line_dashes)
    coefficient_variation_ibi = std_correct_bout_list / correct_bout_list * 100
    print(f"mean IBI: {correct_bout_list}")
    print(f"std IBI: {std_correct_bout_list}")
    print(f"CV IBI: {coefficient_variation_ibi}")

    ypos = ypos
    xpos = xpos + padding + plot_width

xpos = xpos_start
ypos = ypos - padding - plot_height * 2

if show_rt_distributions:
    # parameters
    padding_here = plot_height_small * 2
    palette = style.palette["stimulus"]
    x_limits = [0, 2]  # None  #
    y_limits = [0, 0.2]

    plot_height_here = plot_height_small * 3
    plot_width_here = plot_width * 0.7

    plot_section = {}

    df_dict.pop(all_label, None)  # get rid of the df with all data merged

    for i_k, k in enumerate(df_dict.keys()):
        df = df_dict[k]
        for i_param, parameter in enumerate(coherence_list):
            # plot
            line_dashes = None
            plot_title = None
            if i_k == 0:
                if i_param == 0:
                    plot_title = f"Coh={parameter}%"
                else:
                    plot_title = f"{parameter}%"
            plot_section[i_param] = fig.create_plot(plot_label=style.get_plot_label() if i_k == 0 and i_param == 0 else None,
                                                            plot_title=plot_title,
                                                            xpos=xpos + i_param * (plot_width_here + padding_plot),
                                                            ypos=ypos + plot_height_small + padding_vertical,
                                                            plot_height=plot_height_here,
                                                            plot_width=plot_width_here,
                                                            xmin=x_limits[0], xmax=x_limits[-1],
                                                            xticks=None, yticks=None,
                                                            ymin=-y_limits[-1], ymax=y_limits[-1],
                                                            hlines=[0])

            if i_param == len(coherence_list) - 1 and i_k == len(df_dict)-1:
                y_location_scalebar = y_limits[-1] / 6
                x_location_scalebar = x_limits[-1] / 6
                plot_section[i_param].draw_line((1.7, 1.7), (y_location_scalebar, y_location_scalebar + 0.1), lc="k")
                plot_section[i_param].draw_text(2, y_location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

                plot_section[i_param].draw_line((x_location_scalebar, x_location_scalebar + 0.5), (-y_location_scalebar, -y_location_scalebar), lc="k")
                plot_section[i_param].draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                        textlabel_rotation='horizontal', textlabel_ha='left', textlabel_va="bottom")

            plot_section_corr = plot_section[i_param]
            plot_section_err = plot_section[i_param]

            df_filtered = df[df[coherence_label] == parameter]
            df_filtered = df_filtered.query(query_time)

            duration = np.sum(BehavioralProcessing.get_duration_trials_in_df(df_filtered,
                                                                             fixed_time_trial=time_end_stimulus-time_start_stimulus))

            # plot distribution of data over coherence levels
            data_corr = df_filtered[df_filtered[CorrectBoutColumn] == 1][ResponseTimeColumn]
            data_err = df_filtered[df_filtered[CorrectBoutColumn] == 0][ResponseTimeColumn]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                   duration=duration,
                                                                                   center_bin=True)
            index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
            data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
            data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                 duration=duration,
                                                                                 center_bin=True)
            index_in_limits = np.argwhere(
                    np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
            data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
            data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

            plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=style.palette["correct_incorrect"][0],  # lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                        lw=0.75, line_dashes=line_dashes)
            plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=style.palette["correct_incorrect"][1],  # lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                        lw=0.75, line_dashes=line_dashes)

        ypos = ypos - (padding_here + plot_height_small)

    plot_height = 1
    ypos = ypos
    xpos = xpos + i_param * (plot_width_here + padding_plot) + plot_width_here + padding

if show_cv:
    plot_cv = fig.create_plot(xpos=xpos, ypos=ypos,
                             plot_height=plot_height,
                             plot_width=plot_width,
                             errorbar_area=True,
                             xl=analysed_parameter_label, xmin=min(parameter_list), xmax=max(parameter_list),
                             xticks=[int(p) for p in parameter_list], yl="Coefficient variation (%)",
                             ymin=0, ymax=100,
                             yticks=[0, 50, 100])
    plot_cv.draw_line(x=parameter_list, y=coefficient_variation_accuracy, lc="k", lw=1, line_dashes=(1, 2), label="Percentage correct swims")
    plot_cv.draw_line(x=parameter_list, y=coefficient_variation_ibi, lc="k", lw=1, line_dashes=(0.1, 3), label="Interbout interval")

fig.save(path_save / "figure_1.pdf", open_file=True, tight=style.page_tight)

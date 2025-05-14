import pathlib

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values
from scipy.stats import norm, laplace, goodness_of_fit

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, \
    alphabet
from analysis_helpers.analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis_helpers.analysis.utils.figure_helper import Figure

# parameters plot
style = BehavioralModelStyle()
palette = style.palette["default"]
plot_height = 1
plot_height_small = plot_height / 2

plot_width = 2
plot_width_small = plot_width / 2
padding = 1.5
padding_plot = 0.5
padding_vertical = plot_height_small
xpos_start = 0.5
ypos_start = 0.5
xpos = xpos_start
ypos = ypos_start
i_plot_label = 0  # 3  # 6  #
plot_label_list = alphabet
# palette = Palette.blue_green  # blue_short  # Palette.green_short  # Palette.red_short  #
# palette.append("#000000")
color_neutral = "#bfbfbfff"

show_trial_structure = False
show_data_amount_vs_ci_and_error_double_plot = True
show_loss_reduction = False
show_repeatability = False

# Make a standard figure
fig = Figure()

if show_trial_structure:
    time_start_stimulus = 10
    time_end_stimulus = 40
    time_experimental_trial = 50

    color_exp_1 = Palette.green_short
    color_exp_1.append("#000000")
    experiment_list = [
        {"analysed_parameter": StimulusParameterLabel.COHERENCE.value,
         "analysed_parameter_list": [50],
         "analysed_parameter_label": "C",
         "unit": "%",
         "input_function": lambda t, coh: coh/100,
         "palette": color_exp_1}
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
    plot_width_here = plot_width * 1.5 + padding

    number_plots = 0
    height_drift = 0
    for i_ex, ex in enumerate(experiment_list):

        for i_param, parameter in enumerate(ex["analysed_parameter_list"]):
            plot_section = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos - (padding_vert_here + plot_height_here) * number_plots - height_drift,
                                           plot_height=plot_height_here,
                                           plot_width=plot_width_here,
                                           xmin=0, xmax=time_experimental_trial,
                                           ymin=-0.1, ymax=1.1,
                                           vspans=[[time_start_stimulus+2, time_end_stimulus, "k", 0.2]],
                                           legend_xpos=xpos + plot_width_here,
                                           legend_ypos=ypos + 0.3 - (padding_vert_here + plot_height_here) * number_plots - height_drift)

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
    plot_section = fig.create_plot(xpos=xpos, ypos=ypos - (padding_vert_here/2 + plot_height_here) * number_plots - height_drift,
                                   plot_height=plot_height_here,
                                   plot_width=plot_width_here,
                                   xl="time [s]", xticks=[0, time_start_stimulus, time_end_stimulus, time_experimental_trial],
                                   xmin=0, xmax=time_experimental_trial,
                                   ymin=0.1, ymax=1.1)

    xpos = xpos_start
    ypos = ypos - plot_height - padding
    i_plot_label += 1

ypos_start_here = ypos

if show_data_amount_vs_ci_and_error_double_plot:
    this_plot_width = plot_width * 2
    # configurations
    score_max = None
    data_amount_unit = "trials/|coh| []"
    data_dicts = [
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\1",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\1",
         "data_amount": int(1)},
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\10",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\10",
         "data_amount": int(10)},
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\20",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\20",
         "data_amount": int(20)},
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\30",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\30",
         "data_amount": int(30)},
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\50",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\50",
         "data_amount": int(50)},
        {"path_fit": r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\100",
         "path_control": r"C:\Users\Roberto\Academics\data\benchmarking\control\100",
         "data_amount": int(100)},
    ]

    parameter_list = [
        {"label": "noise_sigma",
         "label_show": "diffusion",
         "min": 0.0,
         "mean": 1.5,
         "max": 3.0},
        {"label": 'scaling_factor',
         "label_show": "drift",
         "min": -3,
         "mean": 0,
         "max": 3},
        {"label": 'leak',
         "label_show": "leak",
         "min": -3,
         "mean": 0,
         "max": 3},
        {"label": 'residual_after_bout',
         "label_show": "reset",
         "min": 0.0,
         "mean": 0.5,
         "max": 1.0},
        {"label": 'inactive_time',
         "label_show": "delay",
         "min": 0.0,
         "mean": 0.5,
         "max": 1.0},
    ]

    # compute
    data_amount_array = np.array([d["data_amount"] for d in data_dicts])
    x = np.arange(len(data_amount_array)) + 1
    ci_array = {"fit": {p["label"]: np.zeros(len(data_dicts)) for p in parameter_list},
                "control": {p["label"]: np.zeros(len(data_dicts)) for p in parameter_list}}
    error_array = {"fit": {p["label"]: np.zeros(len(data_dicts)) for p in parameter_list},
                   "control": {p["label"]: np.zeros(len(data_dicts)) for p in parameter_list}}
    error_array_raw = [{"fit": {p["label"]: [] for p in parameter_list},
                       "control": {p["label"]: [] for p in parameter_list}}
                       for _ in data_dicts]

    # ci_array["score"] = np.zeros(len(data_dicts))
    score_mean_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}
    score_sem_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}
    score_min_array_dict = {"fit": np.zeros(len(data_dicts)), "control": np.zeros(len(data_dicts))}
    for i_data, data in enumerate(data_dicts):
        number_models = 0
        source_path = Path(data["path_fit"])
        model_dict = {}
        for model_filepath in source_path.glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            test_label = model_filename.split("_")[2].replace(".hdf5", "")
            model_dict[test_label] = {"fit": model_filepath}
            number_models += 1
            for target_filepath in source_path.glob(f'model_test_{test_label}_*.hdf5'):
                if "_fit." in target_filepath.name or "_fit_all." in target_filepath.name:
                    continue
                else:
                    model_dict[test_label]["target"] = target_filepath
                    break
        if data["path_control"] is not None:
            source_path = Path(data["path_control"])
            for model_filepath in source_path.glob('model_*.hdf5'):
                model_filename = str(model_filepath.name)
                test_label = model_filename.split("_")[2].replace(".hdf5", "")
                try:
                    model_dict[test_label]["control"] = model_filepath
                except KeyError:
                    print(f"model {test_label} only available as control, no fit")
            label_list = ["fit", "control"]
        else:
            label_list = ["fit"]

        ci = {}
        for label in label_list:
            ci[label] = {parameter["label"]: [] for parameter in parameter_list}
            ci[label]["score"] = []
            error_all = {parameter["label"]: [] for parameter in parameter_list}
            for i_model, id_model in enumerate(model_dict.keys()):
                df_model_fit_list = pd.read_hdf(model_dict[id_model][label])
                df_model_target = pd.read_hdf(model_dict[id_model]["target"])
                for i_parameter, parameter in enumerate(parameter_list):
                    parameter_values = np.array(df_model_fit_list[parameter["label"]])
                    ci[label][parameter["label"]].append(np.percentile(a=parameter_values, q=95) - np.percentile(a=parameter_values, q=5))
                    # error_all[parameter["label"]].append(np.abs(df_model_fit_best[parameter["label"]][0] - df_model_target[parameter["label"]][0]))
                    error_all[parameter["label"]].extend(np.abs(parameter_values - df_model_target[parameter["label"]][0]))

                    normalized_error = np.abs(parameter_values - df_model_target[parameter["label"]][0]) / (parameter["max"] - parameter["min"])
                    error_array_raw[i_data][label][parameter["label"]].append(normalized_error)  # #####
                score_values = np.array(df_model_fit_list["score"])
                ci[label]["score"].extend(score_values)

            for i_parameter, parameter in enumerate(parameter_list):
                mean_ci = np.std(np.array(error_all[parameter["label"]]) / (parameter["max"] - parameter["min"])) * 100  # / number_models
                ci_array[label][parameter["label"]][i_data] = mean_ci
                mean_error = np.median(np.array(error_all[parameter["label"]]) / (parameter["max"] - parameter["min"])) * 100
                error_array[label][parameter["label"]][i_data] = mean_error

            mean_score = np.mean(ci[label]["score"])
            sem_score = np.std(ci[label]["score"])  # / number_models
            min_score = np.min(ci[label]["score"])
            # ci_array["score"][i_data] = mean_ci_score
            score_mean_array_dict[label][i_data] = mean_score
            score_sem_array_dict[label][i_data] = sem_score
            score_min_array_dict[label][i_data] = min_score


    # plot
    plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=this_plot_width, errorbar_area=True,  # xlog=True,
                             xl=data_amount_unit, xmin=0, xmax=np.max(x) + 1,
                             xticks=x, xticklabels=[str(d) for d in data_amount_array],
                             yl="loss [-]", ymin=0, ymax=30, yticks=[0, 15, 30])  # ymax=125, yticks=[0, 25, 50, 75, 100, 125])

    i_plot_label += 1
    for label in ["fit", "control"]:
    # for label in ["fit"]:
        if label == "fit":
            line_dashes = None
            color = "k"
        else:
            line_dashes = None  # (1, 2)
            color = "r"
        plot_loss.draw_line(x, score_mean_array_dict[label], lc=color, lw=0.75, line_dashes=line_dashes, yerr=score_sem_array_dict[label])  # , label=label)
        plot_loss.draw_scatter(x, score_mean_array_dict[label], pc=color, ec=color)

    xpos = xpos_start
    ypos = ypos - plot_height - padding

    # i_plot_label += 1
    # xpos = xpos + this_plot_width + padding
    # # ypos = ypos - plot_height - padding
    #
    # plot_ci = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos,
    #                          plot_height=plot_height, plot_width=this_plot_width,
    #                          xl=data_amount_unit, xmin=0, xmax=np.max(x) + 1,
    #                          xticks=x, xticklabels=[str(d) for d in data_amount_array],  # xlog=True,
    #                          yl="CI [%]", ymin=0, ymax=75, yticks=[0, 25, 50, 75])
    #
    # i_plot_label += 1
    # # ypos = ypos - plot_height - padding
    # xpos = xpos + this_plot_width + padding
    #
    # plot_error = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos,
    #                           plot_height=plot_height, plot_width=this_plot_width,
    #                           xl=data_amount_unit, xmin=0, xmax=np.max(x) + 1,
    #                           xticks=x, xticklabels=[str(d) for d in np.sort(data_amount_array)],  # xlog=True,
    #                           yl="error [%]", ymin=0, ymax=75, yticks=[0, 25, 50, 75])

    for i_p, p in enumerate(parameter_list):
        plot_error = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 else None, xpos=xpos, ypos=ypos,
                                     plot_height=plot_height, plot_width=this_plot_width,
                                     xl=data_amount_unit, xmin=0, xmax=np.max(x) + 1,
                                     xticks=x, xticklabels=[str(d) for d in np.sort(data_amount_array)],
                                     # xlog=True,
                                     yl=f"{p['label_show']}\nerror [%]", ymin=0, ymax=75,
                                     yticks=[0, 25, 50, 75])  # ymax=25, yticks=[0, 12.5, 25])  #

        xpos = xpos + this_plot_width + padding
        if i_p == 0:
            i_plot_label += 1

        # for label in ["fit", "control"]:
        for label in ["fit", "control"]:
            if label == "fit":
                line_dashes = None
                label_plot = f"{p['label_show']} fit"
                color = "k"
            else:
                line_dashes = None  # (1, 2)
                label_plot = f"{p['label_show']} control"
                color = "r"

            # ypos = ypos - plot_height - padding

            # plot_ci = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos,
            #                           plot_height=plot_height, plot_width=this_plot_width,
            #                           xl=data_amount_unit, xmin=0, xmax=np.max(x) + 1,
            #                           xticks=x, xticklabels=[str(d) for d in data_amount_array],  # xlog=True,
            #                           yl="CI [%]", ymin=0, ymax=75, yticks=[0, 25, 50, 75])
            #
            # i_plot_label += 1
            # ypos = ypos - plot_height - padding
            # xpos = xpos + this_plot_width + padding


        # for label in ["fit", "control"]:
        #     if label == "fit":
        #         line_dashes = None
        #         label_plot = f"{p['label_show']} fit"
        #     else:
        #         line_dashes = (1, 2)
        #         label_plot = f"{p['label_show']} control"
        #     plot_ci.draw_line(x, ci_array[label][p["label"]], lc=Palette.arlecchino[i_p], lw=0.75, line_dashes=line_dashes)
        #     plot_ci.draw_scatter(x, ci_array[label][p["label"]], pc=Palette.arlecchino[i_p])
            plot_error.draw_line(x, error_array[label][p["label"]], lc=color,
                                 # lc=Palette.arlecchino[i_p],
                                 lw=0.75, line_dashes=line_dashes, yerr=ci_array[label][p["label"]]/2)  #, label=label_plot)
            plot_error.draw_scatter(x, error_array[label][p["label"]], pc=color, ec=color)  # pc=Palette.arlecchino[i_p])

            xpos = xpos_start
            ypos = ypos - plot_height - padding

    mean_rmse = {data['data_amount']: {"fit": {"laplace": [], "gauss": []}, "control": {"laplace": [], "gauss": []}} for data in data_dicts}
    for i_data, data in enumerate(data_dicts):
        print(f"DATA AMOUNT: {data['data_amount']}")
        for i_p, p in enumerate(parameter_list):
            print(f"PARAMETER: {p['label_show']}")
            for label in ["fit", "control"]:
                print(f"dataset: {label}")
                error_array_end = np.squeeze(np.array(error_array_raw[i_data][label][p["label"]]))
                data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end,
                                                                                     bins=np.arange(-3, 3, 0.1),
                                                                                     density=True, center_bin=True)

                # DISTRIBUTION ESTIMATION FIT
                mu, b = laplace.fit(error_array_end)
                y_dist = laplace.pdf(data_hist_time_end, mu, b)
                # y_dist = normpdf(data_hist_time_end, mu, std)
                y_dist /= np.sum(y_dist)

                res_laplace = goodness_of_fit(laplace, error_array_end)
                rms_laplace = np.sqrt(np.mean((data_hist_value_end - y_dist) ** 2))
                print(f"RMSE Laplace = {rms_laplace}")
                mean_rmse[data['data_amount']][label]["laplace"].append(rms_laplace)
                # print(f"SUM Laplace | {np.sum(y_dist)}")

                mu, sigma = norm.fit(error_array_end)
                y_dist = norm.pdf(data_hist_time_end, mu, sigma)
                # y_dist = normpdf(data_hist_time_end, mu, std)
                y_dist /= np.sum(y_dist)

                res_gauss = goodness_of_fit(norm, error_array_end)
                rms_gauss = np.sqrt(np.mean((data_hist_value_end - y_dist) ** 2))
                print(f"RMSE Gauss = {rms_gauss}")
                mean_rmse[data['data_amount']][label]["gauss"].append(rms_gauss)
                # print(f"SUM Gauss | {np.sum(y_dist)}")

    for i_data, data in enumerate(data_dicts):
        for i_p, p in enumerate(parameter_list):
            for label in ["fit", "control"]:
                mean_rmse[data['data_amount']][label]["laplace"] = np.mean(mean_rmse[data['data_amount']][label]["laplace"])
                mean_rmse[data['data_amount']][label]["gauss"] = np.mean(mean_rmse[data['data_amount']][label]["gauss"])

    xpos += padding + this_plot_width
    ypos = ypos_start_here

if show_loss_reduction:
    path_dir_loss = r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\_30"

    loss_start_list = []
    loss_end_list = []
    for path_error in pathlib.Path(path_dir_loss).glob("error_test_*.hdf5"):
        df_error = pd.read_hdf(path_error)
        loss_start = df_error["score"][0]
        loss_end = df_error["score"][len(df_error)-1]
        loss_start_list.append(loss_start)
        loss_end_list.append(loss_end)

    loss_mean = np.mean(loss_end_list)
    loss_std = np.std(loss_end_list)
    plot_title = fr"{loss_mean:0.02f}$\pm${loss_std:0.02f}"
    plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label], plot_title=plot_title,
                                xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                ymin=0, ymax=30, yticks=[0, 15, 30], yl=f"loss",
                                xl=None, xmin=0.5, xmax=2.5,
                                xticks=None
                                # xticks=[1, 2], xticklabels=["iteration 0", "iteration 1500"], xticklabels_rotation=45
                                )

    for i in range(len(loss_end_list)):
        loss_start = loss_start_list[i]
        loss_end = loss_end_list[i]
        plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
        plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

    ypos = ypos - padding - plot_height
    i_plot_label += 1

if show_repeatability:
    path_dir_repeat = Path(r"C:\Users\Roberto\Academics\data\benchmarking\test_repeatability")
    test_id = "006"
    for path in path_dir_repeat.glob(f"model_test_{test_id}_*.hdf5"):
        if "_fit." in path.name:
            continue

        # test_id = path.name.split("_")[2].replace(".hdf5", "")
        df_target = pd.read_hdf(path)

        for i_p, p in enumerate(parameter_list):
            p_target = df_target[p["label"]]
            plot_p = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 else None,
                                     xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                     errorbar_area=False, ymin=p["min"], ymax=p["max"],
                                     yticks=[p["min"], p["mean"], p["max"]],
                                     yl=f"{p['label_show']}",
                                     xl="iteration" if i_p == len(parameter_list)-1 else None,
                                     xmin=0.5, xmax=2.5, xticks=[1, 2] if i_p == len(parameter_list)-1 else None,
                                     xticklabels=["0", "1500"] if i_p == len(parameter_list)-1 else None,
                                     # xticklabels_rotation=45 if i_p == len(parameter_list)-1 else None,
                                     hlines=[df_target[p["label"]][0]])
            ypos -= (plot_height + padding)
            if i_p == 0:
                i_plot_label += 1

            p_fit_end_list = []
            for path_fit in path_dir_repeat.glob(f"error_test_{test_id}_*_fit.hdf5"):
                df_fit = pd.read_hdf(path_fit)
                p_fit_start = df_fit[f"{p['label']}_value"][0]
                p_fit_end = df_fit[f"{p['label']}_value"][len(df_fit)-1]
                p_fit_end_list.append(p_fit_end)

                plot_p.draw_line((1, 2), (p_fit_start, p_fit_end), lc=palette[i_p], lw=0.05, alpha=0.5)
                plot_p.draw_scatter((1, 2), (p_fit_start, p_fit_end), pc=palette[i_p], ec=palette[i_p])

            plot_p.draw_scatter([2], [np.median(p_fit_end_list)], pc="k")

# fig.save(Path.home() / 'Academics' / 'graphics' / 'pictures' / 'figures_for_papers' / 'behavior_model' / "figure_5_stationarity_temp.pdf", open_file=True, tight=True)
fig.save(Path.home() / 'Desktop' / "figure_s2_model_identifiability.pdf", open_file=True, tight=True)

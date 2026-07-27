"""
The present script produces the following figure (or figure panels) in Garza et al 2026:
- Extended Data Fig. 1a-b

Overview:
Stationarity analysis of behavioral performance across coherence levels and time.
This script loads bout-level behavioral data from 5 dpf wild-type zebrafish larvae
(base_dataset_5dpfWT) and compares accuracy (percentage correct) and interbout
interval (IBI) across stimulus coherence levels, split into two 60-minute time
bins (0-60 min vs 60-120 min) to assess response stationarity over the session.

Workflow:
1. Load environment paths and experiment data (HDF5), filtering out invalid bouts.
2. For each time bin and coherence level, compute per-fish mean accuracy and IBI
   (averaged across bouts within each experiment/fish).
3. Plot per-fish values (scatter, offset by time bin) and across-fish means for
   accuracy and IBI as a function of coherence.
4. Run paired Wilcoxon signed-rank tests between adjacent coherence levels
   (within each time bin) for both accuracy and IBI, marking significant
   differences (p < 0.05) on the plots and logging p-values to console.
5. Save the resulting two-panel figure (accuracy and IBI vs. coherence) as a PDF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from scipy.stats import wilcoxon, normaltest

from figures.style import BehavioralModelStyle
from service.figure_helper import Figure
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel


# --------------------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / ".env"
env = dotenv_values(env_path)
path_dir = Path(env["PATH_DIR"])
path_data = path_dir / "base_dataset_5dpfWT" / "data_fish_all.hdf5"
path_save = Path(env["PATH_SAVE"])

# --------------------------------------------------------------------------
# Plot style and layout configuration
# --------------------------------------------------------------------------
style = BehavioralModelStyle()
xpos = style.xpos_start
ypos = style.ypos_start
plot_height = style.plot_height * 1.5
plot_width = style.plot_width * 1.5
padding = style.padding

# --------------------------------------------------------------------------
# Data plotting configuration
# --------------------------------------------------------------------------
time_bin_list = [
    {"label": "start",
     "offset": 0,      # seconds
     "duration": 3600  # seconds
    },
    {"label": "end",
     "offset": 3600,   # seconds
     "duration": 3600  # seconds
    },
]
coherence_list = ConfigurationExperiment.coherence_list

# --------------------------------------------------------------------------
# Initialize main figure container
# --------------------------------------------------------------------------
fig = Figure()

# --------------------------------------------------------------------------
# Load data and filter
# --------------------------------------------------------------------------
df = pd.read_hdf(path_data)
df = df.loc[df[ConfigurationExperiment.CorrectBoutColumn] != -1]

# --------------------------------------------------------------------------
# Initialize figure panels
# --------------------------------------------------------------------------
plot_accuracy = fig.create_plot(
                plot_label=style.get_plot_label(), xl=ConfigurationExperiment.coherence_label,
                yl="Percentage correct (%)",
                xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                xmin=-10, xmax=110, xticks=coherence_list, ymin=0, ymax=100,
                yticks=[0, 50, 100])
xpos += plot_width + padding
plot_ibi = fig.create_plot(
                plot_label=style.get_plot_label(), xl=ConfigurationExperiment.coherence_label,
                yl="Interswim interval (s)",
                xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                xmin=-10, xmax=110, xticks=coherence_list, ymin=0, ymax=5,
                yticks=[0, 2.5, 5])

# --------------------------------------------------------------------------
# Computation of percentage correct and IBI by time bin
# --------------------------------------------------------------------------
fish_dict = {}
max_ibi = 0
min_ibi = 100
for i_tb, tb in enumerate(time_bin_list):
    fish_dict[i_tb] = {}
    df_tb = df.query(f'start_time_absolute > {tb["offset"]} and end_time_absolute < {tb["offset"] + tb["duration"]}')
    for i_coh, coh in enumerate(coherence_list):
        df_tb_coh = df_tb.loc[df_tb[StimulusParameterLabel.COHERENCE.value] == coh]
        df_tb_coh_meanfish = df_tb_coh.groupby("experiment_ID").mean()
        for exp_ID in df_tb_coh_meanfish.index.unique("experiment_ID"):
            # Organize in the dictionary creating new key or populating with values an existing one
            if exp_ID in fish_dict[i_tb].keys():
                if coh in fish_dict[i_tb][exp_ID].keys():
                    fish_dict[i_tb][exp_ID][coh]["accuracy"].append(df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.CorrectBoutColumn])
                    ibi_value = df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.ResponseTimeColumn]
                    fish_dict[i_tb][exp_ID][coh]["ibi"].append(ibi_value)
                else:
                    ibi_value = df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.ResponseTimeColumn]
                    fish_dict[i_tb][exp_ID][coh] = {"accuracy": [df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.CorrectBoutColumn]],
                                              "ibi": [ibi_value]}
            else:
                ibi_value = df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.ResponseTimeColumn]
                fish_dict[i_tb][exp_ID] = {coh: {"accuracy": [df_tb_coh_meanfish.loc[exp_ID][ConfigurationExperiment.CorrectBoutColumn]],
                                           "ibi": [ibi_value]}}
        # Dynamic (data-dependent) plotting configurations
        if ibi_value > max_ibi: max_ibi = ibi_value
        if ibi_value < min_ibi: min_ibi = ibi_value
        x = np.ones(len(df_tb_coh_meanfish)) * coh
        if i_tb == 0:
            x -= 5
            color = "#808080"
            label = "0-60min"
            pt="s"
        elif i_tb == len(time_bin_list)-1:
            x += 5
            color = "#808080"
            label = "60-120min"
            pt="D"

        # --------------------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------------------
        plot_accuracy.draw_scatter(x, np.array(df_tb_coh_meanfish[ConfigurationExperiment.CorrectBoutColumn])*100,
                                   pc=color, ec=color, alpha=0.3, pt=pt)
        plot_accuracy.draw_scatter(x[0], np.mean(df_tb_coh_meanfish[ConfigurationExperiment.CorrectBoutColumn])*100,
                                   pc="k", ec="k")
        plot_ibi.draw_scatter(x, df_tb_coh_meanfish[ConfigurationExperiment.ResponseTimeColumn], pt=pt,
                                   pc=color, ec=color, label=label if i_coh == 0 else None, alpha=0.3)
        plot_ibi.draw_scatter(x[0], np.mean(df_tb_coh_meanfish[ConfigurationExperiment.ResponseTimeColumn]),
                                   pc="k", ec="k", label="Fish mean" if i_tb == len(time_bin_list)-1 and i_coh == 0 else None)

# --------------------------------------------------------------------------
# Statistical test between pairs of coh for each time bin
# --------------------------------------------------------------------------
pval_threshold = 0.05
for i_tb in range(len(time_bin_list)):
    for i_coh in range(len(coherence_list)-1):
        # Compute accuracy (percentage correct) and IBI in pairs of time bins
        array_acc_0 = np.array([fish_dict[i_tb][exp_ID][coherence_list[i_coh]]["accuracy"][0] for exp_ID in fish_dict[i_tb].keys()])
        array_acc_1 = np.array([fish_dict[i_tb][exp_ID][coherence_list[i_coh+1]]["accuracy"][0] for exp_ID in fish_dict[i_tb].keys()])
        array_ibi_0 = np.array([fish_dict[i_tb][exp_ID][coherence_list[i_coh]]["ibi"][0] for exp_ID in fish_dict[i_tb].keys()])
        array_ibi_1 = np.array([fish_dict[i_tb][exp_ID][coherence_list[i_coh+1]]["ibi"][0] for exp_ID in fish_dict[i_tb].keys()])

        # Test accuracy
        res_acc = wilcoxon(array_acc_0, array_acc_1)
        # Mark significant pairs on the accuracy plot
        x = [coherence_list[i_coh]+5, coherence_list[i_coh+1]-5]
        y_acc = np.ones(2)*10
        plot_accuracy.draw_line(x, y_acc)
        if res_acc.pvalue < pval_threshold:
            y_star = np.mean(y_acc)+10 + i_tb * 5
            plot_accuracy.draw_scatter(np.mean(x), y_star, pt="*", ec="k" if i_tb == 0 else "r")
        # Log
        print(f"Wilcoxon accuracy | tb={i_tb} | coh={coherence_list[i_coh]}-{coherence_list[i_coh+1]} | pval={res_acc.pvalue}")

        # Test IBI
        res_ibi = wilcoxon(array_ibi_0, array_ibi_1)

        # Mark significant pairs on the IBI plot
        y_ibi = np.ones(2)*4
        plot_ibi.draw_line(x, y_ibi)
        if res_ibi.pvalue < pval_threshold:
            y_star = np.mean(y_ibi)+0.1 + i_tb * 0.50
            plot_ibi.draw_scatter(np.mean(x), y_star, pt="*", ec="k" if i_tb == 0 else "r")
        # Log
        print(f"Wilcoxon IBI | tb={i_tb} | coh={coherence_list[i_coh]}-{coherence_list[i_coh+1]} | pval={res_ibi.pvalue}")

# --------------------------------------------------------------------------
# Save final figure
# --------------------------------------------------------------------------
fig.save(path_save / "figure_s1_stationarity.pdf", open_file=False, tight=style.page_tight)

        

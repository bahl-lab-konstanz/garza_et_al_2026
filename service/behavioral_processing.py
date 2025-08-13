import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from itertools import islice, chain

from matplotlib import pyplot as plt
from numba import jit
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.special import rel_entr
from scipy.stats import poisson
from typing import List

from analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, Keyword, Direction, \
    CorrectBoutColumn, ResponseTimeColumn, StraightBout, StimulusParameterLabel2Dir
from analysis.personal_dirs.Roberto.utils.pre_analysis import PreAnalysis
from analysis.personal_dirs.Roberto.utils.service.signal_processing import SignalProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis.personal_dirs.Roberto.utils.toolkit import Converter


class BehavioralProcessing():
    group_by_idxs = ['experiment_ID', 'experiment_repeat', 'stimulus_name', 'trial']
    tag = 'behavioral_processing'


    @classmethod
    def get_multiple_df_reactive(cls, path_dir, path_dataset_in_hdf, side_threshold=1, sanity_check=False,
                                 time_start_stimulus=0, time_hours_offset=0, time_hours_to_analyse=100,
                                 inverted_stimulus=False):
        df_dict = {}
        for fish_path in path_dir.glob("*"):
            if (not fish_path.is_dir()):
                continue

            try:
                df = cls.get_df_reactive(fish_path / f"{fish_path.name}.hdf5", path_dataset_in_hdf,
                                         side_threshold=side_threshold, sanity_check=sanity_check,
                                         time_start_stimulus=time_start_stimulus, time_hours_offset=time_hours_offset,
                                         time_hours_to_analyse=time_hours_to_analyse, inverted_stimulus=inverted_stimulus)
                df_dict[fish_path.name] = df
            except FileNotFoundError:
                print(f"INFO | file {str(fish_path)}/{fish_path.name}.hdf5 not found")
        return df_dict

    @classmethod
    def get_df_reactive(cls, path_data, path_dataset_in_hdf, side_threshold=1, sanity_check=False,
                        time_start_stimulus=0, time_hours_offset=0, time_hours_to_analyse=100, inverted_stimulus=False):
        hf = h5py.File(path_data, 'r')
        dataset = hf.get(path_dataset_in_hdf)
        array = np.array(dataset)
        info = dict(dataset.attrs.items())
        columns = info["column_names"]

        df = pd.DataFrame(array, columns=columns)
        df["fish_ID"] = Path(path_data).name
        df.rename(columns={"end_result_info0": "reaction_time", "end_result_info1": "estimated_orientation_change"},
                  inplace=True)

        duration_trial_array_long = array[:, 1] - array[:, 0]
        duration_trial_array = duration_trial_array_long - time_start_stimulus
        df["stimulus_duration"] = duration_trial_array
        df["trial_duration"] = duration_trial_array_long
        df["time_start_stimulus"] = time_start_stimulus
        df["start_time_absolute"] = df["start_time"] - df["start_time"][0]
        df["end_time_absolute"] = df["end_time"] - df["start_time"][0]

        for i_row, row in df.iterrows():
            stimulus_name = info["stimulus_index_names"][int(row["stimulus_index"])]
            stimulus_parameter_dict = cls.parse_stimulus_name(stimulus_name)
            for parameter in stimulus_parameter_dict:
                if parameter not in df.columns:
                    df[parameter] = np.nan
                df.loc[i_row, parameter] = stimulus_parameter_dict[parameter]
        if "heading_angle_change" in df.columns:
            df["estimated_orientation_change"] = df["heading_angle_change"]
            df.drop("heading_angle_change", inplace=True, axis=1)
        df[CorrectBoutColumn] = cls.compute_correct_bout_list(df, side_threshold=side_threshold,
                                                              inverted_stimulus=inverted_stimulus, simmetric=False)
        df["flipped_estimated_orientation_change"] = cls.flip_column(df,"estimated_orientation_change")
        query_time = f"start_time_absolute > {Converter.hour2sec(time_hours_offset)} and end_time_absolute < {Converter.hour2sec(time_hours_offset + time_hours_to_analyse)}"
        df = df.query(query_time)
        if sanity_check:
            df = BehavioralProcessing.sanity_check_reactive(df, max_interbout_interval=30)
        return df

    @staticmethod
    def get_df_for_parameter_analysis(
            path,
            analysed_parameter,  # may be set to a bool or to a string with the name of the analysed parameter (in case of ranges the latter is mandatory)
            parameter_range_list=None,
            time_hours_offset=0,
            time_hours_to_analyse=2,
            sanity_check=True,
            debug=False,
            wrong_direction_label=False,
            compute_correct_bout=True,
            side_threshold=3,
            single_dataset=False
    ):
        if single_dataset:
            df_all = PreAnalysis.get_df(path)
        else:
            df_all = PreAnalysis.get_df_all(path)
        if sanity_check:
            df_all = BehavioralProcessing.sanity_check(df_all, max_interbout_interval=30)

        # slice desired hours out of experiment
        df_all['start_time_absolute'] = df_all['start_time_absolute'].astype(np.int64) * 1e-9  # convert to POSIX timestamp in seconds
        df_all['end_time_absolute'] = df_all['end_time_absolute'].astype(np.int64) * 1e-9  # convert to POSIX timestamp in seconds
        time_start_experiment = df_all['start_time_absolute'].min()
        df_all['start_time_absolute'] -= time_start_experiment
        df_all['end_time_absolute'] -= time_start_experiment
        query_time = f"start_time_absolute > {Converter.hour2sec(time_hours_offset)} and end_time_absolute < {Converter.hour2sec(time_hours_offset + time_hours_to_analyse)}"
        df = df_all.query(query_time)
        if df.empty:
            raise "WARNING | plotting | define dataframe | dataframe is empty"
        if debug:
            print(f"DEBUG | plotting | define dataframe | min start time absolute: {df['start_time_absolute'].min()}")
            print(f"DEBUG | plotting | define dataframe | max start time absolute: {df['start_time_absolute'].max()}")

        # add stimulus parameters columns
        stimulus_name_list = df.index.unique('stimulus_name')
        BehavioralProcessing.add_empty_parameter_column_list(df)
        for stimulus_name in stimulus_name_list:
            stimulus_parameter_dict = BehavioralProcessing.parse_stimulus_name(stimulus_name)
            stimulus_index_list = df.index.get_level_values('stimulus_name') == stimulus_name
            if analysed_parameter is not None and analysed_parameter is not False:
                if parameter_range_list is None:
                    for parameter in stimulus_parameter_dict:
                        df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                else:
                    for parameter in stimulus_parameter_dict:
                        df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                        if parameter == analysed_parameter:
                            try:
                                df.loc[stimulus_index_list, 'parameter_range'] = [parameter_range['label']
                                                                                  for parameter_range in parameter_range_list
                                                                                  if float(parameter_range['min']) < float(
                                        stimulus_parameter_dict[parameter]) <= float(parameter_range['max'])][0]
                            except IndexError:
                                print(
                                    f"WARNING | plotting | define dataframe | parameter {parameter} has value {stimulus_parameter_dict[parameter]} out of analysed ranges")
                    df = df[df['parameter_range'].notna()]
            # only for initial dataset with wrong direction label
            if wrong_direction_label:
                if 'left' in stimulus_name:
                    df.loc[
                        stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.RIGHT.value  # initally inverted
                elif 'right' in stimulus_name:
                    df.loc[
                        stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.LEFT.value  # initally inverted
        if compute_correct_bout:
            df['correct_bout'] = BehavioralProcessing.compute_correct_bout_list(df, side_threshold=side_threshold)
        return df

    def initialize_opts_dict(self):
        opts_dict = dict({})
        opts_dict["head_embedded_real_fish"] = dict({})
        opts_dict["freely_swimming_intelligent_agent"] = dict({})
        opts_dict["freely_swimming_real_fish"] = dict({})
        opts_dict["tail_real_fish"] = dict({})
        opts_dict["stimulus"] = dict({})
        opts_dict["general"] = dict({})

        opts_dict["general"]["compression"] = 0
        opts_dict["general"]["verbose"] = 0  # Added verbose key
        opts_dict["general"]["recompute_windowed_variances"] = 0  # Added recompute_windowed_variances key
        opts_dict["general"]["stimulus_alignment_start_delta_t"] = 0  # Added stimulus_alignment_start_delta_t key
        opts_dict["general"]["stimulus_alignment_end_delta_t"] = 0  # Added stimulus_alignment_end_delta_t key
        opts_dict["general"]["stimulus_alignment_strategy"] = 0  # Added stimulus_alignment_strategy key

        opts_dict["stimulus"]["interpolation_dt"] = 1 / 60

        opts_dict["freely_swimming_real_fish"]["interpolation_dt"] = 1 / 90
        opts_dict["freely_swimming_real_fish"]["bout_detection_start_threshold"] = 2.000
        opts_dict["freely_swimming_real_fish"][
            "bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.020
        opts_dict["freely_swimming_real_fish"]["bout_detection_end_threshold"] = 1.500
        opts_dict["freely_swimming_real_fish"][
            "bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.050
        opts_dict["freely_swimming_real_fish"]["bout_detection_maximal_length_of_event"] = 2
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_start"] = -0.0600
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_start"] = -0.0300
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0600
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t3_relative_to_event_start"] = -0.0300
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_end"] = 0.0100
        opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0500

        opts_dict["freely_swimming_intelligent_agent"]["interpolation_dt"] = 1 / 60
        opts_dict["freely_swimming_intelligent_agent"]["bout_detection_start_threshold"] = 2.000
        opts_dict["freely_swimming_intelligent_agent"][
            "bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.010
        opts_dict["freely_swimming_intelligent_agent"]["bout_detection_end_threshold"] = 1.500
        opts_dict["freely_swimming_intelligent_agent"][
            "bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.080
        opts_dict["freely_swimming_intelligent_agent"]["bout_detection_maximal_length_of_event"] = 2
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t0_relative_to_event_start"] = -0.0600
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t1_relative_to_event_start"] = -0.0300
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0600
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t3_relative_to_event_start"] = -0.0300
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t0_relative_to_event_end"] = 0.0100
        opts_dict["freely_swimming_intelligent_agent"][
            "swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0500

        opts_dict["head_embedded_real_fish"]["interpolation_dt"] = 1 / 350
        opts_dict["head_embedded_real_fish"]["bout_detection_start_threshold"] = 0.500
        opts_dict["head_embedded_real_fish"][
            "bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.020
        opts_dict["head_embedded_real_fish"]["bout_detection_end_threshold"] = 0.250
        opts_dict["head_embedded_real_fish"][
            "bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.050
        opts_dict["head_embedded_real_fish"]["bout_detection_maximal_length_of_event"] = 2
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_start"] = -0.1500
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_start"] = -0.1000
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0300
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t3_relative_to_event_start"] = 0.0300
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_end"] = -0.0300
        opts_dict["head_embedded_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0300

        opts_dict["tail_real_fish"]["interpolation_dt"] = 1 / 350
        opts_dict["tail_real_fish"]["bout_detection_start_threshold"] = 10.00
        opts_dict["tail_real_fish"]["bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.020
        opts_dict["tail_real_fish"]["bout_detection_end_threshold"] = 5.0
        opts_dict["tail_real_fish"]["bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.050
        opts_dict["tail_real_fish"]["bout_detection_maximal_length_of_event"] = 4
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_start"] = -0.1500
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_start"] = -0.1000
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0300
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t3_relative_to_event_start"] = 0.0300
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_end"] = -0.0300
        opts_dict["tail_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0300
        return opts_dict

    def get_df_single_fish_tail_tracking(
            path,
            analysed_parameter,
            # may be set to a bool or to a string with the name of the analysed parameter (in case of ranges the latter is mandatory)
            parameter_range_list=None,
            time_hours_offset=0,
            time_hours_to_analyse=2,
            sanity_check=True,
            debug=False,
            wrong_direction_label=False,
            compute_correct_bout=True,
            side_threshold=1
    ):
        df_dict = {}
        for fish_path in path.glob("*"):
            if (not fish_path.is_dir()):
                continue

            try:
                # df = PreAnalysis.preprocess_df_reactive(fish_path / f"{fish_path.name}.hdf5")  # #####
                df = PreAnalysis.get_df_tail_tracking(fish_path / f"{fish_path.name}.hdf5")  # #####
                df_dict[fish_path.name] = df
            except FileNotFoundError:
                print(f"INFO | file {str(fish_path)}/{fish_path.name}.hdf5 not found")

        for key, df in df_dict.items():
            if sanity_check:
                df = BehavioralProcessing.sanity_check(df, max_interbout_interval=3600)

            # slice desired hours out of experiment
            df['start_time_absolute'] = df['start_time_absolute'].astype(
                np.int64) * 1e-9  # convert to POSIX timestamp in seconds
            df['end_time_absolute'] = df['end_time_absolute'].astype(
                np.int64) * 1e-9  # convert to POSIX timestamp in seconds
            time_start_experiment = df['start_time_absolute'].min()
            df['start_time_absolute'] -= time_start_experiment
            df['end_time_absolute'] -= time_start_experiment
            query_time = f"start_time_absolute > {Converter.hour2sec(time_hours_offset)} and end_time_absolute < {Converter.hour2sec(time_hours_offset + time_hours_to_analyse)}"
            df = df.query(query_time)
            if df.empty:
                raise "WARNING | plotting | define dataframe | dataframe is empty"
            if debug:
                print(f"DEBUG | plotting | define dataframe | min start time absolute: {df['start_time_absolute'].min()}")
                print(f"DEBUG | plotting | define dataframe | max start time absolute: {df['start_time_absolute'].max()}")

            # add stimulus parameters columns
            stimulus_name_list = df.index.unique('stimulus_name')
            BehavioralProcessing.add_empty_parameter_column_list(df)
            for stimulus_name in stimulus_name_list:
                stimulus_parameter_dict = BehavioralProcessing.parse_stimulus_name(stimulus_name)
                stimulus_index_list = df.index.get_level_values('stimulus_name') == stimulus_name
                if analysed_parameter is not None and analysed_parameter is not False:
                    if parameter_range_list is None:
                        for parameter in stimulus_parameter_dict:
                            df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                    else:
                        for parameter in stimulus_parameter_dict:
                            df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                            if parameter == analysed_parameter:
                                try:
                                    df.loc[stimulus_index_list, 'parameter_range'] = [parameter_range['label']
                                                                                      for parameter_range in
                                                                                      parameter_range_list
                                                                                      if
                                                                                      float(parameter_range['min']) < float(
                                                                                          stimulus_parameter_dict[
                                                                                              parameter]) <= float(
                                                                                          parameter_range['max'])][0]
                                except IndexError:
                                    print(
                                        f"WARNING | plotting | define dataframe | parameter {parameter} has value {stimulus_parameter_dict[parameter]} out of analysed ranges")
                        df = df[df['parameter_range'].notna()]
                # only for initial dataset with wrong direction label
                if wrong_direction_label:
                    if 'left' in stimulus_name:
                        df.loc[
                            stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.RIGHT.value  # initally inverted
                    elif 'right' in stimulus_name:
                        df.loc[
                            stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.LEFT.value  # initally inverted
            if compute_correct_bout:
                df['correct_bout'] = BehavioralProcessing.compute_correct_bout_list(df, side_threshold=side_threshold, inverted_stimulus=False)

            df_dict[key] = df

        return df_dict

    @staticmethod
    def get_df_single_fish_for_parameter_analysis(
            path,
            analysed_parameter,
            # may be set to a bool or to a string with the name of the analysed parameter (in case of ranges the latter is mandatory)
            parameter_range_list=None,
            time_hours_offset=0,
            time_hours_to_analyse=2,
            sanity_check=True,
            debug=False,
            wrong_direction_label=False,
            compute_correct_bout=True,
            side_threshold=3
    ):
        df_dict = {}
        for fish_path in path.glob("*"):
            if (not fish_path.is_dir()):
                continue

            try:
                # df = PreAnalysis.preprocess_df_reactive(fish_path / f"{fish_path.name}.hdf5")  # #####
                # df = PreAnalysis.get_df_alt_stim(fish_path / f"{fish_path.name}.hdf5")  # #####
                df = PreAnalysis.get_df(fish_path / f"{fish_path.name}.hdf5")  # #####
                df_dict[fish_path.name] = df
            except FileNotFoundError:
                print(f"INFO | file {str(fish_path)}/{fish_path.name}.hdf5 not found")

        for key, df in df_dict.items():
            if sanity_check:
                df = BehavioralProcessing.sanity_check(df, max_interbout_interval=30)

            # slice desired hours out of experiment
            df['start_time_absolute'] = df['start_time_absolute'].astype(
                np.int64) * 1e-9  # convert to POSIX timestamp in seconds
            df['end_time_absolute'] = df['end_time_absolute'].astype(
                np.int64) * 1e-9  # convert to POSIX timestamp in seconds
            time_start_experiment = df['start_time_absolute'].min()
            df['start_time_absolute'] -= time_start_experiment
            df['end_time_absolute'] -= time_start_experiment
            query_time = f"start_time_absolute > {Converter.hour2sec(time_hours_offset)} and end_time_absolute < {Converter.hour2sec(time_hours_offset + time_hours_to_analyse)}"
            df = df.query(query_time)
            if df.empty:
                raise "WARNING | plotting | define dataframe | dataframe is empty"
            if debug:
                print(f"DEBUG | plotting | define dataframe | min start time absolute: {df['start_time_absolute'].min()}")
                print(f"DEBUG | plotting | define dataframe | max start time absolute: {df['start_time_absolute'].max()}")

            # add stimulus parameters columns
            stimulus_name_list = df.index.unique('stimulus_name')
            BehavioralProcessing.add_empty_parameter_column_list(df)
            for stimulus_name in stimulus_name_list:
                stimulus_parameter_dict = BehavioralProcessing.parse_stimulus_name(stimulus_name)  # , stimulus_enum=StimulusParameterLabel2Dir)
                stimulus_index_list = df.index.get_level_values('stimulus_name') == stimulus_name
                if analysed_parameter is not None and analysed_parameter is not False:
                    if parameter_range_list is None:
                        for parameter in stimulus_parameter_dict:
                            df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                    else:
                        for parameter in stimulus_parameter_dict:
                            df.loc[stimulus_index_list, parameter] = stimulus_parameter_dict[parameter]
                            if parameter == analysed_parameter:
                                try:
                                    df.loc[stimulus_index_list, 'parameter_range'] = [parameter_range['label']
                                                                                      for parameter_range in
                                                                                      parameter_range_list
                                                                                      if
                                                                                      float(parameter_range['min']) < float(
                                                                                          stimulus_parameter_dict[
                                                                                              parameter]) <= float(
                                                                                          parameter_range['max'])][0]
                                except IndexError:
                                    print(
                                        f"WARNING | plotting | define dataframe | parameter {parameter} has value {stimulus_parameter_dict[parameter]} out of analysed ranges")
                        df = df[df['parameter_range'].notna()]
                # only for initial dataset with wrong direction label
                if wrong_direction_label:
                    if 'left' in stimulus_name:
                        df.loc[
                            stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.RIGHT.value  # initally inverted
                    elif 'right' in stimulus_name:
                        df.loc[
                            stimulus_index_list, StimulusParameterLabel.DIRECTION.value] = Keyword.LEFT.value  # initally inverted
            if compute_correct_bout:
                df['correct_bout'] = BehavioralProcessing.compute_correct_bout_list(df, side_threshold=side_threshold, inverted_stimulus=True)

            df_dict[key] = df

        return df_dict

    @classmethod
    def exclude_outlier_fish(
            cls,
            df,
            time_start_stimulus,
            time_end_stimulus,
            analysed_parameter=StimulusParameterLabel.COHERENCE.value,
            side_threshold=3  # degrees
    ):
        query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'
        if CorrectBoutColumn not in df.columns:
            df[CorrectBoutColumn] = BehavioralProcessing.compute_correct_bout_list(df, side_threshold=side_threshold)

        df_stimulus = df[df[CorrectBoutColumn] != -1].query(query_time)
        accuracy_all_fish = {}
        accuracy = {}
        parameter_list_fish = np.sort(df_stimulus[analysed_parameter].unique())
        for parameter in parameter_list_fish:
            accuracy[parameter] = {}
            df_parameter = df_stimulus[df_stimulus[analysed_parameter] == parameter]
            fish_id_list = df_parameter.index.unique(level='fish_ID')
            for fish_id in fish_id_list:
                df_filtered = df_parameter.xs(fish_id, level='fish_ID')
                accuracy[parameter][fish_id] = np.mean(df_filtered['correct_bout'])
            accuracy_all_fish[parameter] = {}
            accuracy_all_fish[parameter]['mean'] = np.mean(list(accuracy[parameter].values()))
            accuracy_all_fish[parameter]['std'] = np.std(list(accuracy[parameter].values()))

        fish_to_exclude = []
        fish_id_list = df.index.unique(level='fish_ID')
        for fish_id in fish_id_list:
            condition_list = [np.abs(accuracy[parameter][fish_id] - accuracy_all_fish[parameter]['mean']) > 3 * accuracy_all_fish[parameter]['std']
                              for parameter in accuracy.keys() if fish_id in accuracy[parameter].keys()]
            if any(condition_list):
                fish_to_exclude.append(fish_id)
        print(f"INFO | {cls.tag} | exclude_outlier_fish | excluded fish: {fish_to_exclude}")
        if len(fish_to_exclude) > 0:
            df = df.drop(fish_to_exclude, level='fish_ID')
        return df


    @staticmethod
    def parse_stimulus_name(stimulus_name: str, stimulus_enum = None):
        if stimulus_enum is None:
            stimulus_enum = StimulusParameterLabel
        parameter_list = stimulus_name.split(stimulus_enum.SEPARATOR.value)
        parameter_dict = {}
        for parameter in parameter_list:
            for label in islice(stimulus_enum, 1000):
                if parameter.find(label.value) == 0:
                    try:
                        parameter_dict[label.value] = float(parameter.replace(label.value, ''))
                    except ValueError:
                        parameter_dict[label.value] = parameter.replace(label.value, '')
                    break
        return parameter_dict

    @staticmethod
    def add_empty_parameter_column_list(df: pd.DataFrame, parameter_dict: dict = None):
        if parameter_dict is None:
            parameter_dict = {label.name: label.value for label in StimulusParameterLabel if
                              label.value != StimulusParameterLabel.SEPARATOR.value}
        for parameter in parameter_dict.values():
            if parameter not in df.columns:
                df[parameter] = None

    @staticmethod
    def compute_correct_bout_list(df, side_threshold=0, inverted_stimulus=False, simmetric=False, column_dir=StimulusParameterLabel.DIRECTION.value):
        # if simmetric is set to True
        #    - label_corr = 1
        #    - label_straight = 0
        #    - label_err = -1
        # else
        #    - label_corr = 1
        #    - label_straight = -1
        #    - label_err = 0

        sign_correction = -1 if inverted_stimulus else 1
        correct_bout_list = np.sign(np.array(df["estimated_orientation_change"]) * np.array(df[column_dir])) * sign_correction
        if not simmetric:
            correct_bout_list = np.sign(1+correct_bout_list)

        index_straight_bout = np.argwhere(abs(df['estimated_orientation_change']) <= side_threshold)
        correct_bout_list[index_straight_bout] = 0 if simmetric else -1
        return correct_bout_list

    # substitute with this for rest state analysis
    # @staticmethod
    # def compute_correct_bout_list(df, side_threshold=0):
    #     correct_bout_list = []
    #     for index, row in df.iterrows():
    #         # if abs(row['estimated_orientation_change']) <= side_threshold:
    #         #     correct_bout_list.append(0)
    #         if row['dir'] == str(Keyword.LEFT.value):
    #             correct_bout_list.append(1 if row['estimated_orientation_change'] > side_threshold else 0)
    #         elif row['dir'] == str(Keyword.RIGHT.value):
    #             correct_bout_list.append(1 if row['estimated_orientation_change'] < side_threshold else 0)
    #     return correct_bout_list

    @staticmethod
    def compute_bout_direction_list_discrete(df, side_threshold=0, left_value=Direction.LEFT.value, right_value=Direction.RIGHT.value, straight_value=Direction.STRAIGHT.value):
        bout_direction_list = []
        for index, row in df.iterrows():
            if row['estimated_orientation_change'] > side_threshold:
                bout_direction_list.append(left_value)
            elif row['estimated_orientation_change'] < side_threshold:
                bout_direction_list.append(right_value)
            else:
                bout_direction_list.append(straight_value)
        return bout_direction_list

    @staticmethod
    def compute_bout_direction_list_2dir(df):
        dir0 = np.array(df["dir0"], dtype=float) / 180 * np.pi
        x0 = np.cos(dir0) * np.array(df["coh"], dtype=float) / 100
        y0 = np.sin(dir0) * np.array(df["coh"], dtype=float) / 100
        dir1 = np.array(df["dir1"], dtype=float) / 180 * np.pi
        x1 = np.cos(dir1) * (100 - np.array(df["coh"], dtype=float)) / 100
        y1 = np.sin(dir1) * (100 - np.array(df["coh"], dtype=float)) / 100
        x = x0 + x1
        y = y0 + y1
        dir = np.arctan2(y, x)
        dir_final = (dir / np.pi * 180)
        df["dir"] = dir_final
        return df

    @classmethod
    def windowing_column(
            cls,
            df,
            column_name,
            window_size=1,
            window_step_size=0.01,
            window_operation='mean',
            time_column='end_time',
            time_start=None,
            time_end=None
    ):
        time_stamp_list, _ = cls.extract_time_stamp_list(df, time_column_name=time_column)
        if time_end is None:
            time_end = np.nanmax(time_stamp_list) + window_step_size / 2
        if time_start is None:
            time_start = np.nanmin(time_stamp_list) - window_step_size / 2
        window_start_time_list = np.arange(time_start, time_end-window_size/2+window_step_size, window_step_size)
        window_end_time_list = np.arange(time_start+window_size, time_end+window_step_size+window_size/2, window_step_size)
        window_center_list = (window_start_time_list + window_end_time_list) / 2
        windowed_array = []
        error_list = []
        for i_window_time in range(len(window_end_time_list)):
            window_start_time = window_start_time_list[i_window_time]
            window_end_time = window_end_time_list[i_window_time]
            if time_column is None:
                query_string = f"end_time > {window_start_time} and end_time <= {window_end_time}"
            else:
                query_string = f"{time_column} > {window_start_time} and {time_column} <= {window_end_time}"
            filtered_series = df.query(query_string)[column_name].dropna()
            if window_operation == 'mean_multiple_fish':
                try:
                    window_value_groups = filtered_series.groupby("experiment_ID").mean()
                except KeyError:
                    try:
                        window_value_groups = filtered_series.groupby("fish_ID").mean()
                    except KeyError:
                        window_value_groups = df.query(query_string).groupby("fish_ID")[column_name].mean()
                window_value = float(window_value_groups.mean())
                error_value = window_value_groups.std() / len(window_value_groups) if len(window_value_groups) > 0 else window_value_groups.std()
            if window_operation == 'mean':
                window_value = float(filtered_series.mean())
                error_value = filtered_series.std()  # / len(filtered_series) if len(filtered_series) > 0 else filtered_series.std()
            elif window_operation == 'sum':
                window_value = float(filtered_series.sum())
                error_value = None
            elif window_operation == 'count':
                window_value = float(len(filtered_series))
                error_value = None
            elif window_operation is None:
                window_value = np.array(filtered_series)
                error_value = None
            else:
                NotImplementedError()
            windowed_array.append(window_value)
            error_list.append(error_value)
        if window_operation == None:
            max_size_array = np.max([len(w) for w in windowed_array])
            for i_w in range(len(windowed_array)):
                windowed_array[i_w] = np.pad(windowed_array[i_w], (0, max_size_array - len(windowed_array[i_w])), constant_values=(np.nan,))
        return np.array(windowed_array), window_center_list, np.array(error_list)

    @classmethod
    def windowed_distribution_column(
            cls,
            df,
            column_name,
            window_size=1,
            window_step_size=0.01,
            time_column='end_time',
            time_start=None,
            time_end=None,
            number_bins=10,
            hist_range=None
    ):
        if hist_range is None:
            hist_range = (df[column_name].min(), df[column_name].max())
        time_stamp_list, _ = cls.extract_time_stamp_list(df, time_column_name=time_column)
        if time_end is None:
            time_end = time_stamp_list.max() + window_step_size / 2
        if time_start is None:
            time_start = time_stamp_list.min() - window_step_size / 2
        window_end_time_list = np.arange(time_start, time_end, window_step_size)
        windowed_array = np.zeros(shape=[len(window_end_time_list), number_bins])
        for i_w, window_end_time in enumerate(window_end_time_list[1:]):
            if time_column is None:
                query_string = f"end_time > {window_end_time - window_size} and end_time <= {window_end_time}"
            else:
                query_string = f"{time_column} > {window_end_time - window_size} and {time_column} <= {window_end_time}"
            filtered_series = df.query(query_string)[column_name].dropna()
            windowed_array[i_w], bins = StatisticsService.get_hist(filtered_series, bins=number_bins,
                                                                  hist_range=hist_range, center_bin=True, density=True)
        return windowed_array, bins, window_end_time_list

    # like windowing_column() method, but the parameter over which windowing is applied is not fixed to a time column,
    # it can be changed into any column name
    @classmethod
    def windowing_column_general(
            cls,
            df,
            column_name,
            window_size=1,
            window_step_size=0.01,
            window_operation='mean',
            windowing_column=None,
    ):
        if windowing_column is None:
            windowing_default = True
            data_list, windowing_column = cls.extract_time_stamp_list(df, time_column_name='start_time')
        else:
            data_list = df[windowing_column]
            windowing_default = False
        window_end_list = np.arange(0, data_list.max() + window_step_size / 2, window_step_size)
        windowed_array = []
        for window_end in window_end_list:
            if windowing_default:
                query_string = f"end_time > {window_end - window_size} and end_time <= {window_end}"
            else:
                query_string = f"{windowing_column} > {window_end - window_size} and {windowing_column} <= {window_end}"
            filtered_series = df.query(query_string)[column_name]
            if window_operation == 'mean':
                window_value = filtered_series.mean()
            elif window_operation == 'sum':
                window_value = filtered_series.sum()
            else:
                NotImplementedError()
            windowed_array.append(float(window_value))
        return windowed_array, window_end_list

    @classmethod
    def windowing_arrays(
            cls,
            time_stamp_list,
            value_array,
            window_size=1,
            window_step_size=0.01,
            window_operation='mean',
            time_start=None,
            time_end=None,
            default_value=0
    ):
        if time_end is None:
            time_end = time_stamp_list.max() + window_step_size / 2
        if time_start is None:
            time_start = time_stamp_list.min() - window_step_size / 2
        window_end_time_list = np.arange(time_start, time_end, window_step_size)
        windowed_array = []
        error_list = []
        for window_end_time in window_end_time_list:
            index_in_window = np.argwhere(np.logical_and(time_stamp_list > window_end_time - window_size, time_stamp_list <= window_end_time))
            filtered_series = value_array[index_in_window]
            if len(filtered_series) == 0:
                window_value = default_value
                error_value = default_value
            else:
                if window_operation == 'mean':
                    window_value = filtered_series.mean()
                    error_value = filtered_series.std() / np.sqrt(len(filtered_series))
                elif window_operation == 'sum':
                    window_value = filtered_series.sum()
                    error_value = None
                else:
                    NotImplementedError()
            windowed_array.append(float(window_value))
            error_list.append(error_value)
        return np.array(windowed_array), window_end_time_list, np.array(error_list)

    @staticmethod
    def extract_time_stamp_list(df, time_column_name=None):
        if time_column_name is not None and time_column_name in df.columns:
            time_stamp_list = df[time_column_name]
            time_column = time_column_name
        elif 'time' in df.columns:
            time_stamp_list = df['time']
            time_column = 'time'
        elif 'time' in df.index.names:
            time_stamp_list = df.index.get_level_values('time')
            time_column = 'time'
        else:
            try:
                time_stamp_list = df['end_time']
                time_column = 'end_time'
            except KeyError:
                time_stamp_list = []
                time_column = None
        return time_stamp_list, time_column

    @staticmethod
    def quantity_bout_in_trial(df, quantity: str, index_list_in_trial: list, time_window_start=0,
                               time_window_end=np.inf):
        quantity_dict = {}
        for index in index_list_in_trial:
            quantity_dict[index] = []
            for trial in df.index.unique('trial'):
                index_list_trial = df.index.get_level_values('trial') == trial
                query_string = f"start_time > {time_window_start} and end_time <= {time_window_end}"
                df_filtered = df.iloc[index_list_trial].query(query_string).sort_values(by=['start_time'])
                try:
                    quantity_dict[index].append(list(df_filtered[quantity])[index])
                except IndexError:
                    print(
                        f"WARNING | behavioral_processing | accuracy_bout_in_trial | could not find bout number {index} in trial {trial}")
        return quantity_dict

    @staticmethod
    def value_bout_in_trial(df, index_list_in_trial: list, column_name=CorrectBoutColumn, time_window_start=0, time_window_end=np.inf):
        correct_bout_dict = {}
        try:
            fish_id_list = df.index.unique('experiment_ID')
        except KeyError:
            try:
                fish_id_list = df['experiment_ID'].unique()
            except:
                fish_id_list = df['fish_ID'].unique()

        for index in index_list_in_trial:
            correct_bout_dict[index] = []
            for fish_id in fish_id_list:
                fish_accuracy_list = []
                try:
                    df_fish = df.xs(fish_id, level='experiment_ID')
                except (KeyError, TypeError):
                    try:
                        df_fish = df[df['experiment_ID'] == fish_id]
                    except KeyError:
                        df_fish = df[df['fish_ID'] == fish_id]
                try:
                    trial_list = df_fish.index.unique('trial')
                except KeyError:
                    trial_list = df_fish['trial'].unique()
                for trial in trial_list:
                    try:
                        df_filtered = df_fish.xs(trial, level='trial')
                    except (TypeError, KeyError):
                        df_filtered = df_fish[df_fish['trial'] == trial]
                    query_string = f"start_time > {time_window_start} and end_time <= {time_window_end}"
                    df_filtered = df_filtered.query(query_string).sort_values(by=['start_time'])
                    if len(df_filtered) > index:
                        fish_accuracy_list.append(df_filtered.reset_index()[column_name][index])
                fish_accuracy = np.mean(fish_accuracy_list)
                try:
                    correct_bout_dict[index].append(fish_accuracy)
                except IndexError:
                    print(f"WARNING | behavioral_processing | accuracy_bout_in_trial | could not find bout number {index} for fish {fish_id}")
        return correct_bout_dict

    @staticmethod
    def value_bout_in_trial_single_fish(df_fish, index_list_in_trial: list, column_name=CorrectBoutColumn, time_window_start=0,
                            time_window_end=np.inf):
        correct_bout_dict = {}
        for index in index_list_in_trial:
            fish_accuracy_list = []
            try:
                trial_list = df_fish.index.unique('trial')
            except KeyError:
                trial_list = df_fish['trial'].unique()
            for trial in trial_list:
                try:
                    df_filtered = df_fish.xs(trial, level='trial')
                except (TypeError, KeyError):
                    df_filtered = df_fish[df_fish['trial'] == trial]
                query_string = f"start_time > {time_window_start} and end_time <= {time_window_end}"
                df_filtered = df_filtered.query(query_string).sort_values(by=['start_time'])
                if len(df_filtered) > index:
                    fish_accuracy_list.append(df_filtered.reset_index()[column_name][index])
            try:
                correct_bout_dict[index] = fish_accuracy_list
            except IndexError:
                print(
                    f"WARNING | behavioral_processing | accuracy_bout_in_trial | could not find bout number {index}")
        return correct_bout_dict

    @classmethod
    def accuracy_bout_in_trial(cls, df, index_list_in_trial: list, time_window_start=0, time_window_end=np.inf):
        return cls.value_bout_in_trial(df, index_list_in_trial, column_name=CorrectBoutColumn, time_window_start=time_window_start, time_window_end=time_window_end)

    @staticmethod
    def add_column_from_previous_trial(df, column=StimulusParameterLabel.DIRECTION.value):
        value_previous_trial = np.zeros(len(df))
        try:
            df_sort = df.sort_values(["experiment_ID", "trial_count_since_experiment_start"])
            i_trial_index = df_sort.index.names.index("trial_count_since_experiment_start")
            def get_trial(i_row, row):
                return i_row[i_trial_index]
        except KeyError:
            df_sort = df.sort_values(["experiment_ID", "trial"])
            i_trial_index = None
            def get_trial(i_row, row):
                return row["trial"]

        i_in_df = 0
        for i_row, row in df_sort.iterrows():
            present_trial_temp = get_trial(i_row, row)

            if i_in_df == 0:
                initial_trial = present_trial_temp
                present_trial = initial_trial
                present_value = row[column]

            if present_trial_temp != present_trial:
                previous_value = present_value
                present_value = row[column]

            present_trial = present_trial_temp

            if present_trial == initial_trial:
                value_previous_trial[i_in_df] = None
            else:
                value_previous_trial[i_in_df] = previous_value
            i_in_df += 1

        df_sort["value_previous_trial"] = value_previous_trial
        return df_sort

    @classmethod
    def accuracy_bout_with_previous_trial(cls, df, side_threshold=3):
        df_sort = cls.add_column_from_previous_trial(df, StimulusParameterLabel.DIRECTION.value)
        df_sort["accuracy_previous_trial"] = BehavioralProcessing.compute_correct_bout_list(df_sort, side_threshold=side_threshold, column_dir="value_previous_trial")
        return df_sort

    @classmethod
    def sanity_check_reactive(
            cls,
            df,
            # Sanity check parameters
            max_interbout_interval=20,  # seconds
            max_ratio_false_interbout_interval=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_orientation_change=150,  # deg
            max_ratio_false_orientation_change=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_radius_fraction=1,  # for dish with radius from 0 to 1.
            max_ratio_false_radius_fraction=0.05,
    ):
        """Perform sanity checks"""
        print("Sanity check")

        # Add radius and absolute estimated orientation change
        with pd.option_context("mode.chained_assignment", None):
            df['abs_estimated_orientation_change'] = df['estimated_orientation_change'].abs()

        # Exclude bouts with a NaN value
        df = df[df['estimated_orientation_change'].notnull()]

        # Get number of trials at start
        n_trials_start = len(df.groupby(cls.group_by_idxs))

        # Interbout interval
        prop_name = 'reaction_time'
        df = cls._sanity_check(df, prop_name, max_interbout_interval, max_ratio_false_interbout_interval)

        # Pirouettes (large estimated_orientation_change)
        prop_name = 'abs_estimated_orientation_change'
        df = cls._sanity_check(df, prop_name, max_orientation_change, max_ratio_false_orientation_change)

        # Too close to border
        prop_name = 'radius'
        df = cls._sanity_check(df, prop_name, max_radius_fraction, max_ratio_false_radius_fraction)

        # Print results
        n_dropped_trials = n_trials_start - len(df.groupby(cls.group_by_idxs))
        print(f'\tTotal dropped: {n_dropped_trials / n_trials_start * 100:.2f} % ({n_dropped_trials} trials dropped)')

        return df

    # this sanity check has been derived from the one in the standard pipeline
    @classmethod
    def sanity_check(
            cls,
            df,
            show_plots=False,
            # Sanity check parameters
            max_interbout_interval=20,  # seconds
            max_ratio_false_interbout_interval=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_speed=1,  # cm/s?  # TODO: find proper unit
            max_ratio_false_speed=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_contour_area=2000,  # pixels^2
            max_ratio_false_contour_area=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_orientation_change=150,  # deg
            max_ratio_false_orientation_change=0.05,  # 5% of all bouts within a trial can be bigger than max_value
            max_radius_fraction=1,  # for dish with radius from 0 to 1.
            max_ratio_false_radius_fraction=0.05,
    ):
        """Perform sanity checks"""
        print("Sanity check")

        # Add radius and absolute estimated orientation change
        with pd.option_context("mode.chained_assignment", None):
            df['radius'] = np.sqrt(df['end_x_position'] ** 2 + df['end_y_position'] ** 2)
            df['abs_estimated_orientation_change'] = df['estimated_orientation_change'].abs()

        # Exclude bouts with a NaN value
        # df = df.dropna()
        df = df[df['estimated_orientation_change'].notnull()]
        # print(f'\t{(df.size - df.size) / df.size * 100:.2f} % ({df.size - df.size} bouts dropped)\t NaN')

        # Get number of trials at start
        n_trials_start = len(df.groupby(cls.group_by_idxs))

        # Interbout interval
        prop_name = 'interbout_interval'
        df = cls._sanity_check(df, prop_name, max_interbout_interval, max_ratio_false_interbout_interval)
        # if show_plots:
        #     cls._plot(prop_name, max_interbout_interval, 'IBI [s]')  # For illustration purposes

        # Jumping
        prop_name = 'average_speed'
        df = cls._sanity_check(df, prop_name, max_speed, max_ratio_false_speed)
        # if show_plots:
        #     cls._plot(prop_name, max_speed, 'Speed [cm/s?]')  # For illustration purposes

        # Fish contour area
        prop_name = 'end_contour_area'
        df = cls._sanity_check(df, prop_name, max_contour_area, max_ratio_false_contour_area)
        # if show_plots:
        #     cls._plot(prop_name, max_contour_area, 'Contour area [px^2]')  # For illustration purposes

        # Pirouettes (large estimated_orientation_change)
        prop_name = 'abs_estimated_orientation_change'
        df = cls._sanity_check(df, prop_name, max_orientation_change, max_ratio_false_orientation_change)
        # if show_plots:
        #     cls._plot(prop_name, max_orientation_change, 'Abs orientation change\n[deg]')  # For illustration purposes

        # Too close to border
        prop_name = 'radius'
        df = cls._sanity_check(df, prop_name, max_radius_fraction, max_ratio_false_radius_fraction)
        # if show_plots:
        #     cls._plot(prop_name, max_radius_fraction, 'Radius [-]', bin_min=0, bin_max=1)  # For illustration purposes
        print(df)  # ##### DEBUG
        # Remove the columns we added
        # df.drop('radius', inplace=True)
        # df.drop('abs_estimated_orientation_change', inplace=True)

        # Print results
        n_dropped_trials = n_trials_start - len(df.groupby(cls.group_by_idxs))
        print(f'\tTotal dropped: {n_dropped_trials / n_trials_start * 100:.2f} % ({n_dropped_trials} trials dropped)')

        return df

    @classmethod
    def _sanity_check(cls, df, prop_name, max_value, max_ratio_false):
        """Remove all trials with a too high fraction of "wrong" bouts"""
        # Create group for each trial
        if df.empty:
            new_df = df
        else:
            grouped = df.groupby(cls.group_by_idxs)

            # Remove all trials with a too high fraction of "wrong" bouts
            new_df = grouped.filter(lambda x: (sum(x[prop_name] > max_value) / len(x[prop_name])) <= max_ratio_false)

            # Print results
            n_dropped_trials = len(grouped) - len(new_df.groupby(cls.group_by_idxs))
            print(f'\t{n_dropped_trials / len(grouped) * 100:.2f} % ({n_dropped_trials} trials dropped)\t{prop_name}')
        return new_df

    @classmethod
    def compute_accuracy_index(cls, df_signal, df_baseline, time_start_stimulus, time_end_stimulus, accuracy_parameter='correct_bout'):
        query_time_stimulus = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'
        query_time_rest = f'start_time < {time_start_stimulus} or end_time > {time_end_stimulus}'
        signal_stimulus = BehavioralProcessing.windowing_column(
            df_signal.query(query_time_stimulus),
            accuracy_parameter,
            window_step_size=0.1,
            window_operation='mean',
            time_start=time_start_stimulus,
            time_end=time_end_stimulus
        )[0]
        signal_stimulus = signal_stimulus[~np.isnan(signal_stimulus)]
        signal_rest = np.concatenate((BehavioralProcessing.windowing_column(
            df_signal.query(query_time_rest),
            accuracy_parameter,
            window_step_size=0.1,
            window_operation='mean',
            time_start=0,
            time_end=time_start_stimulus
        )[0],
        BehavioralProcessing.windowing_column(
          df_signal.query(query_time_rest),
          accuracy_parameter,
          window_step_size=0.1,
          window_operation='mean',
          time_start=time_end_stimulus
        )[0]))
        signal_rest = signal_rest[~np.isnan(signal_rest)]
        baseline_stimulus = BehavioralProcessing.windowing_column(
            df_baseline.query(query_time_stimulus),
            accuracy_parameter,
            window_step_size=0.1,
            window_operation='mean',
            time_start=time_start_stimulus,
            time_end=time_end_stimulus
        )[0]
        baseline_stimulus = baseline_stimulus[~np.isnan(baseline_stimulus)]
        baseline_rest = np.concatenate((BehavioralProcessing.windowing_column(
            df_baseline.query(query_time_rest),
            accuracy_parameter,
            window_step_size=0.1,
            window_operation='mean',
            time_start=0,
            time_end=time_start_stimulus
        )[0],
        BehavioralProcessing.windowing_column(
            df_baseline.query(query_time_rest),
            accuracy_parameter,
            window_step_size=0.1,
            window_operation='mean',
            time_start=time_end_stimulus
        )[0]
        ))
        baseline_rest = baseline_rest[~np.isnan(baseline_rest)]
        accuracy_index_stimulus = sum([abs(signal_stimulus[i] - baseline_stimulus[i]) for i in range(min(len(signal_stimulus), len(baseline_stimulus)))])
        accuracy_index_rest = sum([abs(signal_rest[i] - baseline_rest[i]) for i in range(min(len(signal_rest), len(baseline_rest)))])
        accuracy_index = accuracy_index_stimulus / accuracy_index_rest
        return accuracy_index

    @staticmethod
    def resample_data(x, y, x_resample_list):
        y_resample_list = []
        for x_resample in x_resample_list:
            x_value_greater, x_index_greater = StatisticsService.closest(x, x_resample, sign='greater_than')
            x_value_less, x_index_less = StatisticsService.closest(x, x_resample, sign='less_than')
            y_resample = np.mean([y[x_index_greater], y[x_index_less]])
            y_resample_list.append(y_resample)
        return y_resample_list

    @staticmethod
    def flip_column(df, column_name, inverted_direction=False) -> list:
        sign_factor = -1 if inverted_direction else 1
        return np.array(df[column_name]) * np.sign(np.array(df[StimulusParameterLabel.DIRECTION.value], dtype=float)) * sign_factor

    @staticmethod
    def remove_fast_straight_bout(df, threshold_response_time=0.1):
        try:
            invalid_condition = np.array(StatisticsService.and_list(df[CorrectBoutColumn] == StraightBout,
                                                                    df[ResponseTimeColumn] < threshold_response_time))
        except KeyError:
            try:
                invalid_condition = np.array(StatisticsService.and_list(df[CorrectBoutColumn] == StraightBout,
                                                                    df['reaction_time'] < threshold_response_time))
            except KeyError:
                invalid_condition = np.array(StatisticsService.and_list(df[CorrectBoutColumn] == StraightBout,
                                                                        df['response_delay'] < threshold_response_time))
        return df.iloc[~invalid_condition]

    @staticmethod
    def get_straight_turn_GMM(GMM):
        #
        # INPUT
        # GMM: output from GaussianMixture(n_components=2, random_state=0).fit(...)
        #
        # OUTPUT
        # tuple of dicts containing mean, variance and weighting factor of each component of this GMM
        #
        # to distinguish between straight and turn bouts, I assume there is a greater amount of the first kind,
        # thus having a stronger impact on the model
        index_straight = GMM.weights_.argmax()
        index_turn = GMM.weights_.argmax()
        GMM_straight = {
            "mean": GMM.means_[index_straight],
            "var": GMM.covariances_[index_straight],
            "scale": GMM.weights_[index_straight]
        }
        GMM_turn = {
            "mean": GMM.means_[index_turn],
            "var": GMM.covariances_[index_turn],
            "scale": GMM.weights_[index_turn]
        }
        return GMM_straight, GMM_turn

    @classmethod
    def add_parameter_from_model(cls, model_parameters, target_dict, label='param'):
        for key in model_parameters.keys():
            label = f"{label}_{key}"
            if isinstance(model_parameters[key], dict):
                cls.add_parameter_from_model(model_parameters, target_dict, label)
            else:
                target_dict[label] = model_parameters[key]

    @classmethod
    def get_duration_trials_in_df(cls, df, properties_to_group_by: List[str] = None, time_column="end_time", fixed_time_trial: float = None):
        if properties_to_group_by is None:
            properties_to_group_by = ["folder_name", "trial_count_since_experiment_start", "trial"]
        # exclude terms that are not in index (maybe dataframe has already be filtered by index)
        properties_to_group_by_in_df = [p for p in properties_to_group_by if p in df.index.names]
        diff_list = list(set(properties_to_group_by) - set(properties_to_group_by_in_df))
        if len(diff_list) > 0:
            print(f"WARNING | {cls.tag} | get_duration_trials_in_df | df does not contain the following indices: {', '.join(diff_list)}")
        if len(properties_to_group_by_in_df) == 0:
            time_list = len(df["trial"].unique()) * fixed_time_trial
        elif fixed_time_trial is None:
            time_list = df.groupby(level=properties_to_group_by_in_df).max()[time_column]
        else:
            time_list = df.groupby(level=properties_to_group_by_in_df).count().shape[0] * [fixed_time_trial]
        return time_list

    @staticmethod
    def fit_to_logistic(df_train=None, x=None, y=None, x_column="start_time", y_column=CorrectBoutColumn, fitting_method="lm"):
        if df_train is not None:
            df = df_train.sort_values(x_column)
            x = np.array(df[x_column])
            y = np.array(df[y_column])

        x = np.array(x)
        y = np.array(y)
        def sigmoid(x, L, x0, k, b):
            y = L / (1 + np.exp(-k * (x - x0))) + b
            return (y)

        initial_guess = [max(y), np.median(x), 1, min(y)]  # initial guess (needed by curve_fit)
        return curve_fit(sigmoid, x, y, initial_guess, method=fitting_method)

    @staticmethod
    def fit_to_maxwell_boltzmann(df_train=None, x=None, y=None, x_column="start_time", y_column=CorrectBoutColumn,
                                 fitting_method="lm"):
        if df_train is not None:
            df = df_train.sort_values(x_column)
            x = np.array(df[x_column])
            y = np.array(df[y_column])

        x = np.array(x)
        y = np.array(y)

        def maxwell_boltzmann(x, a):
            y = np.sqrt(2/np.pi) * x**2/a**3 * np.exp(-(x**2)/(2*(a**2)))
            y = y / np.sum(y)
            return (y)

        initial_guess = [1]  # initial guess (needed by curve_fit)
        return curve_fit(maxwell_boltzmann, x, y, initial_guess, method=fitting_method)

    @staticmethod
    def fit_to_poisson(df_train=None, x=None, y=None, x_column="start_time", y_column=CorrectBoutColumn, fitting_method="lm"):
        if df_train is not None:
            df = df_train.sort_values(x_column)
            x = np.array(df[x_column])
            y = np.array(df[y_column])

        x = np.array(x)
        y = np.array(y)

        def poisson_scaled(x, L):
            y = poisson.pmf(x, L)
            y = y / max(y)
            return (y)

        initial_guess = [2]

        return curve_fit(poisson_scaled, x, y, p0=initial_guess, method=fitting_method)

    @classmethod
    def compute_focus_scope(cls, data_0, data_1=[]):
        try:
            focus_scope = (np.min(np.concatenate((data_0, data_1))), np.max(np.concatenate((data_0, data_1))))
        except ValueError:
            try:
                focus_scope = (np.min(data_0), np.max(data_0))
            except ValueError:
                try:
                    focus_scope = (np.min(data_1), np.max(data_1))
                except ValueError:
                    focus_scope = 0  # it means both response_time_list_0 and response_time_list_1 are empty
        return focus_scope

    @classmethod
    def kl_divergence_rt_distribution_weight(cls, response_time_list_0, response_time_list_1, duration_0=None, duration_1=None, resolution=10, focus_scope=None, order_sum_result=False, order_max_result=False, smoothing=None, correct_by_area=False, plot_distributions=False, perturbation=None):
        smoothing_1 = None
        if focus_scope is None:
            focus_scope = cls.compute_focus_scope(response_time_list_0, response_time_list_1)
        if len(response_time_list_0) == 0:
            distribution_response_time_list_0 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_0, bin_list_0 = StatisticsService.get_hist(
                response_time_list_0, bins=resolution, hist_range=focus_scope,
                duration=duration_0, allow_zero=False, center_bin=True)
            if smoothing is not None:
                if smoothing["is_simmetric"] == True or smoothing["is_simmetric"] is None:
                    smoothing_1 = smoothing
                if smoothing["label"] == "savitzky_golay":
                    distribution_response_time_list_0 = SignalProcessing.savitzky_golay(distribution_response_time_list_0, smoothing["window_size"], smoothing["polynomial_order"])
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
                elif smoothing["label"] == "polynomial":
                    p = np.polyfit(bin_list_0, distribution_response_time_list_0, smoothing["deg"])
                    distribution_response_time_list_0 = np.polyval(p, bin_list_0)
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
        if len(response_time_list_1) == 0:
            distribution_response_time_list_1 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_1, bin_list_1 = StatisticsService.get_hist(
                response_time_list_1, bins=resolution, hist_range=focus_scope,
                duration=duration_1, allow_zero=False, center_bin=True)
            if smoothing_1 is not None:
                if smoothing_1["label"] == "savitzky_golay":
                    distribution_response_time_list_1 = SignalProcessing.savitzky_golay(distribution_response_time_list_1, smoothing_1["window_size"], smoothing_1["polynomial_order"])
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps
                elif smoothing_1["label"] == "polynomial":
                    p = np.polyfit(bin_list_1, distribution_response_time_list_1, smoothing_1["deg"])
                    distribution_response_time_list_1 = np.polyval(p, bin_list_1)
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps

        if perturbation is not None:
            distribution_response_time_list_0 = distribution_response_time_list_0 + perturbation
        distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps

        weight_array = distribution_response_time_list_0 / np.sum(distribution_response_time_list_0)

        KL_divergence_response = sum(np.abs(rel_entr(distribution_response_time_list_0, distribution_response_time_list_1)) * weight_array)
        if correct_by_area:
            area_0 = simpson(distribution_response_time_list_0, dx=(focus_scope[1]-focus_scope[0])/resolution)
            KL_divergence_response = KL_divergence_response / np.sqrt(area_0)
        if order_max_result:
            KL_divergence_response_1 = sum(np.abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)) * weight_array)
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = max([KL_divergence_response_1, KL_divergence_response])

        elif order_sum_result:
            KL_divergence_response_1 = sum(np.abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)) * weight_array)
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = KL_divergence_response_1 + KL_divergence_response

        if plot_distributions:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            try:
                ax.plot(bin_list_0, distribution_response_time_list_0, color="k", alpha=0.5)
            except UnboundLocalError:
                pass
            try:
                ax.plot(bin_list_1, distribution_response_time_list_1, "--", color="blue")
            except UnboundLocalError:
                pass
            fig.show()

        return KL_divergence_response

    @classmethod
    def kl_divergence_rt_distribution(cls, response_time_list_0, response_time_list_1, duration_0=None, duration_1=None, resolution=10, focus_scope=None, order_sum_result=False, order_max_result=False, smoothing=None, correct_by_area=False, plot_distributions=False, perturbation=None):
        smoothing_1 = None
        if focus_scope is None:
            focus_scope = cls.compute_focus_scope(response_time_list_0, response_time_list_1)
        if len(response_time_list_0) == 0:
            distribution_response_time_list_0 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_0, bin_list_0 = StatisticsService.get_hist(
                response_time_list_0, bins=resolution, hist_range=focus_scope,
                duration=duration_0, allow_zero=False, center_bin=True)
            if smoothing is not None:
                if smoothing["is_simmetric"] == True or smoothing["is_simmetric"] is None:
                    smoothing_1 = smoothing
                if smoothing["label"] == "savitzky_golay":
                    distribution_response_time_list_0 = SignalProcessing.savitzky_golay(distribution_response_time_list_0, smoothing["window_size"], smoothing["polynomial_order"])
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
                elif smoothing["label"] == "polynomial":
                    p = np.polyfit(bin_list_0, distribution_response_time_list_0, smoothing["deg"])
                    distribution_response_time_list_0 = np.polyval(p, bin_list_0)
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
        if len(response_time_list_1) == 0:
            distribution_response_time_list_1 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_1, bin_list_1 = StatisticsService.get_hist(
                response_time_list_1, bins=resolution, hist_range=focus_scope,
                duration=duration_1, allow_zero=False, center_bin=True)
            if smoothing_1 is not None:
                if smoothing_1["label"] == "savitzky_golay":
                    distribution_response_time_list_1 = SignalProcessing.savitzky_golay(distribution_response_time_list_1, smoothing_1["window_size"], smoothing_1["polynomial_order"])
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps
                elif smoothing_1["label"] == "polynomial":
                    p = np.polyfit(bin_list_1, distribution_response_time_list_1, smoothing_1["deg"])
                    distribution_response_time_list_1 = np.polyval(p, bin_list_1)
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps

        if perturbation is not None:
            distribution_response_time_list_0 = distribution_response_time_list_0 + perturbation
        distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps

        KL_divergence_response = sum(abs(rel_entr(distribution_response_time_list_0, distribution_response_time_list_1)))
        if correct_by_area:
            area_0 = simpson(distribution_response_time_list_0, dx=(focus_scope[1]-focus_scope[0])/resolution)
            KL_divergence_response = KL_divergence_response / np.sqrt(area_0)
        if order_max_result:
            KL_divergence_response_1 = sum(abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)))
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = max([KL_divergence_response_1, KL_divergence_response])

        elif order_sum_result:
            KL_divergence_response_1 = sum(
                abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)))
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = KL_divergence_response_1 + KL_divergence_response

        if plot_distributions:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            try:
                ax.plot(bin_list_0, distribution_response_time_list_0, color="k", alpha=0.5)
            except UnboundLocalError:
                pass
            try:
                ax.plot(bin_list_1, distribution_response_time_list_1, "--", color="blue")
            except UnboundLocalError:
                pass
            fig.show()

        return KL_divergence_response

    @classmethod
    def rmse_rt_distribution(cls, response_time_list_0, response_time_list_1, duration_0=None, duration_1=None,
                             resolution=10, focus_scope=None, smoothing=None, noise_sigma=0):

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        smoothing_1 = None
        if focus_scope is None:
            focus_scope = cls.compute_focus_scope(response_time_list_0, response_time_list_1)
        if len(response_time_list_0) == 0:
            distribution_response_time_list_0 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_0, bin_list_0 = StatisticsService.get_hist(
                response_time_list_0, bins=resolution, hist_range=focus_scope,
                duration=duration_0, allow_zero=False, center_bin=True)
            if smoothing is not None:
                if smoothing["is_simmetric"] == True or smoothing["is_simmetric"] is None:
                    smoothing_1 = smoothing
                if smoothing["label"] == "savitzky_golay":
                    distribution_response_time_list_0 = SignalProcessing.savitzky_golay(
                        distribution_response_time_list_0, smoothing["window_size"], smoothing["polynomial_order"])
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
                elif smoothing["label"] == "polynomial":
                    p = np.polyfit(bin_list_0, distribution_response_time_list_0, smoothing["deg"])
                    distribution_response_time_list_0 = np.polyval(p, bin_list_0)
                    distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps
        if len(response_time_list_1) == 0:
            distribution_response_time_list_1 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_1, bin_list_1 = StatisticsService.get_hist(
                response_time_list_1, bins=resolution, hist_range=focus_scope,
                duration=duration_1, allow_zero=False, center_bin=True)
            if smoothing_1 is not None:
                if smoothing_1["label"] == "savitzky_golay":
                    distribution_response_time_list_1 = SignalProcessing.savitzky_golay(
                        distribution_response_time_list_1, smoothing_1["window_size"], smoothing_1["polynomial_order"])
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps
                elif smoothing_1["label"] == "polynomial":
                    p = np.polyfit(bin_list_1, distribution_response_time_list_1, smoothing_1["deg"])
                    distribution_response_time_list_1 = np.polyval(p, bin_list_1)
                    distribution_response_time_list_1[distribution_response_time_list_1 <= 0] = np.finfo(float).eps

        distribution_response_time_list_0 = distribution_response_time_list_0 + np.random.normal(scale=noise_sigma, size=len(distribution_response_time_list_0))
        distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps

        rmse_response = rmse(distribution_response_time_list_0, distribution_response_time_list_1)

        return rmse_response

    @staticmethod
    @jit(nopython=True)
    def compute_decision_variable_trajectory(time_decision_list, decision_variable_time, decision_variable_list, rounding_digits=3, threshold_value=1):
        if len(time_decision_list) == 0:
            return decision_variable_list, decision_variable_time

        # decision_variable_time = np.arange(time_start_stimulus, time_end_stimulus, dt)
        # time_list_extended = np.concatenate((np.array([float(time_start_stimulus)]), time_decision_list))
        decision_variable_trajectory = np.full(len(decision_variable_time), np.nan)
        number_events_list = np.zeros(len(decision_variable_time))
        # for i in range(len(time_list_extended)):
        #     time_list_extended[i] -= int(time_list_extended[i] / time_end_experiment) * time_end_experiment
        longest_decision = 0
        for index_decision in range(len(time_decision_list[:-1])):
            # time_start_window = round(time_list_extended[index_decision], rounding_digits)
            # time0 = time.time()
            index_time_points_in_decision = np.argwhere(
                np.logical_and(decision_variable_time < round(time_decision_list[index_decision + 1], rounding_digits),
                               decision_variable_time >= time_decision_list[index_decision]))[:, 0]
            # print(f"PROFILING | {index_decision} | 0 | {time.time() - time0}")
            # time0 = time.time()
            if len(index_time_points_in_decision) > 0:
                decision_list_in_trial = decision_variable_list[index_time_points_in_decision]
                for i in range(len(index_time_points_in_decision)):
                    if np.isnan(decision_variable_trajectory[i]):
                        decision_variable_trajectory[i] = 0
                        longest_decision = i
                    number_events_list[i] += 1
                    decision_variable_trajectory[i] += decision_list_in_trial[i]
            # print(f"PROFILING | {index_decision} | 1 | {time.time() - time0}")

        number_events_to_report = int(max(number_events_list))
        number_time_points_above_threshold = number_events_to_report - number_events_list[:longest_decision]
        output_decision = (decision_variable_trajectory[:longest_decision]).flatten()
        output_decision = ((output_decision + number_time_points_above_threshold * threshold_value) / number_events_to_report).flatten()
        output_time = decision_variable_time[:longest_decision].flatten()

        return output_decision, output_time

    @classmethod
    def process_time_dict(cls, time_dict, time_step, time_start=0):
        time_array = pd.DataFrame(time_dict).values
        return cls._process_time_dict(time_array, time_step, time_start)

    @staticmethod
    @jit(nopython=True)
    def _process_time_dict(time_array, time_step, time_start=0):
        duration_so_far = 0
        total_duration = 0
        for i in range(time_array.shape[0]):
            total_duration += time_array[i][1] / time_step
        time_list = np.zeros(int(total_duration))
        stimulus_signal = np.zeros(int(total_duration))
        for i in range(time_array.shape[0]):
            time_list_new = np.arange(time_start, time_start + time_array[i][1], time_step)
            stimulus_signal_new = [time_array[i][0] for _ in time_list_new]
            for j in range(len(stimulus_signal_new)):
                time_list[duration_so_far + j] = time_list_new[j]
                stimulus_signal[duration_so_far + j] = stimulus_signal_new[j]
            time_start += time_array[i][1]
            duration_so_far += len(stimulus_signal_new)
        return stimulus_signal, time_list

    @staticmethod
    def check_dataset_accepted(response_time_list, decision_list, limit_area=0.005, limit_area_ratio_0_100=0.75,
                               limit_area_ratio_100_100=0.30, duration=None, limit_peak_0=0.15, limit_area_100_plus=0.01,
                               correct_label=1, incorrect_label=0, parameter_list=None):
        if parameter_list is None:
            parameter_list = [0, 100]
        area_plus = {}
        area_minus = {}
        for parameter in parameter_list:
            # flatten lists over trials
            response_time_list_param = np.array(list(chain.from_iterable(response_time_list[parameter].values())))
            decision_list_param = np.array(list(chain.from_iterable(decision_list[parameter].values())), dtype=int)

            # compute rt-distribution
            index_corr = np.argwhere(decision_list_param == correct_label)
            index_err = np.argwhere(decision_list_param == incorrect_label)
            data_corr = response_time_list_param[index_corr.flatten()]
            data_err = response_time_list_param[index_err.flatten()]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   bins=np.arange(0, 2, 0.1),
                                                                                   duration=duration,
                                                                                   # bins=np.arange(0, 3, 0.03),
                                                                                   center_bin=True)
            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 bins=np.arange(0, 2, 0.1),
                                                                                 duration=duration,
                                                                                 # bins=np.arange(0, 3, 0.03),
                                                                                 center_bin=True)

            # compute area
            area_plus[parameter] = simpson(data_hist_value_corr, data_hist_time_corr)
            if parameter == 100 and area_plus[parameter] < limit_area_100_plus:
                return False

            area_minus[parameter] = simpson(data_hist_value_err, data_hist_time_err)
            if area_plus[parameter] <= limit_area * parameter/100 or area_minus[parameter] <= limit_area * (100 - parameter)/100:
                return False
            if duration is not None and parameter == 0 and (any(data_hist_value_corr > limit_peak_0) or any(data_hist_value_err > limit_peak_0)):
                return False

        # apply rule to check
        area_ratio_p0_p100 = area_plus[0] / area_plus[100]
        area_ratio_m0_m100 = area_minus[0] / area_minus[100]
        area_ratio_m100_p100 = area_minus[100] / area_plus[100]
        area_ratio_p100_m100 = area_plus[100] / area_minus[100]
        return (area_ratio_p0_p100 < limit_area_ratio_0_100 and area_ratio_m100_p100 < limit_area_ratio_100_100) or (area_ratio_m0_m100 < limit_area_ratio_0_100 and area_ratio_p100_m100 < limit_area_ratio_100_100)

    @staticmethod
    def check_dataset_accepted_minimal(response_time_list, decision_list, limit_area=0.005, correct_label=1, incorrect_label=0):
        parameter_list = response_time_list.keys()
        for parameter in parameter_list:
            # flatten lists over trials
            response_time_list_param = np.array(list(chain.from_iterable(response_time_list[parameter].values())))
            decision_list_param = np.array(list(chain.from_iterable(decision_list[parameter].values())), dtype=int)

            # compute rt-distribution
            index_corr = np.argwhere(decision_list_param == correct_label)
            index_err = np.argwhere(decision_list_param == incorrect_label)
            data_corr = response_time_list_param[index_corr.flatten()]
            data_err = response_time_list_param[index_err.flatten()]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   bins=100,
                                                                                   # bins=np.arange(0, 3, 0.03),
                                                                                   center_bin=True)
            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 bins=100,
                                                                                 # bins=np.arange(0, 3, 0.03),
                                                                                 center_bin=True)

            # compute area
            area_plus = simpson(data_hist_value_corr, data_hist_time_corr)
            # check
            if area_plus < limit_area:
                return False

            # compute area
            area_minus = simpson(data_hist_value_err, data_hist_time_err)
            # check
            if area_minus < limit_area:
                return False

        return True

    @staticmethod
    def check_dataset_accepted_dev(response_time_list, decision_list, limit_area=0.005, duration=None, limit_peak_0=0.15, limit_area_100_plus=0.01):
        parameter_list = [0, 100, 25]
        area_plus = {}
        area_minus = {}
        area_straight = {}
        for parameter in parameter_list:
            # flatten lists over trials
            response_time_list_param = np.array(list(chain.from_iterable(response_time_list[parameter].values())))
            decision_list_param = np.array(list(chain.from_iterable(decision_list[parameter].values())), dtype=int)

            # compute rt-distribution
            index_corr = np.argwhere(decision_list_param == 1)
            index_err = np.argwhere(decision_list_param == -1)
            index_straight = np.argwhere(decision_list_param == 0)
            data_corr = response_time_list_param[index_corr.flatten()]
            data_err = response_time_list_param[index_err.flatten()]
            data_straight = response_time_list_param[index_straight.flatten()]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   # bins=100,
                                                                                   duration=duration,
                                                                                   bins=np.arange(0, 5, 0.05),
                                                                                   center_bin=True)
            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 # bins=100,
                                                                                 duration=duration,
                                                                                 bins=np.arange(0, 5, 0.05),
                                                                                 center_bin=True)
            data_hist_value_straight, data_hist_time_straight = StatisticsService.get_hist(data_straight,
                                                                                 # bins=100,
                                                                                 duration=duration,
                                                                                 bins=np.arange(0, 5, 0.05),
                                                                                 center_bin=True)

            # compute area
            area_plus[parameter] = simpson(data_hist_value_corr, data_hist_time_corr)
            area_minus[parameter] = simpson(data_hist_value_err, data_hist_time_err)
            area_straight[parameter] = simpson(data_hist_value_straight, data_hist_time_straight)
            if area_plus[parameter] <= limit_area * parameter / 100 \
                    or area_minus[parameter] <= limit_area * parameter / 100 \
                    or area_straight[parameter] <= limit_area * parameter / 100:
                return False

            if parameter == 100 and area_plus[parameter] < limit_area_100_plus:
                return False
            if area_plus[parameter] <= limit_area * parameter/100 or area_minus[parameter] <= limit_area * (100 - parameter)/100:
                return False
            if duration is not None and parameter == 0 and (any(data_hist_value_corr > limit_peak_0) or any(data_hist_value_err > limit_peak_0)):
                return False

        return True

    @staticmethod
    def get_output_dicts_from_df(df):
        parameter_list = df[StimulusParameterLabel.COHERENCE.value].unique()
        response_time_list = {p: {} for p in parameter_list}
        decision_list = {p: {} for p in parameter_list}
        time_list = {p: {} for p in parameter_list}
        for p in parameter_list:
            df_p = df[df[StimulusParameterLabel.COHERENCE.value] == p]
            trial_list = df_p["trial"].unique()
            for t in trial_list:
                df_p_t = df_p[df_p["trial"] == t]
                response_time_list[p][t] = np.array(df_p_t[ResponseTimeColumn])
                decision_list[p][t] = np.array(df_p_t[CorrectBoutColumn])
                time_list[p][t] = np.array(df_p_t["start_time"])

        return response_time_list, decision_list, time_list

    @staticmethod
    def compute_quantities_per_parameters(df, column_name=CorrectBoutColumn, analysed_parameter=StimulusParameterLabel.COHERENCE.value):
        parameter_list = np.sort(df[analysed_parameter].unique())

        if column_name == CorrectBoutColumn:
            df = df[df[column_name] != -1]

        correct_bout_list = np.array([df[df[analysed_parameter] == parameter][column_name].mean()
                             for parameter in parameter_list])
        std_correct_bout_list = np.array([df[df[analysed_parameter] == parameter][column_name].std() for parameter in parameter_list])
        return parameter_list, correct_bout_list, std_correct_bout_list

    @staticmethod
    def compute_quantities_per_parameters_multiple_fish(df, column_name=CorrectBoutColumn,
                                          analysed_parameter=StimulusParameterLabel.COHERENCE.value):
        parameter_list = np.sort(df[analysed_parameter].unique())

        if column_name == CorrectBoutColumn:
            df = df[df[column_name] != -1]

        correct_bout_list = np.zeros(len(parameter_list))
        std_correct_bout_list = np.zeros(len(parameter_list))
        for i_p, parameter in enumerate(parameter_list):
            df_filtered = df[df[analysed_parameter] == parameter]
            try:
                window_value_groups = df_filtered.groupby("experiment_ID")[column_name].mean()
            except KeyError:
                try:
                    window_value_groups = df_filtered.groupby("fish_ID")[column_name].mean()
                except KeyError:
                    window_value_groups = df_filtered.groupby("folder_name")[column_name].mean()
            correct_bout_list[i_p] = float(window_value_groups.mean())
            std_correct_bout_list[i_p] = window_value_groups.std()   # / len(window_value_groups) if len(window_value_groups) > 0 else window_value_groups.std()

        return parameter_list, correct_bout_list, std_correct_bout_list

    @staticmethod
    def randomly_sample_df(df, sample_number=1, sample_percentage_size=0.5, sample_per_column=None, with_replacement=False):
        df_list = []
        # sample dataframes keeping same composition in terms of coherence levels
        if sample_per_column is not None:
            parameter_list = df[sample_per_column].unique()

        for i in range(sample_number):
            df_p_list = []
            if sample_per_column is not None:
                for p in parameter_list:
                    df_p = df[df[sample_per_column] == p]
                    n = len(df_p)
                    if with_replacement:
                        index_to_sample = random.choices(range(n), k=int(sample_percentage_size * n))
                    else:
                        index_to_sample = random.sample(range(n), int(sample_percentage_size * n))
                    df_p_i = df_p.iloc[index_to_sample]
                    df_p_list.append(df_p_i)
                df_list.append(pd.concat(df_p_list))
            else:
                n = len(df)
                if with_replacement:
                    index_to_sample = random.choices(range(n), k=int(sample_percentage_size * n))
                else:
                    index_to_sample = random.sample(range(n), int(sample_percentage_size * n))
                df_list.append(df.iloc[index_to_sample])
        return df_list

    @staticmethod
    def remove_border_bout(df, border_threshold):
        return df[np.abs(df["radius"]) < border_threshold]

    @staticmethod
    def transform_arena_measure(measure_in_cm, radius_arena_in_cm=6):
        return measure_in_cm / radius_arena_in_cm

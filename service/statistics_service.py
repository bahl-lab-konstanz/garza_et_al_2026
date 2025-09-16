import itertools
import random

import numpy as np
import pandas as pd
from numba import jit, njit

from analysis.personal_dirs.Roberto.utils.fast_functions import numba_histogram


class StatisticsService:
    number_resampling_bootstrapping = 10000
    threshold_p_value_significant = 0.01

    # return the element of an array lst which is the closest to the value K
    @staticmethod
    def closest(lst, K, sign=None):
        if sign == 'greater_than':
            closest_value = lst[min(range(len(lst)), key=lambda i: lst[i] - K)]
        elif sign == 'less_than':
            closest_value = lst[min(range(len(lst)), key=lambda i: K - lst[i])]
        else:
            closest_value = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]
        closest_index = lst.index(closest_value)
        return closest_value, closest_index

    @classmethod
    def get_window_list_from_data(cls, data, window_number: int = 10, center=None):
        edge_max = data.max()
        edge_min = data.min()
        window_size = (edge_max - edge_min) / window_number
        window_start = np.linspace(data.min(), data.max() - window_size, window_number)
        window_end = np.linspace(data.min() + window_size, data.max(), window_number)
        window_center = np.array([np.mean([window_start[i], window_end[i]]) for i in range(window_number)])
        if center is not None:
            closest_window_center, _ = cls.closest(list(window_center), center)
            distance_from_center = closest_window_center - center
            window_start = window_start - distance_from_center
            # avoid centering to change the edges of the windowed interval
            window_start[0] = edge_min
            window_end = window_end - distance_from_center
            # avoid centering to change the edges of the windowed interval
            window_end[-1] = edge_max
            window_center = window_center - distance_from_center
            # adjust the center of the first and last windows, as a consequence to avoid centering to change the edges of the windowed interval
            window_center[0] = np.mean([window_start[0], window_end[0]])
            window_center[-1] = np.mean([window_start[-1], window_end[-1]])
        return window_start, window_center, window_end

    @staticmethod
    def summary(self, data, label: str = 'data'):
        print("______________")
        print(f"{label} | mean: {np.mean(data)}")
        print(f"{label} | median: {np.median(data)}")
        print(f"{label} | std: {np.std(data)}")
        print(f"{label} | max: {data.max()}")
        print(f"{label} | min: {data.min()}")

    @staticmethod
    def and_list(list1, list2):
        return [item1 and item2 for item1, item2 in zip(list1, list2)]

    @staticmethod
    def or_list(list1, list2):
        return [item1 or item2 for item1, item2 in zip(list1, list2)]

    @staticmethod
    def scale_array(distribution_list, scale: int = None):
        if scale is None:
            scale = sum(distribution_list)
        return np.array(distribution_list) / scale

    @staticmethod
    def map_to_interval(value, starting_interval=(0, 1), target_interval=(0, 1)):
        return (value - starting_interval[0]) * (target_interval[1] - target_interval[0]) / (starting_interval[1] - starting_interval[0]) + target_interval[0]

    @staticmethod
    def get_hist(value_list, bins=10, hist_range=None, duration=None, allow_zero=True, center_bin=False, density=False):
        # value_list can be any iterable supported by numba, i.e. not pandas Series
        if len(value_list) == 0:
            return np.zeros(1), np.zeros(1)
        binned_value_list, bin_list = numba_histogram(np.array(value_list), bins, hist_range)
        if duration is not None:
            binned_value_list = binned_value_list / duration
        if density:
            binned_value_list = binned_value_list / np.sum(binned_value_list)
        if not allow_zero:
            binned_value_list[binned_value_list == 0] = np.finfo(float).eps
        if center_bin:
            bin_list = (bin_list[:-1] + bin_list[1:]) / 2
        return binned_value_list, bin_list

    @staticmethod
    @njit
    def windowing(
            x,
            y,
            window_step_size=0.01,
            window_size=None,
            window_operation='mean',
            x_min=None,
            x_max=None
    ):
        if x_min is None:
            x_min = x.min()
        if x_max is None:
            x_max = x.max()
        if window_size is None:
            window_size = window_step_size
        window_start_list = np.arange(x_min, x_max, window_step_size)
        window_center_list = window_start_list + window_size/2

        windowed_array = np.zeros(len(window_start_list))
        for index_window_start in np.arange(len(window_start_list)):
            window_end = window_start_list[index_window_start] + window_size
            index_in_window_list = np.argwhere(np.logical_and(window_start_list[index_window_start] < x, x < window_end))
            if len(index_in_window_list) != 0:
                y_in_window = y[index_in_window_list.flatten()]
                if window_operation == 'mean':
                    window_value = y_in_window.mean()
                elif window_operation == 'sum':
                    window_value = y_in_window.sum()
                elif window_operation == 'std':
                    window_value = y_in_window.std()
                else:
                    NotImplementedError()
                windowed_array[index_window_start] = float(window_value)

        return windowed_array, window_start_list, window_center_list

    @staticmethod
    # adapted from https://stackoverflow.com/questions/41648058/what-is-the-difference-between-import-numpy-and-import-math
    def norm_pdf(x, mean=0, std=1):
        var = np.power(std, 2)
        return np.exp(-np.power(x - mean, 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


    @staticmethod
    def cartesian_product(x, k):
        # return all combinations with repetitions of arrays of size k with entries taken from x
        return [p for p in itertools.product(x, repeat=k)]

    @staticmethod
    def sample_from_distribution(distribution_dict):
        if "label" not in distribution_dict.keys():
            distribution_dict["label"] = "gaussian"
        if distribution_dict["label"] in ["gaussian", "gaus", "normal", "norm"]:
            return np.random.normal(distribution_dict["mean"], distribution_dict["std"], distribution_dict["size"])
        elif distribution_dict["label"] in ["laplace"]:
            return np.random.laplace(distribution_dict["mu"], distribution_dict["b"], distribution_dict["size"])
        else:
            raise NotImplementedError

    @classmethod
    def sample_random(cls, array, sample_number=1, sample_percentage_size=0.5, with_replacement=False, add_noise=None):
        array_list = []

        for i in range(sample_number):
            n = len(array)
            if with_replacement:
                index_to_sample = random.choices(range(n), k=int(sample_percentage_size * n))
            else:
                index_to_sample = random.sample(range(n), int(sample_percentage_size * n))
            array_sampled = array[index_to_sample]
            if add_noise is not None:
                array_sampled += cls.sample_from_distribution(add_noise)
            array_list.append(array_sampled)
        return array_list

    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

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

    @classmethod
    def bootstrap_test_correlation(cls, df_0, df_1, number_bootstraps=10000):
        delta_corr = np.abs(np.array(df_0.corr()) - np.array(df_1.corr()))

        df_combined_original = pd.concat((df_0, df_1))
        df_combined_list_a = cls.randomly_sample_df(
            df=df_combined_original,
            sample_number=number_bootstraps,
            sample_percentage_size=1,
            with_replacement=True)
        df_combined_list_b = cls.randomly_sample_df(
            df=df_combined_original,
            sample_number=number_bootstraps,
            sample_percentage_size=1,
            with_replacement=True)
        corr_tensor_delta = np.zeros((number_bootstraps, len(df_0.columns), len(df_1.columns)))

        for i in range(number_bootstraps):
            df_combined_a = df_combined_list_a[i]
            corr_combined_a = np.array(df_combined_a.corr())
            df_combined_b = df_combined_list_b[i]
            corr_combined_b = np.array(df_combined_b.corr())
            corr_tensor_delta[i] = np.abs(corr_combined_a - corr_combined_b)

        # check statistical difference of biology from baseline of the model
        p_value = np.mean(corr_tensor_delta >= delta_corr, axis=0)

        return p_value

    @staticmethod
    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x) ** 2 + (ny - 1) * np.std(y) ** 2) / dof)

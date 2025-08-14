import copy
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult, differential_evolution, dual_annealing
from sklearn.model_selection import ParameterGrid

from analysis.personal_dirs.Roberto.model.utils.params import ParameterList, Parameter
from analysis.personal_dirs.Roberto.model.utils.signal import InputSignal
import multiprocessing as mp
from skopt import gp_minimize


def minimize_configured(args):
    f, x, method, bounds, callback, options = args
    res = minimize(f, x, method=method, bounds=bounds, callback=callback, options=options)
    return res


class AbstractModel(ABC):
    save_properties = []
    data = None
    data_train = None
    input_signal = InputSignal()
    loss = None  # LossRobustBIC
    _parameters_fittable = ParameterList()
    parameters_fittable = ParameterList()
    compute_input_signal_at_each_step = False
    history_fitting = []

    # def __init__(self):
    def __init__(self, parameters, tag="model"):
        self.tag = tag
        if parameters is None:
            self.initialize_parameters()
        else:
            self.parameters = copy.deepcopy(parameters)
        self.populate_parameters(initialize=True)
        self._parameters_fittable = copy.deepcopy(self.parameters_fittable)

    def populate_parameters(self, inverted_buffer=False, initialize=False):
        # TODO: add management of inexistent parameters attributes in both blocks
        if inverted_buffer:
            for label, param in self.parameters_fittable:
                getattr(self.parameters, label).value = param.value
        else:
            if initialize:
                self.parameters_fittable = ParameterList()
            for label, param in self.parameters:
                if param.fittable:
                    setattr(self.parameters_fittable, label, param)

    def set_parameter(self, label, value=None, fittable=None, min=None, max=None):
        try:
            if value is not None:
                getattr(self.parameters, label).value = value
            if fittable is not None:
                getattr(self.parameters, label).fittable = fittable
            if min is not None:
                getattr(self.parameters, label).min = min
            if max is not None:
                getattr(self.parameters, label).max = max
        except AttributeError:
            self.parameters.add_parameter(label, Parameter(value=value, fittable=fittable, min=min, max=max))

        self.populate_parameters(initialize=True)
        self._parameters_fittable = copy.deepcopy(self.parameters_fittable)


    def mirror_parameter_fittable(self):
        for label, value in self.parameters_fittable:
            getattr(self.parameters, label).value = value

    @abstractmethod
    def initialize_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def simulate(self, input_signal, dt=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self,
            data_train,
            input_signal: InputSignal = None,
            fitting_method: str = None,
            max_number_iterations: int = None,
            number_trials: int = None,
            focus_scope=None,
            score_low_stop=None,
            score_high_stop=None):
        raise NotImplementedError

    @abstractmethod
    def evaluate_output_signal(self, output_signal, focus_scope=None):
        raise NotImplementedError

    @abstractmethod
    def compute_input_signal(self, index=0):
        raise NotImplementedError

    @abstractmethod
    def prepare_for_fitting(self, input_signal, focus_scope, options):
        raise NotImplementedError


class Model(AbstractModel):
    counter_input_signal = 0
    score = None
    failed_fitting = False
    focus_scope = None  # tuple of 2 elements pointing the minimization function to the actual interval to focus on for optimization (all the rest is neglected)
    debug = True
    is_fitting = False
    randomize_df_train_rows = False
    size_loss_memory = None
    fitting_resolution = None
    fit_options = None
    n_fitting = 0  # number of times the model has already been fitted
    fitting_initialization = None
    iteration = 0

    def _update_counter_input_signal(self, loop=True, random=False):
        if random:
            self.counter_input_signal = int(np.random.uniform(low=0, high=self.data_train.shape[0]))
        elif self.counter_input_signal+1 >= self.data_train.shape[0]:
            if loop:
                self.counter_input_signal = 0
            else:
                raise NotImplementedError
        else:
            self.counter_input_signal += 1

    def _assign_value_model_parameters(self, res: OptimizeResult):
        self.score = res.fun
        i = 0
        for label, value in self.parameters_fittable:
            getattr(self._parameters_fittable, label).value = res.x[i]
            i += 1

    def prepare_for_fitting(self, input_signal, focus_scope, options):
        pass

    def store_history_fitting_de(self, x, convergence):
        score = self.compute_loss_function(x)
        if self.debug:
            print(f"DEBUG | temporary solution: {x}")
            print(f"DEBUG | convergence: {convergence}")
            print(f"DEBUG | score: {score}")
        fitting_now = {
            "x": x,
            "convergence": convergence,
            "score": score,
            "n_fitting": self.n_fitting
        }
        self.history_fitting.append(fitting_now)

    def store_history_fitting(self, x):
        score = self.compute_loss_function(x)
        self.iteration += 1
        if self.debug:
            print(f"DEBUG | temporary solution: {x}")
            print(f"DEBUG | score: {score}")
            print(f"DEBUG | iteration | {self.iteration}")
        fitting_now = {
            "x": x,
            "initialization": self.fitting_initialization,
            "score": score,
            "n_fitting": self.n_fitting
        }
        self.history_fitting.append(fitting_now)

    def store_history_fitting_bayesian(self, res):
        score = self.compute_loss_function(res.x)
        self.iteration += 1
        if self.debug:
            print(f"DEBUG | temporary solution: {res.x}")
            print(f"DEBUG | score: {score}")
            print(f"DEBUG | iteration | {self.iteration}")
        fitting_now = {
            "x": res.x,
            "initialization": self.fitting_initialization,
            "score": score,
            "n_fitting": self.n_fitting
        }
        self.history_fitting.append(fitting_now)

    def compute_loss_function(self, x):
        i = 0
        for label, value in self.parameters_fittable:
            getattr(self.parameters_fittable, label).value = x[i]
            i += 1
        if self.compute_input_signal_at_each_step:
            self.compute_input_signal(index=self.counter_input_signal, scaling_factor=self.scaling_factor_input)
            self._update_counter_input_signal(random=self.randomize_df_train_rows)
        self.populate_parameters(inverted_buffer=True)
        output_signal = self.simulate(self.input_signal, dt=None)
        return self.evaluate_output_signal(output_signal, focus_scope=self.focus_scope)

    # @jit(forceobj=True)
    def _get_initial_guess_list(self, number_trials=1):
        initial_guess_list = []
        for trial in range(number_trials):
            initial_guess_list.append([])
            for attribute_label, attribute_value in self.parameters_fittable:
                initial_guess_list[trial].append(np.random.uniform(low=attribute_value.min, high=attribute_value.max))
        return initial_guess_list

    def _store_good_parameters(self, res):
        score = res.fun
        self._assign_value_model_parameters(res)
        if self.debug:
            print(f"DEBUG | {self.tag} | optimal parameters found with score {score}")
            for property, value in vars(self.parameters_fittable).items():
                print(property, ": ", value.value)
            print("---")

    def _get_fittable_bounds(self):
        return [(float(param.min), float(param.max)) for label, param in self.parameters_fittable]

    def grid_average(self, repetitions=10):
        parameter_dict = {label: np.linspace(value.min, value.max, self.fitting_resolution) for label, value in self.parameters if value.fittable}
        parameter_grid = ParameterGrid(parameter_dict)
        res = {"fun": np.inf, "x": None}
        for param in list(parameter_grid):
            x = [p for p in param.values()]
            for label, value in param.items():
                setattr(getattr(self.parameters, label), "value", value)
            score_list = np.zeros(repetitions)
            for i_rep in range(repetitions):
                score_list[i_rep] = self.compute_loss_function(x)
            score_temp = np.mean(score_list)
            try:
                if score_temp < res["fun"]:
                    res["fun"] = score_temp
                    res["x"] = x
                    res["success"] = True
            except ValueError:
                pass
        return OptimizeResult(res)


    def fit(self,
            data_train=None,
            input_signal: InputSignal = None,
            method="BFGS",
            max_number_iterations: int = 1000,
            focus_scope=None,
            randomize_df_train_rows=False,
            size_loss_memory=None,
            index_data_train=None,
            options=None):
        self.is_fitting = True
        self.prepare_for_fitting(input_signal, focus_scope, options)
        self.fitting_initialization = 0

        if data_train is not None:
            self.data = data_train
        if index_data_train is not None:
            self.data_train = self.data[self.data.index.isin(index_data_train)] \
                if isinstance(self.data, pd.DataFrame) else self.data[index_data_train, :]
        else:
            self.data_train = self.data

        self.focus_scope = focus_scope
        self.randomize_df_train_rows = randomize_df_train_rows
        self.size_loss_memory = size_loss_memory
        self.history_fitting = []
        self.fit_options = options

        if input_signal is not None:
            self.input_signal = input_signal

        if self.score is None:
            self.score = np.inf

        if method == "differential_evolution":
            workers = options["workers"] if (isinstance(options, dict) and "workers" in options and options["workers"] is not None) else 1
            tol = options["tol"] if (isinstance(options, dict) and "tol" in options and options["tol"] is not None) else 0.01
            init = options["init"] if (isinstance(options, dict) and "init" in options and options["init"] is not None) else 'latinhypercube'
            polish = options["polish"] if (isinstance(options, dict) and "polish" in options and options["polish"] is not None) else True
            res = differential_evolution(self.compute_loss_function,
                                         self._get_fittable_bounds(),
                                         callback=self.store_history_fitting_de,
                                         maxiter=max_number_iterations,
                                         tol=tol,
                                         disp=self.debug,
                                         init=init,
                                         workers=workers,
                                         polish=polish)
            self._store_good_parameters(res)
        elif method == "dual_annealing":
            res = dual_annealing(self.compute_loss_function, self._get_fittable_bounds(), maxiter=max_number_iterations, disp=self.debug)
            self._store_good_parameters(res)
        elif method == "grid_search":
            res = self.grid_average(repetitions=max_number_iterations)
            self._store_good_parameters(res)
        elif method == "bayesian":
            workers = options["workers"] if (isinstance(options, dict) and "workers" in options) else 1
            init = options["init"] if (isinstance(options, dict) and "init" in options and options["init"] is not None) else 10
            res = gp_minimize(self.compute_loss_function,  # the function to minimize
                              self._get_fittable_bounds(),  # the bounds on each dimension of x
                              n_calls=max_number_iterations,  # the number of evaluations of f
                              n_random_starts=init,  # the number of random initialization points
                              callback=self.store_history_fitting_bayesian,
                              acq_optimizer="lbfgs",
                              n_jobs=workers)
            self._store_good_parameters(res)
        else:
            workers = options["workers"] if (isinstance(options, dict) and "workers" in options) else 1
            if workers == -1:
                workers = None
            number_trials = options["number_trials"] if (isinstance(options, dict) and "number_trials" in options and options["number_trials"] is not None) else 1
            initial_guess_list = options["init"] if (isinstance(options, dict) and "init" in options and options["init"] is not None) \
                                                 else self._get_initial_guess_list(number_trials)
            score_low_stop = options["score_low_stop "] \
                if (isinstance(options, dict) and "score_low_stop" in options and options["score_low_stop"] is not None) \
                else None
            score_high_stop = options["score_high_stop "] \
                if (isinstance(options, dict) and "score_high_stop" in options and options["score_high_stop"] is not None) \
                else None

            # # multiprocessing
            # pool = mp.get_context("fork").Pool(processes=workers)
            # args = [(self.compute_loss_function, ig, method, self._get_fittable_bounds(), self.store_history_fitting, {"maxiter": max_number_iterations}) \
            #         for ig in initial_guess_list]
            # res_list = pool.map(minimize_configured, args)
            # pool.close()

            # fun_list = np.array([res.fun for res in res_list])
            # index_min_fun = np.argmin(fun_list)
            # self._store_good_parameters(res_list[index_min_fun])

            # print(f"DEBUG | initial guess list | {initial_guess_list}")  # #####
            # print(f"DEBUG | res_list | {res_list}")  # #####
            # print(f"DEBUG | fun_list | {fun_list}")  # #####

            # single-thread
            for index, initial_guess in enumerate(initial_guess_list):
                if self.debug:
                    print(f"DEBUG | {self.tag} | initial guess: {initial_guess}")
                if (score_low_stop is not None and self.score < score_low_stop) or \
                        (score_high_stop is not None and self.score > score_high_stop):
                    break
                try:
                    if self.size_loss_memory is not None:
                        self.loss_memory = []
                    res = minimize(self.compute_loss_function, initial_guess,
                                   method=method,
                                   bounds=self._get_fittable_bounds(),
                                   callback=self.store_history_fitting,
                                   options={"maxiter": max_number_iterations})
                    percentage_complete = index/len(initial_guess_list)*100
                    print(f"DEBUG | nit | {res.nit}")
                    print(f"DEBUG | jacobian | {res.jac}")
                    print(f"DEBUG | hessian | {res.hess}")
                    if percentage_complete % 1 == 0:
                        print(f"INFO | {self.tag} | fit | computing...{int(percentage_complete)}%")
                    if res.fun < self.score:
                        self._store_good_parameters(res)
                except FloatingPointError:
                    pass
        self.parameters_fittable = copy.deepcopy(self._parameters_fittable)
        self.populate_parameters(inverted_buffer=True)

        self.exit_fitting_state()

    def exit_fitting_state(self):
        self.n_fitting += 1
        self.fitting_initialization = None
        self.is_fitting = False

    def get_random_initialization(self, size=1):
        return np.transpose(
            np.array([np.random.uniform(low=p.min, high=p.max, size=size) for _, p in self.parameters_fittable])
        )

    def compute_input_signal(self, index=0):
        raise NotImplementedError

    def initialize_parameters(self):
        raise NotImplementedError

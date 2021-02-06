#!/usr/bin/env python3

# Import modules
import numpy as np


def get_sweep_parameters(parameters, env_config, index):
    """
    Gets the parameters for the hyperparameter sweep defined by the index.
    Each hyperparameter setting has a specific index number, and this function
    will get the appropriate parameters for the argument index. In addition,
    this the indices will wrap around, so if there are a total of 10 different
    hyperparameter settings, then the indices 0 and 10 will return the same
    hyperparameter settings. This is useful for performing loops.

    For example, if you had 10 hyperparameter settings and you wanted to do
    10 runs, the you could just call this for indices in range(0, 10*10). If
    you only wanted to do runs for hyperparameter setting i, then you would
    use indices in range(i, 10, 10*10)

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file
    env_config : dict
        The environment configuration dictionary, as found in the environment's
        json configuration file
    index : int
        The index of the hyperparameters configuration to return

    Returns
    -------
    dict, int
        The dictionary of hyperparameters to use for the agent and the total
        number of combinations of hyperparameters (highest possible unique
        index)
    """
    out_params = {}
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out_params[key] = parameters[key][(index // accum) % num]
        accum *= num
    return (out_params, accum)


def get_sweep_num(parameters):
    """
    Similar to get_sweep_parameters but only returns the total number of
    hyperparameter combinations. This number is the total number of distinct
    hyperparameter settings. If this function returns k, then there are k
    distinct hyperparameter settings, and indices 0 and k refer to the same
    distinct hyperparameter setting.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file

    Returns
    -------
    int
        The number of distinct hyperparameter settings
    """
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        accum *= num
    return accum


def get_hyperparam_indices(data, hp_name, hp_value):
    """
    Gets all hyperparameter indices that have the hyperparameter hp_name
    having the value hp_value.

    Parameters
    ----------
    data : dict
        The data dictionary generated from running main.py
    hp_name : str
        The name of the hyperparameter to check the value of
    hp_value : object
        The value that the hyperparameter should have in each hyperparameter
        settings index

    Returns
    -------
    list of int
        The hyperparameter settings that have the argument hyperparameter
        hp_name having the argument value hp_value
    """
    agent_param = data["experiment"]["agent"]["parameters"]
    env_config = data["experiment"]["environment"]

    hp_indices = []
    for i in range(get_sweep_num(agent_param)):
        # Get the hyperparameters for each hyperparameter setting
        hp_setting = get_sweep_parameters(agent_param, env_config, i)[0]

        if hp_setting[hp_name] == hp_value:
            hp_indices.append(i)

    return hp_indices


def get_varying_single_hyperparam(data, hp_name):
    """
    Gets the hyperparameter indices where only a single hyperparameter is
    varying and all other hyperparameters remain constant.

    Parameters
    ----------
    data : dict
        The data dictionary generated from running main.py
    hp_name : str
        The name of the hyperparameter to vary

    Returns
    -------
    n-tuple of m-tuple of int
        Gets and returns the hyperparameter indices where only a single
        hyperparameter is varying and all others remain constant. The
        total number of values that the varying hyperparameter can take on
        is m; n is the total number of hyperparameter combinations // m.

        For example, if the hyperparameter is the decay rate and it can take
        on values in [0.0, 0.1, 0.5] and there are a total of 81 hyperparameter
        settings combinations, then m = 3 and n = 81 // 3 = 27
    """
    agent_param = data["experiment"]["agent"]["parameters"]
    hps = []  # set(range(exp.get_sweep_num(agent_param)))
    for hp_value in agent_param[hp_name]:
        hps.append(get_hyperparam_indices(data, hp_name, hp_value))

    return tuple(zip(*hps))


def get_best_hp(data, type_, after=0):
    """
    Gets and returns a list of the hyperparameter settings, sorted by average
    return.

    Parameters
    ----------
    data : dict
        They Python data dictionary generated from running main.py
    type_ : str
        The type of return by which to compare hyperparameter settings, one of
        "train" or "eval"
    after : int, optional
        Hyperparameters will only be compared by their performance after
        training for this many episodes (in continuing tasks, this is the
        number of times the task is restarted). For example, if after = -10,
        then only the last 10 returns from training/evaluation are taken
        into account when comparing the hyperparameters. As usual, positive
        values index from the front, and negative values index from the back.

    Returns
    -------
        n-tuple of 2-tuple(int, float)
    A tuple with the number of elements equal to the total number of
    hyperparameter combinations. Each sub-tuple is a tuple of (hyperparameter
    setting number, mean return over all runs and episodes)
    """
    if type_ not in ("train", "eval"):
        raise ValueError("type_ should be one of 'train', 'eval'")

    return_type = "train_episode_rewards" if type_ == "train" \
        else "eval_episode_rewards"

    mean_returns = []
    hp_settings = sorted(list(data["experiment_data"].keys()))
    for hp_setting in hp_settings:
        hp_returns = []
        for run in data["experiment_data"][hp_setting]["runs"]:
            hp_returns.append(run[return_type])
        hp_returns = np.stack(hp_returns)

        # If evaluating, use the mean return over all episodes for each
        # evaluation interval. That is, if 10 eval episodes for each evaluation
        # the take the average return over all these eval episodes
        if type_ == "eval":
            hp_returns = hp_returns.mean(axis=-1)

        # Calculate the average return over all runs
        hp_returns = hp_returns[after:, :].mean(axis=0)

        # Calculate the average return over all "episodes"
        hp_returns = hp_returns.mean(axis=0)

        # Save mean return
        mean_returns.append(hp_returns)

    # Return the best hyperparam settings in order with the
    # mean returns sorted by hyperparmater setting performance
    best_hp_settings = np.argsort(mean_returns)
    mean_returns = np.array(mean_returns)[best_hp_settings]

    return tuple(zip(best_hp_settings, mean_returns))


def combine_runs(data1, data2):
    """
    Adds the runs for each hyperparameter setting in data2 to the runs for the
    corresponding hyperparameter setting in data1.

    Given two data dictionaries, this function will get each hyperparameter
    setting and extend the runs done on this hyperparameter setting and saved
    in data1 by the runs of this hyperparameter setting and saved in data2.
    In short, this function extends the lists
    data1["experiment_data"][i]["runs"] by the lists
    data2["experiment_data"][i]["runs"] for all i. This is useful if
    multiple runs are done at different times, and the two data files need
    to be combined.

    Parameters
    ----------
    data1 : dict
        A data dictionary as generated by main.py
    data2 : dict
        A data dictionary as generated by main.py

    Raises
    ------
    KeyError
        If a hyperparameter setting exists in data2 but not in data1. This
        signals that the hyperparameter settings indices are most likely
        different, so the hyperparameter index i in data1 does not correspond
        to the same hyperparameter index in data2. In addition, all other
        functions expect the number of runs to be consistent for each
        hyperparameter setting, which would be violated in this case.
    """
    for hp_setting in data1["experiment_data"]:
        if hp_setting not in data2.keys():
            # Ensure consistent hyperparam settings indices
            raise KeyError("hyperparameter settings are different " +
                           "between the two experiments")

        extra_runs = data2["experiment_data"][hp_setting]["runs"]
        data1["experiment_data"][hp_setting]["runs"].extend(extra_runs)


def get_returns(data, type_, ind):
    """
    Gets the returns seen by an agent

    Gets the online or offline returns seen by an agent trained with
    hyperparameter settings index ind.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Whether to get the training or evaluation returns, one of 'train',
        'eval'
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index

    Returns
    -------
    array_like
        The array of returns of the form (N, R, C) where N is the number of
        runs, R is the number of times a performance was measured, and C is the
        number of returns generated each time performance was measured
        (offline >= 1; online = 1). For the online setting, N is the number of
        runs, and R is the number of episodes and C = 1. For the offline
        setting, N is the number of runs, R is the number of times offline
        evaluation was performed, and C is the number of episodes run each
        time performance was evaluated offline.
    """
    returns = []
    if type_ == "eval":
        # Get the offline evaluation episode returns per run
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["eval_episode_rewards"])
        returns = np.stack(returns)

    elif type_ == "train":
        # Get the returns per episode per run
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["train_episode_rewards"])
        returns = np.expand_dims(np.stack(returns), axis=2)

    return returns


def get_hyperparams(data, ind):
    """
    Gets the hyperparameters for hyperparameter settings index ind

    data : dict
        The Python data dictionary generated from running main.py
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index

    Returns
    -------
    dict
        The dictionary of hyperparameters
    """
    return data["experiment_data"][ind]["agent_params"]


def get_mean_returns_with_stderr_hp_varying(data, type_, hp_name, combo,
                                            after=0):
    """
    Calculate mean and standard error of return for each hyperparameter value.

    Gets the mean returns for each variation of a single hyperparameter,
    with all other hyperparameters remaining constant. Since there are
    many different ways this can happen (the hyperparameter can vary
    with all other remaining constant, but there are many combinations
    of these constant hyperparameters), the combo argument cycles through
    the combinations of constant hyperparameters.

    Given hyperparameters a, b, and c, let's say we want to get all
    hyperparameter settings indices where a varies, and b and c are constant.
    if a, b, and c can each be 1 or 2, then there are four ways that a can
    vary with b and c remaining constant:

        [
            ((a=1, b=1, c=1), (a=2, b=1, c=1)),         combo = 0
            ((a=1, b=2, c=1), (a=2, b=2, c=1)),         combo = 1
            ((a=1, b=1, c=2), (a=2, b=1, c=2)),         combo = 2
            ((a=1, b=2, c=2), (a=2, b=2, c=2))          combo = 3
        ]

    The combo argument indexes into this list of hyperparameter settings

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    combo : int
        Determines the values of the constant hyperparameters. Given that
        only one hyperparameter may vary, there are many different sets
        having this hyperparameter varying with all others remaining constant
        since each constant hyperparameter may take on many values. This
        argument cycles through all sets of hyperparameter settings indices
        that have only one hyperparameter varying and all others constant.
    """
    hp_combo = get_varying_single_hyperparam(data, hp_name)[combo]

    mean_returns = []
    stderr_returns = []
    for hp in hp_combo:
        mean_return = get_returns(data, type_, hp)

        # If evaluating, use the mean return over all episodes for each
        # evaluation interval. That is, if 10 eval episodes for each evaluation
        # the take the average return over all these eval episodes.
        # If online returns, this axis has a length of 1 so we reduce it
        mean_return = mean_return.mean(axis=-1)

        # Calculate the average return over all "episodes"
        # print(mean_return[:, after:].shape)
        mean_return = mean_return[:, after:].mean(axis=1)

        # Calculate the average return over all runs
        stderr_return = np.std(mean_return, axis=0) / \
            np.sqrt(mean_return.shape[0])
        mean_return = mean_return.mean(axis=0)

        mean_returns.append(mean_return)
        stderr_returns.append(stderr_return)

    # Get each hp value and sort all results by hp value
    hp_values = np.array(data["experiment"]["agent"]["parameters"][hp_name])
    indices = np.argsort(hp_values)

    mean_returns = np.array(mean_returns)[indices]
    stderr_returns = np.array(stderr_returns)[indices]
    hp_values = hp_values[indices]

    return hp_values, mean_returns, stderr_returns


def get_mean_stderr(data, type_, ind, smooth_over):
    """
    Gets the timesteps, mean, and standard error to be plotted for
    a given hyperparameter settings index

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : int
        The hyperparameter settings index to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.

    Returns
    -------
    3-tuple of list(int), list(float), list(float)
        The timesteps, mean episodic returns, and standard errors of the
        episodic returns
    """
    timesteps = None  # So the linter doesn't have a temper tantrum

    # Determine the timesteps to plot at
    if type_ == "eval":
        timesteps = \
            data["experiment_data"][ind]["runs"][0]["timesteps_at_eval"]

    elif type_ == "train":
        timesteps_per_ep = \
            data["experiment_data"][ind]["runs"][0]["train_episode_steps"]
        ep = data["experiment_data"][ind]["runs"][0]["total_train_episodes"]
        timesteps = [timesteps_per_ep[i] * i for i in range(ep)]

    # Get the mean over all episodes per evaluation step (for online
    # returns, this axis will have length 1 so we squeeze it)
    returns = get_returns(data, type_, ind)
    mean = returns.mean(axis=-1)

    # Get the standard deviation of mean episodes per evaluation
    # step over all runs
    runs = returns.shape[0]
    std = np.std(mean, axis=0) / np.sqrt(runs)

    # Get the mean over all runs
    mean = mean.mean(axis=0)

    # Smooth of last k returns if applicable
    if smooth_over != 0:
        mean = np.convolve(mean, np.ones(smooth_over) /
                           smooth_over)[:mean.shape[0]]

    return timesteps, mean, std

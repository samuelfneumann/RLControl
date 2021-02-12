#!/usr/bin/env python3

# Import modules
import matplotlib.pyplot as plt
import utils.experiment_utils as exp
import numpy as np


def plot_mean_with_runs(data, type_, ind, smooth_over, names, colours,
                        figsize=(12, 6), xlim=None, ylim=None,
                        alpha=0.1):
    """
    Plots both the mean return per episode (over runs) as well as the return
    for each individual run (including "mini-runs", if the number of evaluation
    episodes per timestep > 1 and if plotting evaluation data)

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    colours : list of list of str
        The colours to use for each hyperparameter settings plot for each data
        dictionary
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha to use for plots of the runs, by default 0.1
    """
    # Set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Return")
    ax.set_title(f"Mean Return with Runs")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot for each data dictionary given
    for i in range(len(data)):
        for j in range(len(ind[i])):
            timesteps, mean, _ = exp.get_mean_stderr(data[i], type_, ind[i][j],
                                                        smooth_over[i])

            # Plot the mean
            label = f"{names[i]} - Hyperparameter Setting {ind[i][j]}"
            ax.plot(timesteps, mean, label=label, color=colours[i][j])

            # Plot each run
            for run in data[i]["experiment_data"][ind[i][j]]["runs"]:
                run_type = type_ + "_episode_rewards"
                run_data = run[run_type]

                # Expand the dimensions of the training data so that it
                # can be considered in the same way as eval data
                if type_ == "train":
                    run_data = np.expand_dims(run_data, axis=1)

                # Plot each episodes (1 for train, >= 1 for eval) at each
                # timestep evaluated/trained in the run
                for k in range(run_data.shape[1]):

                    # Smooth over timesteps in the episode (1 for train,
                    # >= 1 for eval)
                    if smooth_over[i] > 1:
                        kernel = np.ones((smooth_over[i])) / smooth_over
                        eval_ep = np.convolve(run_data[:, k], kernel, mode="valid")
                    else:
                        eval_ep = run_data[:, k]

                    ax.plot(timesteps, eval_ep, color=colours[i][j],
                            linestyle="--", alpha=alpha)

    ax.legend()

    fig.show()

    return fig, ax


def plot_mean_with_stderr(data, type_, ind, smooth_over, names, fig=None,
                          ax=None, figsize=(12, 6), xlim=None, ylim=None,
                          alpha=0.1, colours=None):
    """
    Plots the average training or evaluation return over all runs for two
    different agents for a number of specific hyperparameter settings.

    Given a list of data dictionaries of the form returned by main.py, this
    function will plot each episodic return for the list of hyperparameter
    settings ind each data dictionary. The ind argument is a list, where each
    element is a list of hyperparameter settings to plot for the data
    dictionary at the same index as this list. For example, if ind[i] = [1, 2],
    then plots will be generated for the data dictionary at location i
    in the data argument for hyperparameter settings ind[i] = [1, 2].
    The smooth_over argument tells how many previous data points to smooth
    over.

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of list of str
        The colours to use for each hyperparameter settings plot for each data
        dictionary

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot
    """
    # Set the colours to be default if not specified
    if colours is None:
        colours = []
        for i in range(len(ind)):
            colours.append([None for _ in ind[i]])

    # Track the total timesteps per hyperparam setting over all episodes and
    # the cumulative timesteps per episode per data dictionary (timesteps
    # should be consistent between all hp settings in a single data dict)
    total_timesteps = []
    cumulative_timesteps = []

    for i in range(len(data)):
        if type_ == "train":
            cumulative_timesteps.append(exp.get_cumulative_timesteps(data[i]
                                        ["experiment_data"][ind[i][0]]["runs"]
                                        [0]["train_episode_steps"]))
        elif type_ == "eval":
            cumulative_timesteps.append(data[i]["experiment_data"][ind[i][0]]
                                        ["runs"][0]["timesteps_at_eval"])
        else:
            raise ValueError("type_ must be one of 'train', 'eval'")
        total_timesteps.append(cumulative_timesteps[-1][-1])

    # Find the minimum of total trained-for timesteps. Each plot will only
    # be plotted on the x-axis until this value
    min_timesteps = min(total_timesteps)

    # For each data dictionary, find the minimum index where the timestep at
    # that index is >=  minimum timestep
    last_ind = []
    for cumulative_timesteps_per_data in cumulative_timesteps:
        final_ind = np.where(cumulative_timesteps_per_data >=
                            min_timesteps)[0][0]
        # Since indexing will stop right before the minimum, increment it
        last_ind.append(final_ind + 1)

    # Plot all data for all HP settings, only up until the minimum index
    fig, ax = None, None
    for i in range(len(data)):
        fig, ax = plot_mean_with_stderr_(data=data[i], type_=type_, ind=ind[i],
                                         smooth_over=smooth_over[i],
                                         name=names[i], fig=fig, ax=ax,
                                         figsize=figsize, xlim=xlim, ylim=ylim,
                                         last_ind=last_ind[i], alpha=alpha,
                                         colours=colours[i])

    return fig, ax


def plot_mean_with_stderr_(data, type_, ind, smooth_over, fig=None, ax=None,
                           figsize=(12, 6), name="", last_ind=-1,
                           xlabel="Timesteps", ylabel="Average Return",
                           timestep_multiply=1, xlim=None, ylim=None,
                           alpha=0.1, colours=None):
    """
    Plots the average training or evaluation return over all runs for a
    list of specific hyperparameter settings for a single agent.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    last_ind : int, optional
        The index of the last element to plot in the returns list,
        by default -1. This is useful if you want to plot many things on the
        same axis, but all of which have a different number of elements. This
        way, we can plot the first last_ind elements of each returns for each
        agent.
    timestep_multiply : int, optional
        A value to multiply each timstep by, by default 1. This is useful if
        your agent does multiple updates per timestep and you want to plot
        performance vs. number of updates.
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")

    if ax is not None and fig is None:
        raise ValueError("must pass figure when passing axis")

    # Set up figure
    if ax is None and fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot with the standard error
    for i in range(len(ind)):
        timesteps, mean, std = exp.get_mean_stderr(data, type_, ind[i],
                                                   smooth_over)
        timesteps = np.array(timesteps[:last_ind]) * timestep_multiply
        mean = mean[:last_ind]
        std = std[:last_ind]
        if colours is not None:
            ax.plot(timesteps, mean, color=colours[i],
                    label=f"{name} hyperparameter settings {ind[i]}")
            ax.fill_between(timesteps, mean-std, mean+std, alpha=alpha,
                            color=colours[i])
        else:
            ax.plot(timesteps, mean,
                    label=f"{name} hyperparameter settings {ind[i]}")
            ax.fill_between(timesteps, mean-std, mean+std, alpha=alpha)

    ax.legend()
    ax.set_title(f"Average {type_.title()} Return per Run with Standard Error")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    fig.show()

    return fig, ax


def hyperparam_sensitivity_plot(data, type_, hp_name, combo, figsize=(12, 6)):
    """
    Plots the hyperparameter sensitivity for hp_name, where the combo argument
    determines which hyperparameters are held constant.

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
    figsize : tuple(int, int)
        The size of the figure
    """
    hp_values, mean_returns, stderr_returns = \
        exp.get_mean_returns_with_stderr_hp_varying(data, type_, hp_name,
                                                    combo, after=0)

    # Set up the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_title(f"{hp_name.title()} Hyperparameter Sensitivity Plot " +
                 f"over {type_.title()} Data")
    ax.set_xlabel("Hyperparameter Values")
    ax.set_ylabel("Average return")

    ax.plot(hp_values, mean_returns)
    ax.fill_between(hp_values, mean_returns-stderr_returns,
                    mean_returns+stderr_returns, alpha=0.15)


def hyperparam_sensitivity_plot_by_settings_index(data, type_, hp_name, ind,
                                                  figsize=(12, 6)):
    """
    Plot hyperparameter sensitivity by hyperparameter settings index.

    Plots the hyperparameter sensitivity plot where all
    constant hyperparameters are equal to those defined by the
    argument hyperparameter settings index. The only hyperparameter that
    is allowed to vary is the one corresponding to hp_name.

    Parameters
    ----------
    data : dict
        The data dictionary generated by running main.py
    type_ : str
        Whether to plot the training or evaluation data, one of
        'train', 'eval'
    hp_name : str
        The name of the hyperparameter to plot the sensitivity of
    ind : int
        The hyperparameter settings index that should define that
        values of all constant hyperparameters. Only the hyperparameter
        defined by hp_name is allowed to vary, all others must be equal to
        the values of those defined by the settings index ind.
    figsize : tuple(int, int)
        The size of the figure
    """
    # Get all the combinations of hyperparameters with only hp_name varying
    hp_name_varying_only_settings = exp.get_varying_single_hyperparam(data,
                                                                      hp_name)

    # Get the index (into the list of tuples with only hp_name varying) of
    # the tuple which has hp_name varying only, but all other constant
    # hyperparameters equal to those defined by hyperparam setting number ind.
    # So hp_combo_constant_hps_equal_ind is the hyperparam combination with
    # all constant hyperparams equal to those in settings number ind.
    hp_combo_constant_hps_equal_ind = \
        next(filter(lambda x: ind in x, hp_name_varying_only_settings))
    combo_index = hp_name_varying_only_settings.index(
        hp_combo_constant_hps_equal_ind)

    # Plot the hyperparameter sensitivity plot using all constant
    # hyperparameters equal to those defined in the settings
    # number ind
    hyperparam_sensitivity_plot(data, type_, hp_name, combo_index,
                                figsize=figsize)


# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# mpl.use('Agg')
# from matplotlib import pyplot as plt
# import numpy as np

# # bit messy but contains all plot functions (for AE_CCEM, DDPG, NAF)
# def plotFunction(agent_name, func_list, state, greedy_action, expl_action, x_min, x_max, resolution=1e3, display_title='', save_title='',
#                  save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):
#     fig, ax = plt.subplots(2, sharex=True)
#     # fig, ax = plt.subplots(figsize=(10, 5))

#     x = np.linspace(x_min, x_max, resolution)
#     y1 = []
#     y2 = []

#     max_point_x = x_min
#     max_point_y = np.float('-inf')

#     if agent_name == 'SoftActorCritic':
#         func1, func2 = func_list[0], func_list[1]


#         for point_x in x:
#             point_y1 = np.squeeze(func1(point_x))  # reduce dimension
#             point_y2 = np.squeeze(func2(point_x))

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')


#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='grey')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     elif agent_name == 'ActorExpert_PICNN':  # Identical to ActorExpert except when passing func1(point_x)
#         func1, func2 = func_list[0], func_list[1]

#         modal_mean = [mean for mean in greedy_action[2]]

#         old_greedy_action = greedy_action[1]
#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

#             ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     elif agent_name == 'ActorCritic':
#         func1, func2 = func_list[0], func_list[1]

#         modal_mean = [mean for mean in greedy_action[1]]

#         old_greedy_action = greedy_action[0]
#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1(point_x))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

#             ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     elif agent_name == 'ActorCritic_unimodal':
#         func1, func2 = func_list[0], func_list[1]

#         modal_mean = greedy_action[1]

#         old_greedy_action = greedy_action[0]
#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1(point_x))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean, linewidth=1.5, color='grey')

#             ax[1].axvline(x=old_greedy_action, linewidth=1.5, color='pink')
#             ax[1].axvline(x=greedy_action, linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     elif agent_name == 'ActorExpert_PICNN':  # Identical to ActorExpert except when passing func1(point_x)
#         func1, func2 = func_list[0], func_list[1]

#         modal_mean = [mean for mean in greedy_action[2]]

#         old_greedy_action = greedy_action[1]
#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

#             ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     elif agent_name == 'ActorExpert':
#         func1, func2 = func_list[0], func_list[1]

#         modal_mean = [mean for mean in greedy_action[2]]

#         old_greedy_action = greedy_action[1]
#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1(point_x))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

#             ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(greedy_action)):
#                 mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + mode_string)
#         else:
#             top_margin = 1.0

#     # elif agent_name == 'ActorExpert_Plus':
#     #     func1, func2 = func_list[0], func_list[1]
#     #
#     #     old_greedy_action = greedy_action[1]
#     #     greedy_action = greedy_action[0]
#     #
#     #     for point_x in x:
#     #         point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
#     #         point_y2 = func2(point_x)
#     #
#     #         if point_y1 > max_point_y:
#     #             max_point_x = point_x
#     #             max_point_y = point_y1
#     #
#     #         y1.append(point_y1)
#     #         y2.append(point_y2)
#     #
#     #     ax[0].plot(x, y1, linewidth=linewidth)
#     #     ax[1].plot(x, y2, linewidth=linewidth)
#     #
#     #     if grid:
#     #         ax[0].grid(True)
#     #         ax[1].grid(True)
#     #         ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
#     #         ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
#     #         ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#     #         ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')
#     #
#     #     if display_title:
#     #         display_title += ", argmax Q(S,A): {:.2f}".format(max_point_x)
#     #         fig.suptitle(display_title, fontsize=11, fontweight='bold')
#     #         top_margin = 0.95
#     #
#     #         mode_string = ""
#     #         for i in range(len(greedy_action)):
#     #             mode_string += "{:.2f}".format(np.squeeze(greedy_action[i])) + ", "
#     #         ax[1].set_title("greedy actions: " + mode_string)
#     #     else:
#     #         top_margin = 1.0

#     elif agent_name == 'DDPG' or agent_name == 'WireFitting' or agent_name == 'SoftQlearning':
#         func1 = func_list[0]
#         for point_x in x:
#             point_y1 = np.squeeze(func1([point_x]))  # reduce dimension

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#         ax[0].plot(x, y1, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:

#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95
#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

#         else:
#             top_margin = 1.0

#     elif agent_name == 'NAF':

#         func1, func2 = func_list[0], func_list[1]

#         for point_x in x:
#             point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
#             # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')

#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:

#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

#         else:
#             top_margin = 1.0




#         # OLD
#         # func1 = func_list[0]
#         # for point_x in x:
#         #     point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
#         #
#         #     if point_y1 > max_point_y:
#         #         max_point_x = point_x
#         #         max_point_y = point_y1
#         #
#         #     y1.append(point_y1)
#         #
#         # ax[0].plot(x, y1, linewidth=linewidth)
#         #
#         # if grid:
#         #     ax[0].grid(True)
#         #     # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
#         #     # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')
#         #
#         #     ax[1].grid(True)
#         #     ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
#         #     ax[1].axvline(x=mean[0], linewidth=1.5, color='red')
#         #
#         # if display_title:
#         #
#         #     display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
#         #         max_point_y) + "\n state: " + str(state)
#         #     fig.suptitle(display_title, fontsize=11, fontweight='bold')
#         #     top_margin = 0.95
#         #
#         #     ax[1].set_title("mean: " + str(mean[0]))
#         #
#         # else:
#         #     top_margin = 1.0

#     elif agent_name == 'PICNN':
#         func1 = func_list[0]
#         for point_x in x:
#             point_y1 = np.squeeze(func1([point_x]))  # reduce dimension

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)

#         ax[0].plot(x, y1, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
#             # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')

#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
#             ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:

#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

#         else:
#             top_margin = 1.0

#     elif agent_name == 'QT_OPT':

#         # utils.plot_utils.plotFunction("QT_OPT", [func1, func2], state, greedy_action, chosen_action, self.action_min,
#         #                               self.action_max,
#         #                               display_title='ep: ' + str(
#         #                                   self.train_ep_count) + ', steps: ' + str(
#         #                                   self.train_global_steps),
#         #                               save_title='steps_' + str(self.train_global_steps),
#         #                               save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
#         #                               show=False)

#         func1, func2 = func_list[0], func_list[1]

#         greedy_action, modal_mean = greedy_action

#         greedy_action = greedy_action[0]

#         for point_x in x:
#             point_y1 = np.squeeze(func1(point_x))  # reduce dimension
#             point_y2 = func2(point_x)

#             if point_y1 > max_point_y:
#                 max_point_x = point_x
#                 max_point_y = point_y1

#             y1.append(point_y1)
#             y2.append(point_y2)

#         ax[0].plot(x, y1, linewidth=linewidth)
#         ax[1].plot(x, y2, linewidth=linewidth)

#         if grid:
#             ax[0].grid(True)
#             ax[1].grid(True)
#             ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')

#             for mean in modal_mean:
#                 ax[1].axvline(x=mean, linewidth=1.5, color='grey')

#             ax[1].axvline(x=greedy_action, linewidth=1.5, color='red')
#             ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

#         if display_title:
#             fig.suptitle(display_title, fontsize=11, fontweight='bold')
#             top_margin = 0.95

#             mode_string = ""
#             for i in range(len(modal_mean)):
#                 mode_string += "{:.2f}".format(np.squeeze(modal_mean[i])) + ", "

#             ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
#             ax[1].set_title("Policy, modal means: " + mode_string)
#         else:
#             top_margin = 1.0

#     # set y range in the Q val plot
#     # ax[0].set_ylim([-0.1, 1.6])

#     # common
#     if equal_aspect:
#         ax.set_aspect('auto')

#     if show:
#         plt.show()

#     else:
#         # print(save_title)
#         save_dir = save_dir + '/figures/'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         plt.savefig(save_dir + save_title)
#         plt.close()

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep, process_time, perf_counter
import mloop.controllers as mlc
import mloop.interfaces as mli


def FourierFromParams(params):
    """Calculate the y-values of a fourier series with given a, b parameters, normalised to the maximum y-value.

    Args:
        params (np.ndarray): array of (2 * n_ab) elements determining a-coefficients

    Returns:
        np.ndarray: array of 200 y-values from FS
    """
    n_ab = len(params) // 2
    a, b = params[:n_ab], params[n_ab:]
    # can alter these if you want to adjust the limits for some reason
    L, xmin, xmax, nsteps = np.pi, -1 * np.pi, np.pi, 200
    xs = np.linspace(xmin, xmax, nsteps)
    ys = np.full(nsteps, 0.5 * a[0])  # the 0th term in the series
    for i, (a_n, b_n) in enumerate(zip(a, b)):  # sum over n_ab terms
        if i != 0:
            for j, x in enumerate(xs):  # calculate correction to y value at each x
                cos_term = a_n * np.cos(i * np.pi * x / L)
                sin_term = b_n * np.sin(i * np.pi * x / L)
                ys[j] += cos_term + sin_term
    if np.amax(np.abs(ys)) != 0.0:
        ys = ys / np.amax(np.abs(ys))  # normalise y-values if they're all non-zero
    return ys


def _MultiNoise(square_diff: float, noise_scale: float) -> float:
    """Combines square_diff (true cost) with gaussian noise in a multiplicative fashion.
    Equivalent to noise proportional to the cost, with std dev noise_scale*square_diff.

    Args:
        square_diff (float): true cost
        noise_scale (float): std dev of noise to be combined with square_diff

    Raises:
        ValueError: Raised if noise_scale < 0 since std dev must be larger than 0

    Returns:
        float: noisy cost value
    """
    if noise_scale < 0:
        raise ValueError("noise_scale must be > 0")
    noise = np.random.normal(loc=1.0, scale=noise_scale)
    cost_noisy = square_diff * noise
    return cost_noisy


def _AddNoise(square_diff: float, noise_scale: float) -> float:
    """Combines square_diff (true cost) with gaussian noise in an additive fashion.
    Equivalent to constant background noise.

    Args:
        square_diff (float): true cost
        noise_scale (float): std dev of noise to be added to square_diff

    Raises:
        ValueError: Raised if noise_scale < 0 since std dev must be larger than 0

    Returns:
        float: noisy cost value
    """
    if noise_scale < 0:
        raise ValueError("noise_scale must be > 0")
    noise = np.random.normal(loc=0.0, scale=noise_scale)
    cost_noisy = square_diff + noise
    return cost_noisy


def Cost(guess_params, y_target, noise_type="None", noise_scale=0.0):
    """Calculates the cost of an FS guess against a target, including noise

    Args:
        guess_params (np.ndarray): a,b-parameters for the guess
        y_target (np.ndarray): y-values of the target
        noise_type (str, optional): Type of noise ("None", "add", or "multi"). Defaults to "None".
        noise_scale (float, optional): Std Dev of noise if noise_type != "None". Defaults to 0.0.

    Raises:
        ValueError: Raised if noise_type is not one of the valid choices

    Returns:
        float: cost including noise
        float: uncertainty on cost
    """
    # true value of cost calculation
    y_guess = FourierFromParams(guess_params)
    square_diff = sum((y_target - y_guess) ** 2) / (
        len(y_target) * (len(guess_params) / 2)
    )

    # adding noise, either additive or multiplicative
    if noise_type != "None":
        if noise_type == "add":
            cost = _AddNoise(square_diff, noise_scale)
            uncert = noise_scale
        elif noise_type == "multi":
            cost = _MultiNoise(square_diff, noise_scale)
            uncert = square_diff * noise_scale
        else:
            raise ValueError(
                f'variable noise_type must be one of "None", "add", "multi". Noise Type was given as "{noise_type}"'
            )
    else:
        cost = square_diff
        uncert = 0
    return cost, uncert


class _FourierInterface(mli.Interface):
    """Interface for M-LOOP.

    Args:
        n_ab (int): number of parameters in a,b-coefficient arrays
        y_target (np.ndarray): y-values of target function
        noise_type (str, optional): Type of noise ("None", "add", or "multi"). Defaults to "None".
        noise_scale (float, optional): Std Dev of noise if noise_type != "None". Defaults to 0.0.
    """

    def __init__(
        self,
        n_ab,
        y_target,
        noise_type=None,
        noise_scale=0.0,
    ):
        super(_FourierInterface, self).__init__()
        self.n_ab = n_ab
        self.y_target = y_target
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def get_next_cost_dict(self, params_dict):
        guess = params_dict["params"]
        cost, uncert = Cost(guess, self.y_target, self.noise_type, self.noise_scale)

        # TODO: Create checks for bad runs, e.g. if cost < 0

        cost_dict = {
            "cost": cost,
            "uncer": uncert,
            "bad": False,
        }
        return cost_dict


def RunOnce(
    max_allowed_runs,
    tcost,
    n_ab,
    y_target,
    noise_type="None",
    noise_scale=0.0,
    learner="gaussian_process",
):
    """Runs M-LOOP once and returns the cost of each run, and the number of runs

    Args:
        max_allowed_runs (int): Max runs for the learner
        tcost (float): Target cost for the learner
        n_ab (int): number of parameters / 2 (length of a,b for FS)
        y_target (np.ndarray): y-values of target function
        noise_type (str, optional): type of noise to combine with costs. Defaults to "None".
        noise_scale (float, optional): std dev of noise to combine with costs. Defaults to 0.0.
        learner (str, optional): Learner type for M-LOOP. Defaults to "gaussian_process".

    Returns:
        np.ndarray: costs of each run
        int: number of runs taken
        list: parameters of each run (list of np.ndarrays)
        float: process_time of controller.optimize()
    """
    if learner not in [
        "gaussian_process",
        "neural_net",
        "differential_evolution",
        "nelder_mead",
        "random",
    ]:
        raise ValueError("learner was not one of the valid choices.")
    interface = _FourierInterface(n_ab, y_target, noise_type, noise_scale)
    controller = mlc.create_controller(
        interface,
        controller_type=learner,
        max_num_runs=max_allowed_runs,
        target_cost=tcost,
        num_params=n_ab * 2,
        min_boundary=np.full(n_ab * 2, -1),  # range [-1, 1]
        max_boundary=np.full(n_ab * 2, 1),
        no_delay=False,
        controller_archive_file_type="pkl",
        learner_archive_file_type="pkl",
    )
    t_init = perf_counter()
    controller.optimize()
    t_fin = perf_counter()
    time_taken = t_fin - t_init
    costs = controller.in_costs
    runs = controller.num_in_costs
    out_params = controller.out_params
    return costs, runs, out_params, time_taken


def TrueCostsFromOuts(out_params, y_target):
    """Calculates the true (noise-less) cost from a set of parameters and a target

    Args:
        out_params (list): Parameters to calculate cost. list of np.ndarrays.
        y_target (np.ndarray): y-values of target function

    Returns:
        np.ndarray: True costs of parameters w.r.t target
    """
    true_costs = np.zeros(len(out_params))
    for i, params in enumerate(out_params):
        true_costs[i], _ = Cost(params, y_target, noise_type="None", noise_scale=0)
    return true_costs


def _MinCosts(costs):
    """Returns an array of the minimum cost found by each run.

    Args:
        costs (np.ndarray): costs measured by the experiment

    Returns:
        np.ndarray: minimum cost array
    """
    min_costs = np.full_like(costs, costs[0])
    for i, c in enumerate(costs):
        if c < min_costs[i]:
            min_costs[i:] = c
    return min_costs


def MinCostAndParams(costs, params):
    """Returns an array of the minimum cost found by each run.

    Args:
        costs (np.ndarray): costs measured by the experiment

    Returns:
        np.ndarray: minimum cost array
    """
    min_costs = np.full_like(costs, costs[0])
    min_params = np.empty_like(params)
    for j in range(min_params):
        min_params[j, :] = params[0, :]
    for i, c in enumerate(costs):
        if c < min_costs[i]:
            min_costs[i:] = c
            for k in range(min_params[i:]):
                min_params[(i + k) :, :] = params[i, :]
    return min_costs, min_params


def _CostLengthEqualiser(costs_list, runs_list, repeats):
    """Equalises length of costs lists by appending the last value repeatedly

    Args:
        costs_list (list): list of np.ndarrays of the costs from each experiment
        runs_list (np.ndarray): length of each costs list
        repeats (int): length of runs_list

    Returns:
        np.ndarray: equalised-length array of costs
        int: length of all costs-lists
    """
    max_length = int(np.amax(runs_list))
    costs_arr = np.empty((repeats, max_length))
    for i, cost in enumerate(costs_list):
        costs_arr[i, : runs_list[i]] = cost
        costs_arr[i, runs_list[i] :] = cost[-1]
    return costs_arr, max_length


def _TimeToString(datetime):
    """Converts a datetime object into a string
    adapted from M-LOOP: mloop.utilities

    Args:
        datetime (datetime): datetime object (e.g. datetime.datetime.now())

    Returns:
        str: date time as 'yyyy-mm-dd_hh-mm'
    """
    return datetime.strftime("%Y-%m-%d_%H-%M")


def _TimingFormatter(times_list):
    """Prints mean and std deviation of the time taken for optimisation

    Args:
        times_list (list): list of process_times for runs
    """
    mean = np.mean(times_list)
    stdev = np.std(times_list)
    print(f"Mean Runtime: {mean:.3g} s")
    print(f"Standard Deviation: {stdev:.3g} s")


def _SaveNPZ(filename, *objs):
    """Save objs as a .npz file with filename in directory ./npz/

    Args:
        filename (str): name of file to be saved as
    """
    if not os.path.isdir("./npz"):
        os.mkdir("./npz")
    np.savez(f"./npz/{filename}.npz", *objs)


def RepeatRuns(
    repeats,
    n_ab,
    savename=None,
    max_allowed_runs=100,
    tcost=0.0,
    y_targets=None,
    noise_type="None",
    noise_scale=0.0,
    learner="gaussian_process",
    sleep_time=0.0,
    save=True,
):
    """Runs M-LOOP for the given parameters repeatedly. Saves to an npz and returns a tuple.

    Args:
        repeats (int): Number of times M-LOOP will be run
        n_ab (int): number of parameters / 2 (length of a,b for FS)
        savename (str, optional): filename to save the data to. Defaults to None.
        max_allowed_runs (int, optional): Max number of runs for each repeat. Defaults to 100.
        tcost (float, optional): Target cost for each run. Defaults to 0.0.
        y_targets (list of np.ndarrays, optional): if specified, gives a target function. If not specified, random FS is used each time. Defaults to None.
        noise_type (str, optional): Type of noise ("None", "add", or "multi"). Defaults to "None" (str).
        noise_scale (float, optional): Std Dev of noise if noise_type != "None". Defaults to 0.0.
        learner (str, optional): Learner type for M-LOOP. Defaults to "gaussian_process".
        sleep_time (float, optional): time in seconds between runs in case each run needs a separate time-label
        save (bool, optional): Whether to save the data to a .npz file. Defaults to True.

    Returns:
        tuple: Results of optimisation.
            start_times: list of start times in case required to access logs
            max_runs: length of longest repeat
            costs_arr: cost of each run in each repeat, padded to be equal lengths if different
            params_list: list of the params for each run in each repeat. List of np.ndarrays with dimensions (runs,params)
            min_costs_arr: noise-less cost of minimum run at each run for each repeat, padded to be equal lengths if different
            min_costs_mean: mean of min_costs_arr (over each repeat)
            min_costs_stderr: std err (stddev/sqrt(repeats)) of min_costs_arr (over each repeat)
    """
    costs_list = []
    min_costs_list = []
    times_list = []
    start_times = []
    params_list = []
    runs_list = np.zeros(repeats, dtype=int)

    # repeatedly run M-LOOP
    if not isinstance(y_targets, list):
        y_targets = np.zeros((repeats, 200))
        for rep in range(repeats):
            guess_params = (np.random.random(n_ab * 2) * 2) - 1
            y_targets[rep] = FourierFromParams(guess_params)

    for rep in range(repeats):
        y_target = y_targets[rep]
        start_time = datetime.datetime.now()
        start_times.append(_TimeToString(start_time))
        # Run
        costs, runs, out_params, time_taken = RunOnce(
            max_allowed_runs,
            tcost,
            n_ab,
            y_target,
            noise_type,
            noise_scale,
            learner,
        )
        # Combine data
        times_list.append(time_taken)
        costs_list.append(costs)
        true_costs = TrueCostsFromOuts(out_params, y_target)
        params_list.append(out_params)
        # minimum costs are calculated as true aka noise-less costs
        min_costs_list.append(_MinCosts(true_costs))
        runs_list[rep] = runs

        if sleep_time > 0.0:  # ensuring it doesn't use the same timestamp twice
            if rep < repeats - 1:
                sleep(sleep_time)

    # equalise costs lists if required
    if np.ndarray.all(runs_list == runs_list[0]):
        max_runs = int(runs_list[0])
        costs_arr = np.array(costs_list)
        min_costs_arr = np.array(min_costs_list)
    else:
        costs_arr, max_runs = _CostLengthEqualiser(costs_list, runs_list, repeats)
        min_costs_arr, _ = _CostLengthEqualiser(min_costs_list, runs_list, repeats)

    # mean & std error
    min_costs_mean = np.array([np.mean(min_costs_arr[:, i]) for i in range(max_runs)])
    min_costs_stderr = [
        np.std(min_costs_arr[:, i]) / np.sqrt(repeats) for i in range(max_runs)
    ]

    # Print Times
    _TimingFormatter(times_list)

    if save:
        if not savename:
            savename = _TimeToString(datetime.datetime.now())
        _SaveNPZ(
            savename,
            # hyperparameters
            repeats,
            max_allowed_runs,
            tcost,
            n_ab,
            y_targets,
            noise_type,
            noise_scale,
            learner,
            # results
            start_times,
            times_list,
            max_runs,
            costs_arr,
            params_list,
            min_costs_arr,
            min_costs_mean,
            min_costs_stderr,
        )
    return (
        start_times,
        times_list,
        max_runs,
        costs_arr,
        params_list,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr,
    )


def ReadRepeatNPZ(filename=None, fullpath=None):
    """Reads an NPZ file saved by RepeatRuns

    Args:
        filename (str, optional): filename supplied to RepeatRuns. Defaults to None.
        fullpath (str, optional): Full file path in case filename fails for some reason. Defaults to None.

    Returns: (various) - see RepeatRuns for explanation of outputs
        ----hyperparameters----
        repeats,
        max_allowed_runs,
        tcost,
        n_ab,
        y_targets,
        noise_type,
        noise_scale,
        learner_type,
        ----outputs----
        start_times,
        max_runs,
        costs_arr,
        params_list,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr
    """
    if not fullpath:
        npz = np.load(f"./npz/{filename}.npz")
    else:
        npz = np.load(fullpath)
    repeats = int(npz["arr_0"])
    max_allowed_runs = int(npz["arr_1"])
    tcost = float(npz["arr_2"])
    n_ab = int(npz["arr_3"])
    y_targets = npz["arr_4"]
    noise_type = str(npz["arr_5"])
    noise_scale = float(npz["arr_6"])
    learner_type = str(npz["arr_7"])
    start_times = list(npz["arr_8"])
    times_list = npz["arr_9"]
    max_runs = int(npz["arr_10"])
    costs_arr = npz["arr_11"]
    params_list = list(npz["arr_12"])
    min_costs_arr = npz["arr_13"]
    min_costs_mean = npz["arr_14"]
    min_costs_stderr = npz["arr_15"]
    return (
        repeats,
        max_allowed_runs,
        tcost,
        n_ab,
        y_targets,
        noise_type,
        noise_scale,
        learner_type,
        start_times,
        times_list,
        max_runs,
        costs_arr,
        params_list,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr,
    )


def ErrorbarRepeatPlot(max_runs, min_costs_mean, min_costs_stderr, savename=None):
    """Plots the mean minimum cost & std error as a scatter plot with error bars

    Args:
        max_runs (int): Length of min_costs_mean array
        min_costs_mean (np.ndarray): Array of min mean costs
        min_costs_stderr (np.ndarray): Array of min costs std errors
        savename (str, optional): Name for file to be saved to, leave blank if no need for saving. Defaults to None.
    """
    runs = np.arange(max_runs)
    plt.errorbar(
        runs,
        min_costs_mean,
        yerr=min_costs_stderr,
        fmt=".",
        c="black",
        ms=5,
        capsize=2,
        capthick=1,
    )
    plt.yscale("log")
    if savename:
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        plt.savefig(f"images/{savename}.png", dpi=600)
    plt.show()


def _SciFormat(preformat, number, postformat="", sf=4):
    """Prints numbers in scientific notation (for graphing using LaTeX in matplotlib)
    note: 4 sig fig
    i.e. 12000 -> 1.2 x 10^4

    Exceptions for zero (returns "{preformat} 0 {postformat}")
    and for powers of 10 (returns "{preformat} 10^(x) {postformat}")

    Args:
        preformat (str): Any text to go before the number. If none, use "".
        number (float): Number to be converted into scientific notation.
        postformat (str, optional): Any text to go after the number. Needs a space to start. Defaults to "".
        sf (int, optional): Sig figs to go on the number. Defaults to 4.
    Returns:
        str: String of scientific notation.
    """
    try:
        exp = int(np.floor(np.log10(number)))
        value = number / (10 ** exp)
        if preformat != "":
            if value != 1.0:
                sci = f"{preformat} {value:.{sf}g} $\\times 10^{{{exp}}}${postformat}"
            else:
                sci = f"{preformat} $10^{{{exp}}}${postformat}"
        else:
            if value != 1.0:
                sci = f"{value:.{sf}g} $\\times 10^{{{exp}}}${postformat}"
            else:
                sci = f"$10^{{{exp}}}${postformat}"
    except OverflowError:  # aka zero
        if preformat != "":
            sci = f"{preformat} {number}{postformat}"
        else:
            sci = f"{number}{postformat}"
    return sci


def SquareDiffPlot(
    guess_params, y_target, ax, fontsize=12, cost_xy=(0, 0), ffamily="serif"
):
    """Plots the square difference between guess and y_target on a given axes

    Args:
        guess_params (np.ndarray): parameters for guess
        y_target (np.ndarray): values of target function (200 length)
        ax (matplotlib.axes.Axes): axes to plot the figure on
        fontsize (int, optional): font size for annotations. Defaults to 12.
        cost_xy (tuple, optional): location for cost annotation. Defaults to (0, 0.9).
        ffamily (str, optional): font family for annotation. Defaults to "serif".
    """
    y_guess = FourierFromParams(guess_params)
    square_diff = (y_target - y_guess) ** 2
    cost = Cost(guess_params, y_target)[0]
    xs = np.linspace(-np.pi, np.pi, 200)
    diff_max = np.amax(square_diff)

    # Plotting
    ax.plot(xs, square_diff)
    ax.annotate(
        _SciFormat("Cost =", cost),
        xy=cost_xy,
        xycoords="axes fraction",
        fontfamily=ffamily,
        fontsize=fontsize,
    )
    ax.axhline(0, c="k")

    # Design
    ax.set_xlim([-np.pi, np.pi])
    ax.set_xticks([])
    ax.set_ylim([-0.15 * diff_max, 1.05 * diff_max])
    ax.set_yticks([diff_max])
    if diff_max < 0.01:
        ax.set_yticklabels([_SciFormat("", diff_max, sf=2)])
        ax.tick_params(axis="y", direction="in", labelsize=11, pad=-65)
    else:
        ax.set_yticklabels([f"{diff_max:.2g}"])
        ax.tick_params(axis="y", direction="in", labelsize=11, pad=-25)

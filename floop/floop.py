import os
import time
import numpy as np
import mloop.controllers as mlc
import mloop.interfaces as mli


def FourierFromParams(a, b):
    """Calculate the y-values of a fourier series with given a, b parameters.

    Args:
        a (np.ndarray): array of n_ab elements determining a-coefficients
        a (np.ndarray): array of n_ab elements determining b-coefficients

    Returns:
        np.ndarray: array of y-values from FS
    """
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


def Cost(guess, y_target, noise_type="None", noise_scale=0.0):
    """Calculates the cost of an FS guess against a target, including noise

    Args:
        guess (np.ndarray): a,b-parameters for the guess
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
    n_ab = len(guess)
    a_g = guess[:n_ab]
    b_g = guess[n_ab:]
    y_guess = FourierFromParams(a_g, b_g)
    square_diff = sum((y_target - y_guess) ** 2) / (len(y_target) * len(a_g))

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
                'variable noise_type must be one of "None", "add", "multi"'
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
    max_allowed_runs, tcost, n_ab, y_target, noise_type="None", noise_scale=0.0
):
    """Runs M-LOOP once and returns the cost of each run, and the number of runs

    Args:
        max_allowed_runs (int): Max runs for the learner
        tcost (float): Target cost for the learner
        n_ab (int): number of parameters / 2 (length of a,b for FS)
        y_target (np.ndarray): y-values of target function
        noise_type (str, optional): type of noise to combine with costs. Defaults to "None".
        noise_scale (float, optional): std dev of noise to combine with costs. Defaults to 0.0.

    Returns:
        np.ndarray: costs of each run
        int: number of runs taken
    """
    interface = _FourierInterface(n_ab, y_target, noise_scale, noise_type)
    controller = mlc.create_controller(
        interface,
        max_num_runs=max_allowed_runs,
        target_cost=tcost,
        num_params=n_ab * 2,
        min_boundary=np.full(n_ab * 2, -2),  # range [-1, 1]
        max_boundary=np.full(n_ab * 2, 2),
        no_delay=False,
        controller_archive_file_type="pkl",
        learner_archive_file_type="pkl",
    )
    controller.optimize
    costs = controller.in_costs
    runs = controller.num_in_costs
    return costs, runs


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
    max_length = np.amax(runs_list)
    costs_arr = np.empty((repeats, max_length))
    for i, cost in enumerate(costs_list):
        costs_arr[i, : runs_list[i]] = cost
        costs_arr[i, runs_list[i] :] = cost[-1]
    return costs_arr, max_length


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
    savename,
    max_allowed_runs=100,
    tcost=0.0,
    y_targets=None,
    noise_type="None",
    noise_scale=0.0,
    sleep_time=0.0,
):
    """Runs M-LOOP for the given parameters repeatedly. Saves to an npz and returns a tuple.

    Args:
        repeats (int): Number of times M-LOOP will be run
        n_ab (int): number of parameters / 2 (length of a,b for FS)
        savename (str): filename to save the data to
        max_allowed_runs (int, optional): Max number of runs for each repeat. Defaults to 100.
        tcost (float, optional): Target cost for each run. Defaults to 0.0.
        y_targets (list of np.ndarrays, optional): if specified, gives a target function. If not specified, random FS is used each time. Defaults to None.
        noise_type (str, optional): Type of noise ("None", "add", or "multi"). Defaults to "None".
        noise_scale (float, optional): Std Dev of noise if noise_type != "None". Defaults to 0.0.
        sleep_time (float, optional): time in seconds between runs in case each run needs a separate time-label

    Returns:
        tuple: Results of optimisation.
            max_runs: length of longest repeat
            costs_arr: cost of each run in each repeat, padded to be equal lengths if different
            min_costs_arr: cost of minimum run at each run for each repeat, padded to be equal lengths if different
            min_costs_mean: mean of min_costs_arr (over each repeat)
            min_costs_stderr: std err (stddev/sqrt(repeats)) of min_costs_arr (over each repeat)
    """
    costs_list = []
    min_costs_list = []
    runs_list = np.zeros(repeats)
    # repeatedly run M-LOOP
    for rep in range(repeats):
        if not y_targets:
            a_t = (np.random.random(n_ab) * 2) - 1
            b_t = (np.random.random(n_ab) * 2) - 1
            y_target = FourierFromParams(a_t, b_t)
        else:
            y_target = y_targets[rep]
        costs, runs = RunOnce(
            max_allowed_runs, tcost, n_ab, y_target, noise_type, noise_scale
        )
        costs_list.append(costs)
        min_costs_list.append(_MinCosts(costs))
        runs_list[rep] = runs
        if sleep_time > 0.0:
            if rep != repeats - 1:
                time.sleep(sleep_time)  # so it doesn't use the same timestamp twice
    # equalise costs lists if required
    if np.ndarray.all(runs_list == runs_list[0]):
        max_runs = runs_list[0]
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
        # results
        max_runs,
        costs_arr,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr,
    )
    return (
        max_runs,
        costs_arr,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr,
    )


def ReadRepeatNPZ(filename):
    """Reads an NPZ file saved by RepeatRuns

    Args:
        filename (str): filename supplied to RepeatRuns

    Returns: (various)
        repeats,
        max_allowed_runs,
        tcost,
        n_ab,
        y_targets,
        noise_type,
        noise_scale,
        max_runs,
        costs_arr,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr
    """
    npz = np.load(f"./npz/{filename}.npz")
    repeats = int(npz["arr_0"])
    max_allowed_runs = int(npz["arr_1"])
    tcost = float(npz["arr_2"])
    n_ab = int(npz["arr_3"])
    y_targets = npz["arr_4"]
    noise_type = str(npz["arr_5"])
    noise_scale = float(npz["arr_6"])
    max_runs = int(npz["arr_7"])
    costs_arr = npz["arr_8"]
    min_costs_arr = npz["arr_9"]
    min_costs_mean = npz["arr_10"]
    min_costs_stderr = npz["arr_11"]
    return (
        repeats,
        max_allowed_runs,
        tcost,
        n_ab,
        y_targets,
        noise_type,
        noise_scale,
        max_runs,
        costs_arr,
        min_costs_arr,
        min_costs_mean,
        min_costs_stderr,
    )

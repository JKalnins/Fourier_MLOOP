import matplotlib.pyplot as plt
import numpy as np
from floop import floop

repeats = 3
n_ab = 3
savename = "test"  # ignored because save==False
max_allowed_runs = 100
tcost = 0.0
y_targets = None  # if not none, get from FourierFromParams
noise_type = "add"
noise_scale = 0.0
sleep_time = 0.0
save = False

# Run Once
def test_once():
    np.random.seed(777)  # Acer picked this seed
    n_ab = 3
    max_allowed_runs = 100
    tcost = 0.0
    noise_type = "add"
    noise_scale = 0.0
    a_t = np.random.random(n_ab) * 2 - 1
    b_t = np.random.random(n_ab) * 2 - 1

    y_target = floop.FourierFromParams(a_t, b_t)
    costs, num_runs = floop.RunOnce(
        max_allowed_runs, tcost, n_ab, y_target, noise_type, noise_scale
    )
    print(a_t, b_t)
    print("mean cost:\n", np.average(costs))
    print("runs:\n", num_runs)
    runs = np.arange(num_runs)
    mins = floop._MinCosts(costs)
    plt.scatter(runs, costs)
    plt.scatter(runs, mins)
    plt.yscale("log")
    plt.show()
    runs_correct = bool(num_runs == 100)
    if runs_correct:
        print("passed test_once (Still requires manual sense check of graph)")
    else:
        print("failed test_once")


# Run Repeats
# outputs
def test_repeat():
    repeats = 3
    n_ab = 3
    savename = "test"  # ignored because save==False
    max_allowed_runs = 100
    tcost = 0.0
    y_targets = None  # if not none, get from FourierFromParams
    noise_type = "add"
    noise_scale = 0.0
    sleep_time = 0.0
    save = False
    (
        _,  # start_times
        max_runs,
        _,  # costs_arr
        _,  # min_costs_arr
        min_costs_mean,
        min_costs_stderr,
    ) = floop.RepeatRuns(  # repeat function
        repeats,
        n_ab,
        savename,
        max_allowed_runs,
        tcost,
        y_targets,
        noise_type,
        noise_scale,
        sleep_time,
        save,
    )
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
    plt.show()


if __name__ == "__main__":
    # test_once()
    test_repeat()
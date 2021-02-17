# FLOOP
 Streamlining module for using M-LOOP Gaussian Processes optimising an arbitrary target function and saving results.

Floop uses online optimisation with M-LOOP to optimise the parameters of a partial Fourier series with $2n_{ab}$ coefficients with respect to an arbitrary target function combined with Gaussian Noise using a cost function defined as:

$C(\vec{f}, \vec{y}) = \frac{\sum_{i=1}^k (f_i - y_i)^2}{kn_{ab}} + \epsilon(0, \sigma).$

Here $\vec{f}, \vec{y}$ are vectors of the Fourier series and target function at a set of arbitrary points and $\epsilon(\mu, \sigma)$ is the Gaussian noise of mean $\mu$ and standard deviation $\sigma$.
 
 Developed as part of my Masters Project working with Gaussian Processes to optimise laser physics experiments (without being in the lab because of covid-19).
 
 If you want to install this, please ensure you have M-LOOP (https://m-loop.readthedocs.io/en/stable/api/mloop.html) installed in a virtual env/conda env first, since I haven't checked the package requirements in this package and M-LOOP's dependencies can be fiddly to install. Clone this repo to a local directory, enter the directory and activate the env with M-LOOP installed, then install with `pip install .`.
 
 Code style is managed with Black.

## Example
```python
def main():
    repeats = 3 # number of simulations run
    n_ab = 3 # number of parameters / 2
    savename = "example"
    max_allowed_runs = 100
    tcost = 0.0 # not supplying a target cost means M-LOOp will run till max_allowed_runs is reached
    y_targets = None # RepeatRuns will automatically create random y_targets
    noise_type = "add"
    noise_scale = 0.0 # standard deviation of noise
    sleep_time = 0.0
    save = False

    # Run Repeats
    (
        _,  # start_times
        _,  # times_list
        max_runs,
        _,  # costs_arr
        _,  # min_costs_arr
        min_costs_mean,
        min_costs_stderr,
    ) = floop.RepeatRuns(
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
    floop.ErrorbarRepeatPlot(max_runs, min_costs_mean, min_costs_stderr, savename=None)


if __name__ == "__main__":
    main()
```

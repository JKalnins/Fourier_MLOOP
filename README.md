# FLOOP
Module to run optimisations using M-LOOP (Machine-Learning Online Optimisation Package: https://www.nature.com/articles/srep25890?sf73151609=1, https://m-loop.readthedocs.io/en/stable/api/mloop.html).

Floop uses online optimisation with M-LOOP to optimise the parameters of a partial Fourier series with $2n_{ab}$ coefficients with respect to an arbitrary target function combined with Gaussian Noise using a cost function defined as:

$C(\vec{f}, \vec{y}) = \frac{\sum_{i=1}^k (f_i - y_i)^2}{kn_{ab}} + \epsilon(0, \sigma).$

Here $\vec{f}, \vec{y}$ are vectors of the values of the Guess and Target functions at a set of points and $\epsilon(\mu, \sigma)$ is a Gaussian variable with mean $\mu$ and standard deviation $\sigma$ representing noise.

The default range of the Fourier series is $[- \pi, \pi]$ with $k=200$ evenly spaced points across this interval.
 
 Developed as part of my Masters Project working with Gaussian Processes to optimise laser physics experiments (without access to labs because of covid-19, which led to the development of this model experiment).
 
 If you want to install this, please ensure you have M-LOOP installed in a virtual env/conda env first, since I haven't checked the package requirements in this package and M-LOOP's dependencies can be fiddly to install. Clone this repo to a local directory, enter the directory and activate the env with M-LOOP installed, then install with `pip install .`. If you want to edit this, create a fork of the repo & clone it (see various sets of instructions online) then use `pip install . -e` to install a version which will update as you save your copy.
 
 Code style and formatting is managed with Black.

## Example
```python
def main():
    repeats = 3 # number of simulations run
    n_ab = 3 # number of parameters / 2
    savename = None 
    max_allowed_runs = 100
    tcost = -10 # arbitrarily low number << noise_scale
    y_targets = None  # if not none, define your own list of np arrays, 200 points
    noise_type = "add"
    noise_scale = 5e-4
    learner = "gaussian_process" # see M-LOOP documentation for options
    sleep_time = 0.0
    save = False
    # Run Repeats
    (
        _,  # start_times
        _,  # times_list
        max_runs,
        _,  # costs_arr
        _,  # params_list
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
        learner,
        sleep_time,
        save,
    )
    floop.ErrorbarRepeatPlot(max_runs, min_costs_mean, min_costs_stderr, savename=None)


if __name__ == "__main__":
    main()
```

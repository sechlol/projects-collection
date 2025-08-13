import time
import numpy as np
import pandas as pd
import formulas as fo
import matplotlib.pyplot as plt

from typing import Callable
from functools import partial
from constants import Si28, Au198, OUTPUT_DIR


def run_benchmarks():
    """
    Run benchmarks for critical functions in the project.
    """

    # Define functions to be benchmarked
    functions = {
        "screening_function_h": partial(fo.screening_function, x=1e10),
        "screening_function_l": partial(fo.screening_function, x=1e-10),
        "coulomb_potential_h": partial(fo.coulomb_potential, r=1e10, z1=14, z2=79),
        "coulomb_potential_l": partial(fo.coulomb_potential, r=1e-10, z1=14, z2=79),
        "g_squared_hh": partial(fo.g_squared, b=1e10, r=1e10 + 1, e_com=1e6, z1=14, z2=79),
        "g_squared_hl": partial(fo.g_squared, b=1e10, r=1e10 + 1, e_com=10, z1=14, z2=79),
        "g_squared_lh": partial(fo.g_squared, b=1e-10, r=1e-9, e_com=1e6, z1=14, z2=79),
        "g_squared_ll": partial(fo.g_squared, b=1e-10, r=1e-9, e_com=10, z1=14, z2=79),
        "universal_stopping_power": partial(fo.universal_stopping_power, e_lab=1000, projectile=Si28, target=Au198),
        "find_r_min": partial(fo.find_r_min, b=0.1, e_com=1000, z1=14, z2=79),
        "scattering_angle": partial(fo.scattering_angle, b=0.1, r_min=0.1, e_com=1000, z1=14, z2=79),
        "stopping_power": partial(fo.stopping_power, e_lab=1000, a1=Si28, a2=Au198),
    }

    data = {}

    # Run benchmarks
    for name, f in functions.items():
        print("\n* Running benchmark for", name)
        result = _benchmark(f, iterations=500)
        data[name] = result

        print(f"\t- mean: {result.mean():.4f} ms")
        print(f"\t- std: {result.std():.4f} ms")
        print(f"\t- min: {result.min():.4f} ms, max: {result.max():.4f} ms")

    data = pd.DataFrame(data)
    _plot_execution_times(data)

    # Save results to CSV
    filename = f"{OUTPUT_DIR}/benchmarks.csv"
    data.to_csv(filename, index=False)
    print("Benchmark results saved to benchmarks.csv")


def _benchmark(f: Callable, iterations: int) -> np.ndarray:
    """
    Benchmark a function by running it `iterations` times and returning the time it took to output each time.

    Parameters
    ----------
        f : function to benchmark. It should not take any arguments.
        iterations : number of times to output the function in ms

    Returns
    -------
        numpy array containing the execution times in ms.
    """
    data = []
    for i in range(iterations):
        t_start = time.time()
        f()
        t_end = time.time()
        data.append(t_end - t_start)

    return np.array(data) * 1000


def _plot_execution_times(data: pd.DataFrame):
    """
    Create a bar plot comparing the execution times of each function.

    Parameters
    ----------
    data : pandas DataFrame containing the benchmark results in ms.
    """

    plt.figure(figsize=(10, 8))
    bars = plt.bar(data.columns, data.mean())
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.xlabel('Functions')
    plt.yscale('log')
    plt.ylabel('Execution Time [ms] (log)')
    plt.title('Execution Times of Benchmark Functions')

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')

    # save plot
    filename = f"{OUTPUT_DIR}/benchmarks.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\t* Benchmark plot saved as {filename}")

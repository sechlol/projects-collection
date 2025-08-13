from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import formulas as f
from atom import Atom
from constants import OUTPUT_DIR

# lab energy in eV, used as index for stopping power
e_lab_ev = np.logspace(1, 7, 200)
e_lab_kev = e_lab_ev / 1000


def study_of_g(interactions: List[Tuple[Atom, Atom]]):
    """
    Plots g(r) for different values of b and e_lab.
    This will help understand the function and choose the best method for finding the solution to g(r_min) = 0
    Each interaction will be plotted in a separate figure.

    Parameters
    ----------
        interactions : List of tuples of interacting atoms (projectile, target)
    """
    # impact parameter in Å
    b_values = [1, 2, 10, 100]

    # lab energy in eV
    e_lab_values = [10, 50, 100, 1000]

    for i, (a1, a2) in enumerate(interactions):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for b, e_lab, ax in zip(b_values, e_lab_values, axes):
            e_com = f.elab_to_ecom(a1, a2, e_lab)

            # Create a function g(r) with b and e_com as parameters. Only r is left as a variable.
            g = partial(f.g_squared, b=b, e_com=e_com, z1=a1.atomic_number, z2=a2.atomic_number)

            # Possible distances between particles in Å
            x = np.linspace(b / 1000, b * 2, 300)
            y = g(r=x)

            # plot the function
            ax.plot(x, y)
            ax.set_xlabel("distance r [Å]")
            ax.set_ylabel("$g(r)^2$")
            ax.set_ylim(-2.3, np.max(y) + 1)
            ax.axhline(0, color="red", linestyle="--", label=f"y=0")
            ax.set_title(f"b={b} [Å], e_com = {e_com:.2e} [eV]")
            ax.legend()

        filename = f"{OUTPUT_DIR}/g2_plot_{i}.png"
        plt.suptitle(f"Study of $g(r)^2$ for {a1.name_latex} -> {a2.name_latex}")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"\t* Plot for {a1.name} -> {a2.name} saved as {filename}")


def study_of_r_min(interactions: List[Tuple[Atom, Atom]]):
    """
    Studies the relationship between r_min and b for different values of e_lab.
    Each interaction will be plotted in a separate figure.

    Parameters
    ----------
        interactions : List of tuples of interacting atoms (projectile, target)
    """
    # lab energy in eV
    e_lab_values = [10, 500, 1e4, 5e6]

    for i, (a1, a2) in enumerate(interactions):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for e_lab, ax in zip(e_lab_values, axes):
            e_com = f.elab_to_ecom(a1, a2, e_lab)
            all_b = np.logspace(-3, 2, 1000)
            r_min = np.array([f.find_r_min(b=b, e_com=e_com, z1=a1.atomic_number, z2=a2.atomic_number) for b in all_b])

            y1 = all_b / r_min
            y2 = r_min - all_b

            ax.plot(all_b, y1, label="$b / r_{min}$")
            ax.set_xscale("log")
            ax.set_ylabel("$b / r_{min}$")
            ax.set_xlabel("impact parameter b [Å] (log)")
            ax.set_title(f"e_com = {e_com:.2e} [eV]")

            # plot second line on the same chart with but different y-axis
            ax2 = ax.twinx()
            ax2.plot(all_b, y2, color="red", label="$r_{min} - b$")
            ax2.set_ylabel("$r_{min}$ [Å]")

            # add common legend
            lines = [ax.get_lines()[0], ax2.get_lines()[0]]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")

        filename = f"{OUTPUT_DIR}/rmin_plot_{i}.png"
        plt.suptitle("Study of $r_{min}$ " + f"{a1.name_latex} -> {a2.name_latex}")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"\t* Plot for {a1.name} -> {a2.name} saved as {filename}")


def study_of_scattering_angle(interactions: List[Tuple[Atom, Atom]]):
    """
    Studies the angle of scattering for different interactions, for chosen values of b and e_lab.
    Each interaction will be plotted in a separate figure.

    Parameters
    ----------
        interactions : List of tuples of interacting atoms (projectile, target)
    """
    # lab energy in eV
    e_lab_values = [10, 1e3, 1e4, 5e6]

    for i, (a1, a2) in enumerate(interactions):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for e_lab, ax in zip(e_lab_values, axes):
            e_com = f.elab_to_ecom(a1, a2, e_lab)

            all_b = np.logspace(-2, 3, 1000)
            r_min = [f.find_r_min(b=b, e_com=e_com, z1=a1.atomic_number, z2=a2.atomic_number) for b in all_b]
            angles = [f.scattering_angle(b=b, r_min=r, e_com=e_com, z1=a1.atomic_number, z2=a2.atomic_number) for b, r
                      in zip(all_b, r_min)]

            # convert radians to degrees
            y = list(map(np.rad2deg, angles))

            ax.plot(all_b, y)
            ax.set_xscale("log")
            ax.set_ylabel("Scattering angle [degrees]")
            ax.set_xlabel("impact parameter b [Å] (log)")
            ax.set_title(f"e_com = {e_com:.2e} [eV]")

        filename = f"{OUTPUT_DIR}/scattering_angle_plot_{i}.png"
        plt.suptitle("Study of scattering angle " + f"{a1.name_latex} -> {a2.name_latex}")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"\t* Plot for {a1.name} -> {a2.name} saved as {filename}")


def study_of_b_max(interactions: List[Tuple[Atom, Atom]]):
    """
    Studies the behaviour of the integrand in formula (8) for different values of b_max, and for chosen values of e_lab.
    Each interaction will be plotted in a separate figure.

    Parameters
    ----------
        interactions : List of tuples of interacting atoms (projectile, target)
    """
    # lab energy in eV
    e_lab_values = [10, 1e2, 1e4, 1e6]
    threshold = 1e-12

    for i, (a1, a2) in enumerate(interactions):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for e_lab, ax in zip(e_lab_values, axes):
            e_com = f.elab_to_ecom(a1, a2, e_lab)

            x = np.logspace(-1, 2, 500)
            y = np.array([f.stopping_power_integrand(a1, a2, b, e_com) for b in x])

            # find index of first element below threshold
            id_threshold = np.where(y < threshold)[0][0]
            xt = x[id_threshold]

            ax.loglog(x, y)
            ax.set_xlabel("impact parameter b [Å] (log)")
            ax.set_ylabel("stepping power integrand (log)")
            ax.set_title(f"E_lab = {e_lab:.2e} eV")
            ax.axvline(xt, color="red", linestyle="--", label=f"b_max = {xt:.2f}Å, y={y[id_threshold]:.1e}")
            ax.legend()

        filename = f"{OUTPUT_DIR}/b_max_plot_{i}.png"
        plt.suptitle(f"Study of b_max {a1.name} -> {a2.name}")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"\t* Plot for {a1.name} -> {a2.name} saved as {filename}")


def stopping_power(interactions: List[Tuple[Atom, Atom]], name: Optional[str] = ""):
    """
    Computes the stopping power between two atoms for different values of e_lab.
    All interactions will be plotted in the same figure.

    Parameters
    ----------
        interactions : List of tuples of interacting atoms (projectile, target)
        name : Optional name to add to the plot's filename
    """
    h = len(interactions) // 2
    fig, axes = plt.subplots(h, 2, figsize=(10, 5 * h))
    axes = axes.flatten()
    data = {}

    for (a1, a2), ax in zip(interactions, axes):
        power = np.array([f.stopping_power(a1, a2, e) for e in e_lab_ev])

        # Find peak energy
        peak = np.argmax(power)
        x_peak = e_lab_ev[peak]
        y_peak = power[peak]
        print(f"\t* Peak for {a1.name} -> {a2.name} at {x_peak:.2f} eV = {y_peak:.2f} eV/(atoms/Å^2)")

        # plot curve
        ax.loglog(e_lab_kev, power)
        ax.set_title(f"{a1.name} --> {a2.name}")
        ax.set_xlabel("E_lab [keV] (log)")
        ax.set_ylabel(r"Stopping Power [eV/(atoms/$Å^2$)] (log)")

        # record data
        data[f"{a1.name}_{a2.name}"] = power

    filename = f"{OUTPUT_DIR}/computed_sn{name}.png"
    plt.suptitle("Stopping Power for various collisions")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\t* Plot for stopping power saved as {filename}")

    return pd.DataFrame(data, index=e_lab_kev)


def universal_stopping_power(collisions: List[Tuple[Atom, Atom]], name: Optional[str] = "") -> pd.DataFrame:
    """
    Calculates the universal stopping power for a list of collisions and plots the results.

    Parameters
    ----------
        collisions : List of tuples of atoms (projectile, target)
        name : Optional name to add to the plot's filename
    """

    h = len(collisions) // 2
    fig, axes = plt.subplots(h, 2, figsize=(10, 5 * h))
    axes = axes.flatten()
    data = {}

    for (a1, a2), ax in zip(collisions, axes):
        power = np.array([f.universal_stopping_power(a1, a2, e) for e in e_lab_kev])

        # Find peak energy
        peak = np.argmax(power)
        x_peak = e_lab_kev[peak] * 1000
        y_peak = power[peak]
        print(f"\t* Peak for {a1.name} -> {a2.name} at {x_peak:.2f} eV = {y_peak:.2f} eV/(atoms/Å^2)")

        # plot curve
        ax.loglog(e_lab_kev, power)
        ax.set_title(f"{a1.name_latex} --> {a2.name_latex}")
        ax.set_xlabel("E_lab [keV] (log)")
        ax.set_ylabel(r"Stopping Power [eV/(atoms/$Å^2$)] (log)")

        # record data
        data[f"{a1.name}_{a2.name}"] = power

    filename = f"{OUTPUT_DIR}/universal_sn{name}.png"
    plt.suptitle("Universal Stopping Power for various collisions")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\t* Plot saved as {filename}")

    return pd.DataFrame(data, index=e_lab_kev)


def compare_results(universal: pd.DataFrame, computed: pd.DataFrame, name: Optional[str] = ""):
    """
    Calculates the errors between the universal stopping power and the computed stopping power.

    Parameters
    ----------
        universal : Dataframe containing the universal stopping power (used as truth reference)
        computed : Dataframe containing the computed stopping power
        name : Optional name to add to the plot's filename
    """
    assert np.all(universal.shape == computed.shape), "Dataframes must have the same shape"
    assert np.all(universal.index == computed.index), "Dataframes must have the same index"

    h = universal.shape[1] // 2
    fig, axes = plt.subplots(h, 2, figsize=(10, 5 * h))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        col = universal.columns[i]
        abs_error = np.abs(universal[col] - computed[col])
        rel_error = abs_error / universal[col]

        # Plot absolute error
        ax.plot(universal.index, abs_error, label="Abs error")
        ax.set_xlabel("E_lab [eV] (log)")
        ax.set_xscale("log")
        ax.set_ylabel("Absolute error [eV/(atoms/Å^2)]")
        ax.set_title(f"Error for {col}")

        # Plot relative error in the same chart with but different y-axis
        ax2 = ax.twinx()
        ax2.plot(universal.index, rel_error, color="red", label="Rel error")
        ax2.set_ylabel("Relative error")

        # add common legend
        lines = [ax.get_lines()[0], ax2.get_lines()[0]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)

    filename = f"{OUTPUT_DIR}/errors{name}.png"
    plt.suptitle("Errors between universal and computed stopping power")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\t* Plot saved as {filename}")

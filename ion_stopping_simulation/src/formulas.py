from typing import Union, Optional

import numpy as np
import scipy
from scipy.optimize import root_scalar, fsolve
from scipy import integrate
from atom import Atom
from constants import VACUUM_PERMITTIVITY, ELEMENTARY_CHARGE, JOULE_TO_EV, CM2_TO_ANG2

# Used in screening_function().
_alphas = np.array([0.1818, 0.5099, 0.2802, 0.02817])
_betas = np.exp(-np.array([3.2, 0.9423, 0.4028, 0.2016]))

# used in coulomb_potential()
_potential_multiplier = 4 * np.pi * VACUUM_PERMITTIVITY
_charge_multiplier = ELEMENTARY_CHARGE ** 2


def screening_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the screening function (7) for a given value of x.
    Because this function is called a lot, it is optimized for speed.
    It can handle both single values and arrays of values.

    Parameters
    ----------
        x : Value of x to calculate the screening function for

    Returns
    -------
        screening_function : Value of the screening function for the given x
    """
    return \
            _alphas[0] * _betas[0] ** x + \
            _alphas[1] * _betas[1] ** x + \
            _alphas[2] * _betas[2] ** x + \
            _alphas[3] * _betas[3] ** x


def coulomb_potential(r: float, z1: float, z2: float) -> float:
    """
    Calculate the Coulomb potential for two ions, corresponding to formula (5).
    Uses pre-computed constants whenever possible to speed up the calculation.

    Parameters
    ----------
        r : Distance between ions in Å
        z1 : Atomic number of projectile
        z2 : Atomic number of target

    Returns
    -------
        culoumb_potential : Coulomb potential for two ions in eV
    """

    screening_length = 0.46848 / (z1 ** 0.23 + z2 ** 0.23)
    potential = (z1 * z2 * _charge_multiplier) / (_potential_multiplier * r) * screening_function(r / screening_length)

    # Convert potential from Joules to eV
    return potential * JOULE_TO_EV


def g_squared(b: float, r: Union[float, np.ndarray], e_com: float, z1: float, z2: float) -> Union[float, np.ndarray]:
    """
    Calculate the g^2 function for two ions. It is a more convenient form to use of formula (3) for
    finding the zero of the expression g(r_min) = 0, without having to worry about negative values.
    """
    potential = coulomb_potential(r, z1, z2)
    return 1 - ((b / r) ** 2) - (potential / e_com)


def scattering_angle(b: float, r_min: float, e_com: float, z1: int, z2: int) -> float:
    """
    Calculate scattering angle of two interacting particles, using formula (11) after the
    change of variable r -> u. Solves the integral between [0, 1] using quadrature.

    Parameters
    ----------
        b : impact parameter in Å
        r_min : distance of minimum approach in Å
        e_com : center of mass energy in eV
        z1 : atomic number of projectile particle
        z2 : atomic number of target particle

    Returns
    -------
        angle : scattering angle in radians
    """

    # pre-compute values that are constant in the integrand function:
    b_squared = b ** 2
    r_min_squared = r_min ** 2

    def integrand(u: float):
        """
        Integrand is function (12) after the change of variable r -> u
        """
        u_squared = u ** 2
        potential = coulomb_potential(r_min, z1, z2) - coulomb_potential(r_min / (1 - u_squared), z1, z2)
        f_u = b_squared * (2 - u_squared) + r_min_squared / (u_squared * e_com) * potential
        return np.sqrt(1 / f_u)

    value = integrate.quad(integrand, 0, 1)[0]
    angle_rad = np.pi - 4 * b * value
    return angle_rad


def gamma_function(mass_projectile: float, mass_target: float) -> float:
    """
    Calculate mass ratio gamma, corresponding to formula (9) in the assignment description.

    Parameters
    ----------
        mass_projectile : mass of projectile particle in atomic mass units
        mass_target : mass of target particle in atomic mass units

    Returns
    -------
        gamma : mass ratio
    """
    return 4 * mass_projectile * mass_target / (mass_projectile + mass_target) ** 2


def stopping_power_integrand(a1: Atom, a2: Atom, b: float, e_com: float) -> float:
    """
    Interaction potential, corresponding to the argument of the integral in formula (8) in the assignment description.

    Parameters
    ----------
        a1 : projectile particle
        a2 : target particle
        b : impact parameter in Å
        e_com : center of mass energy in eV

    Returns
    -------
        potential : interaction potential in eV
    """
    r_min = find_r_min(b, e_com, a1.atomic_number, a2.atomic_number)
    angle = scattering_angle(b, r_min, e_com, a1.atomic_number, a2.atomic_number)
    return np.sin(angle / 2) ** 2 * b


def stopping_power(a1: Atom, a2: Atom, e_lab: float, b_max: Optional[float] = 10.6) -> float:
    """
    Calculate the stopping power of a projectile particle a1 towards a target particle a2.
    Corresponds to formula (8) in the assignment description.

    Parameters
    ----------
        a1 : projectile particle
        a2 : target particle
        e_lab : kinetic energy of projectile particle in eV
        b_max : upper limit for the impact parameter (in Å) after which the interaction potential
                has a negligible value. Default is 10 Å.

    Returns
    -------
        stopping_power : stopping power in eV/atoms/Å^2
    """
    gamma = gamma_function(a1.atomic_mass, a2.atomic_mass)
    e_com = elab_to_ecom(a1, a2, e_lab)

    # define integrand as a function of b. All the other parameters are fixed
    def integrand(b: float):
        return stopping_power_integrand(a1, a2, b, e_com)

    result = integrate.quad(integrand, 0, b_max)[0]
    return 2 * np.pi * gamma * e_lab * result


def find_r_min(b: float, e_com: float, z1: int, z2: int) -> float:
    """
    Finds the distance of minimum approach of two particles with impact parameter b and center of mass energy e_com.
    If the root finding algorithm fails, a ValueError is raised.

    Parameters
    ----------
        b : impact parameter in Å
        e_com : center of mass energy in eV
        z1 : atomic number of projectile particle
        z2 : atomic number of target particle

    Returns
    -------
        r_min : distance of minimum approach in Å
    """

    # define g(r) as a function of r. All the other parameters are fixed
    def g(r: float) -> float:
        return g_squared(b=b, r=r, e_com=e_com, z1=z1, z2=z2)

    # find zero of g(r) using scipy. Use insight from the study_of_g() function to choose
    # suitable initial parameters for the root finding algorithm
    r_left = b / 10
    r_right = 10 / b if b < 1 else b * 10

    try:
        result = root_scalar(g, bracket=[r_left, r_right], x0=b)
        if result.converged:
            return result.root
    except ValueError:
        print(f"Error! Could not find r_min for parameters b={b}, e_com={e_com}. g({r_left}) = {g(r_left)}, g({r_right}) = {g(r_right)}")
        return fsolve(g, x0=b, xtol=1e-15, maxfev=1000)[0]


def elab_to_ecom(projectile: Atom, target: Atom, energy_lab: float) -> float:
    """
    Calculate the center of mass energy for a projectile and target.

    Parameters
    ----------
        projectile : Projectile atom
        target : Target atom
        energy_lab : Energy of projectile in lab frame in eV

    Returns
    -------
        ecom : Center of mass energy in eV
    """
    total_mass = projectile.atomic_mass + target.atomic_mass
    return energy_lab * (target.atomic_mass / total_mass)


def universal_stopping_power(projectile: Atom, target: Atom, e_lab: float) -> float:
    """
    Calculate the universal nuclear stopping power for a projectile and target.

    Parameters
    ----------
        projectile : Projectile atom
        target : Target atom
        e_lab : Energy of projectile in lab frame

    Returns
    -------
        sn : Universal nuclear stopping power in eV/atoms/Å^2
    """
    total_mass = projectile.atomic_mass + target.atomic_mass
    total_charge = projectile.atomic_number ** 0.23 + target.atomic_number ** 0.23

    # Calculate epsilon
    e1 = 32.53 * target.atomic_mass * e_lab
    e2 = projectile.atomic_number * target.atomic_number * total_mass * total_charge
    e = e1 / e2

    # Calculate s_n(e)
    if e > 30:
        sne = np.log(e) / (2 * e)
    else:
        sne = np.log(1 + 1.138 * e) / (2 * (e + e * 0.01321 * e ** 0.21226 + 0.19593 * e ** 0.5))

    # Calculate stopping power S_n(E_lab)
    sn1 = 8.462 * 10 ** -15 * projectile.atomic_number * target.atomic_number * projectile.atomic_mass
    sn2 = total_mass * total_charge
    sn = sn1 / sn2 * sne

    # Convert units from cm^2 to Å^2
    return sn * CM2_TO_ANG2

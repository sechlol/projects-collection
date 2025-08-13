import studies
from atom import Atom
from constants import H1, Si28, Au198
from benchmarks import run_benchmarks

# List of tuples of interacting atoms. Add more if you're curious.
interactions = [(H1, Si28), (Si28, Au198)]  # (H1, Au198), (Au198, Si28)]


def main():
    print("0. Run benchmarks for critical functions.")
    run_benchmarks()

    # This simulates interactions using the universal stopping power formula.
    print("\n1. Calculating stopping power with universal nuclear stopping power formula.")
    result_universal = studies.universal_stopping_power(interactions)

    # Studies the function g(r) for different values of b and e_lab
    print("\n2. Plotting the function g(r)^2 for different values of b and e_lab.")
    studies.study_of_g(interactions)

    # Studies r_min for different values of b and e_lab
    print("\n3. Plotting value of r_min of interactions for different values of b and e_lab.")
    studies.study_of_r_min(interactions)

    # Studies the scattering angle for different values of b and e_lab
    print("\n4. Plotting the scattering angle of interactions for different values of b and e_lab.")
    studies.study_of_scattering_angle(interactions)

    # Studies the relation between b and the integrand in formula (8) for different values of e_lab
    print("\n5. Studying b_max for different values of e_lab.")
    studies.study_of_b_max(interactions)

    # Studies the stopping power for different values of e_lab and b
    print("\n6. Studying the stopping power for different values of e_lab and b. (this can take a minute...)")
    result_computed = studies.stopping_power(interactions)

    print("\n7. Comparing the results from the universal stopping power formula and the computed stopping power.")
    studies.compare_results(result_universal, result_computed)

    # Bonus: Studies the stopping power for other atoms
    print("\n8. BONUS: Studying the stopping power for other atoms (this can take a minute or two...)")
    bonus()


def bonus():
    he4 = Atom("He", atomic_number=2, neutron_number=2, atomic_mass=4.002602)
    fe60 = Atom("Fe", atomic_number=26, neutron_number=34, atomic_mass=59.934071)
    cu63 = Atom("Cu", atomic_number=29, neutron_number=34, atomic_mass=62.929601)
    pb208 = Atom("Pb", atomic_number=82, neutron_number=126, atomic_mass=207.976652)

    bonus_interactions = [(he4, fe60), (he4, cu63), (he4, pb208), (fe60, cu63), (fe60, pb208), (pb208, he4)]
    print("8.1: Universal Stepping power...")
    result_computed = studies.stopping_power(bonus_interactions, name="_bonus")
    print("8.2: Computed Stepping power...")
    result_universal = studies.universal_stopping_power(bonus_interactions, name="_bonus")
    print("8.3: Comparing results...")
    studies.compare_results(result_universal, result_computed, name="_bonus")


if __name__ == "__main__":
    main()

from atom import Atom

# Conversion factors
EV_TO_JOULE = 1.602e-19  # eV to Joules
JOULE_TO_EV = 1 / EV_TO_JOULE  # Joules to eV
METRE_TO_ANG = 1e10  # 1 metre = 10^10 Angstroms
CM2_TO_ANG2 = 1e16  # 1 cm^2 = 10^16 Å^2

# Constants
VACUUM_PERMITTIVITY = 8.854187817e-12 / METRE_TO_ANG  # C^2 / (N Å^2)
ELEMENTARY_CHARGE = 1.602176634e-19  # C

# Define atoms to be used in the simulation
H1 = Atom("H", atomic_number=1, neutron_number=0, atomic_mass=1.00783)
Si28 = Atom("Si", atomic_number=14, neutron_number=14, atomic_mass=27.97692654)
Au198 = Atom("Au", atomic_number=79, neutron_number=119, atomic_mass=197.968244)

OUTPUT_DIR = "../output"

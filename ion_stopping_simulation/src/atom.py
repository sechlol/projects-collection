from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    """
    Class for storing atomic data for a given atom. Once initialized the data cannot be changed.
    """

    symbol: str
    atomic_number: int
    neutron_number: int
    atomic_mass: float

    @property
    def name(self) -> str:
        """
        Formats the name of the atom as "atomic_num + neutron_num + name"
        """
        return f"{self.atomic_number + self.neutron_number}{self.symbol}"

    @property
    def name_latex(self) -> str:
        """
        Formats the name of the atom for a LaTeX-compatible string. Useful for plotting.
        """
        return f"$^{{{self.atomic_number + self.neutron_number}}}{self.symbol}$"

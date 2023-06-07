from ngsolve import Parameter


# EXACT SOLUTION CLASS
class Exact:
    """
    A class for exact solutions for all problems. Its usage is motivated by the fact that it can store
    its own state (time).
    """
    def __init__(self, name, params):
        """
        Default constructor, which initializes time of the object.
        Args:
            params: Dict[float]
                A dictionary of parameters needed for the coefficient functions to be defined.
                Examples: diffusion coefficient nu, kinematic viscosity mu, radius of the surface R.
        """
        self.t = Parameter(0.0)
        self.name = name
        self.params = params
        self.cfs = None

    def set_cfs(self, cfs):
        """
        Sets coefficient functions associated with the exact solution.
        Args:
            cfs: Dict[CoefficientFunction]
                A dictionary of coefficient functions associated with the exact solution.
                Examples: solution u, right-hand side f, ambient velocity field w.
        Returns:

        """
        self.cfs = cfs

    def set_time(self, t):
        """
        Changes the time of the Exact solution object to tval.
        Args:
            tval: float
                New time of the exact solution object.
        Returns:

        """
        self.t.Set(t)

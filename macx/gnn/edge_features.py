from collections.abc import Sequence
from typing import Optional

import e3nn_jax as e3nn
import jax.numpy as jnp


class Envelope:
    def __init__(self, r_cut):
        self.r_cut = r_cut

    def __call__(self, r):
        raise NotImplementedError


class IdentityEnvelope(Envelope):
    def __call__(self, r):
        return jnp.where(r > self.r_cut, jnp.zeros_like(r), jnp.ones_like(r))


class PolynomialEnvelope(Envelope):
    def __init__(self, r_cut: int, n0: int, n1: int):
        super().__init__(r_cut)
        self.envelope = e3nn.poly_envelope(n0, n1, r_cut)

    def __call__(self, r):
        return self.envelope(r)


class RadialBasis:
    r"""
    Base class for the radial bases.

    Args:
        n_rbf (int): the number of basis functions
        r_cut (float): the cutoff radius for the envelope
        envelope_factory (Callable): optional, creates the distance envelope
        kwargs: (dict): optional, kwargs for the envelope function
    """
    def __init__(
        self,
        n_rbf,
        r_cut,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        self.n_rbf = n_rbf
        if envelope_factory is None:
            envelope_factory = IdentityEnvelope
        self.envelope = envelope_factory(r_cut, **(envelope_kwargs or {}))

    def __call__(self, r):
        raise NotImplementedError

    def apply_envelope(self, r, rbfs):
        envelope = self.envelope(r)[..., None]
        return envelope * rbfs


class BesselBasis(RadialBasis):
    def __call__(self, r):
        return self.apply_envelope(r, e3nn.bessel(r, self.n_rbf, self.envelope.r_cut))


class DistancePowersBasis(RadialBasis):
    def __init__(
        self,
        n_rbf,
        r_cut,
        powers: Sequence[int],
        eps: float = 1.0e-2,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        if n_rbf != len(powers):
            raise ValueError(
                f"Length of powers ({len(powers)}) is not equal to n_rbf ({n_rbf})"
            )
        super().__init__(
            n_rbf,
            r_cut,
            envelope_factory=envelope_factory,
            envelope_kwargs=envelope_kwargs,
        )
        self.powers = jnp.asarray(powers)
        self.eps = eps

    def __call__(self, r):
        powers = jnp.where(
            self.powers > 0,
            r[..., None] ** self.powers,
            1 / (r[..., None] ** (-self.powers) + self.eps),
        )
        return self.apply_envelope(r, powers)


class GaussianBasis(RadialBasis):
    def __init__(
        self,
        n_rbf,
        r_cut,
        offset: bool = False,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            n_rbf,
            r_cut,
            envelope_factory=envelope_factory,
            envelope_kwargs=envelope_kwargs,
        )
        delta = 1 / (2 * n_rbf) if offset else 0
        qs = jnp.linspace(delta, 1 - delta, n_rbf)
        self.mus = r_cut * qs**2
        self.sigmas = (1 + r_cut * qs) / 7

    def __call__(self, r):
        gaussians = jnp.exp(-((r[..., None] - self.mus) ** 2) / self.sigmas**2)
        return self.apply_envelope(r, gaussians)


class CombinedRadialBases(RadialBasis):
    r"""
    Class combining multiple radial bases. The total number of basis functions
    :math:`n_{rbf}` has to match the sum of the numbers of basis functions of 
    the constituents.


    Args:
        n_rbf (int): the total number of basis functions
        r_cut (float): the cutoff radius for the radial bases
        n_rbfs (Sequence[int]): the number of basis function per basis respectively
        factories (Sequence[Callable]): the factories of the bases respectively
        kwargs: (Sequence[dict]): optional, kwargs for the bases respectively
    """

    def __init__(
        self,
        n_rbf: int,
        r_cut: float,
        n_rbfs: Sequence[int],
        factories: Sequence,
        kwargs: Optional[Sequence[dict]] = None,
    ):
        if not len(n_rbfs) == len(factories):
            raise ValueError(
                f"length of n_rbfs ({len(n_rbfs)}) should be equal to "
                f"length of factories ({len(factories)})"
            )
        if not sum(n_rbfs) == n_rbf:
            raise ValueError(
                "n_rbf ({n_rbf}) should be equal to sum of n_rbfs ({sum(n_rbfs)})"
            )
        self.bases = []
        kwargs = kwargs or [{} for _ in factories]
        for n_rbf, factory, kwargs in zip(n_rbfs, factories, kwargs):
            self.bases.append(factory(n_rbf, r_cut, **(kwargs or {})))

    def __call__(self, r):
        basis = []
        for base in self.bases:
            basis.append(base(r))
        return jnp.concatenate(basis, axis=-1)


class EdgeFeatures:
    r"""
    Class combining the radial and angular bases to obtain edge features from
    difference vectors.


    Args:
        n_rbf (int): the number of radial basis functions
        r_cut (float): the cutoff radius for the radial basis
        irreps (Sequence[e3nn_jax.Irrep]): the irreducible representations for
            the angular basis
        equivariant: default True, if False the distinction between feature
            dimension and irreps dimension is removed
        radial_basis_factory (Callable): optional, creates the radial basis
        radial_basis_kwargs (dict): optional, kwargs for the radial basis
    """

    def __init__(
        self,
        n_rbf: int,
        r_cut: float,
        irreps: Sequence[e3nn.Irrep],
        equivariant: bool = True,
        *,
        radial_basis_factory: Optional = None,
        radial_basis_kwargs: Optional[dict] = None,
    ):
        if radial_basis_factory is None:
            radial_basis_factory = BesselBasis
        self.radial_basis = radial_basis_factory(
            n_rbf, r_cut, **(radial_basis_kwargs or {})
        )
        self.angular_basis = lambda r: e3nn.spherical_harmonics(
            irreps, r, normalize=True, normalization="component"
        ).array
        self.equivariant = equivariant

    def __call__(self, r: jnp.ndarray):
        radial = self.radial_basis(jnp.linalg.norm(r, axis=-1))[..., None]
        angular = self.angular_basis(r)[..., None, :]
        features = radial * angular
        if not self.equivariant:
            features = features.reshape(*features.shape[:-2], -1)
        return features

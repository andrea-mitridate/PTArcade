from enterprise.signals import utils
import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
import itertools
import numpy as np


### modifies BasisCommonGP to include frequency dependent orfs
def BasisCommonGP(priorFunction, basisFunction, orfFunction, coefficients=False, combine=True, name=""):
    class BasisCommonGP(signal_base.CommonSignal):
        signal_type = "common basis"
        signal_name = "common"
        signal_id = name

        basis_combine = combine

        _orf = orfFunction(name)
        _prior = priorFunction(name)

        def __init__(self, psr):
            super(BasisCommonGP, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id

            pname = "_".join([psr.name, name])
            self._bases = basisFunction(pname, psr=psr)

            self._params, self._coefficients = {}, {}

            for par in itertools.chain(
                self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
            ):
                self._params[par.name] = par

            self._psrpos = psr.pos
            # add here a line that defines self._psrdist

            if coefficients:
                self._construct_basis()

                # if we're given an instantiated coefficient vector
                # that's what we will use
                if isinstance(coefficients, parameter.Parameter):
                    self._coefficients[""] = coefficients
                    self._params[coefficients.name] = coefficients

                    return

                chain = itertools.chain(
                    self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
                )
                priorargs = {par.name: self._params[par.name] for par in chain}

                logprior = parameter.Function(self._get_coefficient_logprior, **priorargs)

                size = self._basis.shape[1]

                cpar = parameter.GPCoefficients(logprior=logprior, size=size)(pname + "_coefficients")

                self._coefficients[""] = cpar
                self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            return [pp.name for pp in self._bases.params]

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", limit=1)
        def _construct_basis(self, params={}):
            self._basis, self._labels = self._bases(params=params)

        if coefficients:

            def _get_coefficient_logprior(self, c, **params):
                # MV: for correlated GPs, the prior needs to use
                #     the coefficients for all GPs together;
                #     this may require parameter groups

                raise NotImplementedError("Need to implement common prior " + "for BasisCommonGP coefficients")

            @property
            def delay_params(self):
                return [pp.name for pp in self.params if "_coefficients" in pp.name]

            @signal_base.cache_call(["basis_params", "delay_params"])
            def get_delay(self, params={}):
                self._construct_basis(params)

                p = self._coefficients[""]
                c = params[p.name] if p.name in params else p.value
                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phicross(cls, signal1, signal2, params):
                return None

            def get_phiinv(self, params):
                return None

        else:

            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)
                prior = BasisCommonGP._prior(self._labels, params=params)
                # orf = BasisCommonGP._orf(self._labels, self._psrpos, self._psrpos, self._psrdis, self._psrdis, params=params)
                orf = BasisCommonGP._orf(self._labels, self._psrpos, self._psrpos, params=params)

                return prior * orf

            @classmethod
            def get_phicross(cls, signal1, signal2, params):
                prior = BasisCommonGP._prior(signal1._labels, params=params)
                # orf = BasisCommonGP._orf(signal1._labels, signal1._psrpos, signal2._psrpos, signal1._psrdis, signal2._psrdis, params=params)
                orf = BasisCommonGP._orf(signal1._labels, signal1._psrpos, signal2._psrpos, params=params)

                return prior * orf

    return BasisCommonGP


def FourierBasisCommonGP(
    spectrum,
    orf,
    coefficients=False,
    combine=True,
    components=20,
    Tspan=None,
    modes=None,
    name="common_fourier",
    pshift=False,
    pseed=None,
):

    if coefficients and Tspan is None:
        raise ValueError(
            "With coefficients=True, FourierBasisCommonGP " + "requires that you specify Tspan explicitly."
        )

    basis = utils.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, pshift=pshift, pseed=pseed)
    BaseClass = BasisCommonGP(spectrum, basis, orf, coefficients=coefficients, combine=combine, name=name)

    class FourierBasisCommonGP(BaseClass):
        signal_type = "common basis"
        signal_name = "common red noise"
        signal_id = name

        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", 1)
        def _construct_basis(self, params={}):
            span = Tspan if Tspan is not None else max(FourierBasisCommonGP._Tmax) - min(FourierBasisCommonGP._Tmin)
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP

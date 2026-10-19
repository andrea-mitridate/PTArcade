"""Pseudo-priors for model selection in discovery mode.

Model selection samples both models in one chain, with a model index `nmodel`. While one
model is active, the parameters that only the other model uses are drawn from a
pseudo-prior instead of their prior. Any normalized pseudo-prior gives the right Bayes
factor; the closer it is to that model's posterior, the more often the chain switches
models, and the cheaper the Bayes factor.

The pseudo-prior here is a block neural autoregressive flow (flowjax), fit by maximum
likelihood to posterior samples from an earlier run of the same model (any mode), in
numpyro's unconstrained space, where it needs no bounds. The flow is smooth, which keeps
NUTS's step size up. Each run saves its fits, which later runs reuse.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from numpyro import distributions as dist
from numpyro.distributions.transforms import biject_to

from ptarcade import chains_utils

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax.typing import ArrayLike

log = logging.getLogger("rich")

# flow and training settings, fixed: the sampler only needs a rough fit
FLOW = {"flow_layers": 2, "nn_depth": 1, "nn_block_dim": 8}
TRAIN = {"learning_rate": 1e-3, "max_epochs": 1000, "max_patience": 50, "batch_size": 256, "val_prop": 0.2}
BURN_FRAC = 0.25


@dataclass(frozen=True)
class PseudoPrior:
    """A normalized density over some numpyro sites, used in place of their prior.

    Attributes
    ----------
    sites : dict[str, numpyro.distributions.Distribution]
        The names of the sites it covers and their priors, in a fixed order.
    flow : flowjax.distributions.Transformed
        Density of the sites' concatenated unconstrained values.

    """

    sites: dict[str, dist.Distribution]
    flow: Transformed

    def log_ratio(self, values: dict[str, jax.Array]) -> jax.Array:
        """Compute the log pseudo-prior minus the log prior at the sites' values.

        Both densities are taken in the unconstrained space, where the Jacobians of the
        bijections to the sites' supports cancel.

        Parameters
        ----------
        values : dict[str, jax.Array]
            The value of each site in `sites`, in its constrained space.

        Returns
        -------
        log_ratio : jax.Array
            The scalar log pseudo-prior minus the log prior.

        """
        u, log_prior = [], 0.0
        for site, prior in self.sites.items():
            to_support = biject_to(prior.support)
            u_site = to_support.inv(values[site])
            log_prior = log_prior + prior.log_prob(values[site]) + jnp.sum(
                to_support.log_abs_det_jacobian(u_site, values[site])
            )
            u.append(jnp.ravel(u_site))
        return self.flow.log_prob(jnp.concatenate(u)) - log_prior


def flow_skeleton(dim: int) -> Transformed:
    """Build an untrained flow over standardized values.

    Parameters
    ----------
    dim : int
        The dimension of the flow.

    Returns
    -------
    flow : flowjax.distributions.Transformed
        A block neural autoregressive flow with settings `FLOW` and a standard normal base.

    """
    return block_neural_autoregressive_flow(jax.random.key(0), base_dist=Normal(jnp.zeros(dim)), **FLOW)


def standardized(flow: Transformed, mean: ArrayLike, std: ArrayLike) -> Transformed:
    """Turn a density over standardized values into one over the unstandardized values.

    Loading saved weights needs a skeleton of the same shape,
    `standardized(flow_skeleton(dim), jnp.zeros(dim), jnp.ones(dim))`.

    Parameters
    ----------
    flow : flowjax.distributions.Transformed
        Density of `(x - mean) / std`.
    mean : array_like
        The mean of each dimension of `x`.
    std : array_like
        The standard deviation of each dimension of `x`. It has to be positive.

    Returns
    -------
    flow : flowjax.distributions.Transformed
        Density of `x`, with the Jacobian of the standardization included.

    """
    return Transformed(flow, Affine(jnp.asarray(mean), jnp.asarray(std)))


def chain_names(site: str) -> list[tuple[str, Callable | None]]:
    """List the names a site can have in the chain of an earlier run.

    Parameters
    ----------
    site : str
        The name of the site.

    Returns
    -------
    list[tuple[str, Callable | None]]
        Each name, with the map from the chain's values to the site's, or None if they are
        the same.

    """
    if site.endswith("__lin"):
        # a log-linear prior is sampled as `<par>__lin`; chains store `<par>`, its log10
        return [(site.removesuffix("__lin"), lambda x: 10.0**x)]
    if site.startswith("gw_bhb_np"):
        # a new-physics model's SMBHB is `gw_bhb_np` in model selection and enterprise
        # runs, but `gw_bhb` in other discovery runs and in ceffyl runs
        return [(site, None), (site.replace("gw_bhb_np", "gw_bhb", 1), None)]
    return [(site, None)]


def site_samples(chain: pd.DataFrame, sites: dict[str, dist.Distribution]) -> dict[str, np.ndarray]:
    """Get the unconstrained values of the sites in each row of a chain.

    Parameters
    ----------
    chain : pandas.DataFrame
        Samples from a run of PTArcade in any mode, one column per parameter.
    sites : dict[str, numpyro.distributions.Distribution]
        The names of the sites and their priors, in a fixed order.

    Returns
    -------
    dict[str, numpy.ndarray]
        The unconstrained values of each site, shape (n_rows, size of the site). A value
        outside the site's prior is not finite.

    Raises
    ------
    ValueError
        If `chain` has no samples of some of the sites.

    """
    columns = set(chain.columns)
    u, missing = {}, []
    for site, prior in sites.items():
        for name, to_site in chain_names(site):
            cols = [f"{name}_{i}" for i in range(prior.event_shape[0])] if prior.event_shape else [name]
            if columns.issuperset(cols):
                x = chain[cols].to_numpy()
                x = to_site(x) if to_site else x
                x = x if prior.event_shape else x[:, 0]
                u[site] = np.asarray(biject_to(prior.support).inv(jnp.asarray(x))).reshape(len(chain), -1)
                break
        else:
            missing.append(site)
    if missing:
        msg = (f"The pseudo-prior chain has no samples of {missing}. It has to be a run of the same "
               f"model (and SMBHB prior) as this one. Its columns are {sorted(columns)}.")
        raise ValueError(msg)
    return u


def chain_files(source: Path) -> list[list]:
    """Record the chain files of a run, to tell whether a fit to it is still current.

    Parameters
    ----------
    source : Path
        A run's output directory, `out_dir/<model name>/`.

    Returns
    -------
    list[list]
        The path, size and modification time of each `chain_1.*` file in `source` or its
        chain directories.

    """
    files = sorted(source.glob("chain_1.*")) + sorted(source.glob("*/chain_1.*"))
    return [[str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def read_chain(source: Path) -> pd.DataFrame:
    """Read the chains of a run.

    Discovery mode writes `chain_1.feather` files, which hold no warmup. Enterprise and
    ceffyl modes write `chain_1.txt` files, whose first `BURN_FRAC` is dropped as burn-in.

    Parameters
    ----------
    source : Path
        A run's output directory, `out_dir/<model name>/`.

    Returns
    -------
    pandas.DataFrame
        The samples of all its chains.

    """
    feathers = sorted(source.glob("chain_1.feather")) or sorted(source.glob("*/chain_1.feather"))
    if feathers:
        return pd.concat(
            [chains_utils.import_to_dataframe(f.parent, chain_name=f.name)[1] for f in feathers], ignore_index=True,
        )
    return chains_utils.import_to_dataframe(source, burn_frac=BURN_FRAC)[1]


def fit(u: np.ndarray, source: Path) -> tuple[Transformed, float]:
    """Fit a flow to samples.

    Parameters
    ----------
    u : numpy.ndarray
        The samples, shape (n_samples, dim).
    source : Path
        The run they come from, for messages.

    Returns
    -------
    flow : flowjax.distributions.Transformed
        Density of the samples.
    val_loss : float
        The best validation loss of the fit.

    Raises
    ------
    ValueError
        If the validation loss is not finite.

    """
    mean, std = u.mean(0), u.std(0)
    if np.any(fixed := std == 0):
        # a dimension that never moves in the chain: only shift it, as scaling by 0 is singular
        log.warning(f"The pseudo-prior chain {source} has constant columns {np.flatnonzero(fixed).tolist()}")
        std = np.where(fixed, 1.0, std)
    flow, losses = fit_to_data(
        jax.random.key(1), flow_skeleton(u.shape[1]), jnp.asarray((u - mean) / std), show_progress=True, **TRAIN,
    )
    val_loss = float(min(losses["val"]))
    if not np.isfinite(val_loss):
        msg = f"The pseudo-prior fit to {source} failed: its validation loss is {val_loss}."
        raise ValueError(msg)
    return standardized(flow, mean, std), val_loss


def load(path: Path, sites: dict[str, dist.Distribution]) -> PseudoPrior:
    """Load a saved fit.

    Parameters
    ----------
    path : Path
        The `.eqx` file of the fit, next to its `.json`.
    sites : dict[str, numpyro.distributions.Distribution]
        The names of the sites and their priors, in a fixed order.

    Returns
    -------
    pseudo_prior : PseudoPrior
        The pseudo-prior over `sites`.

    Raises
    ------
    ValueError
        If the fit covers different sites.

    """
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["sites"] != list(sites):
        msg = f"The pseudo-prior {path} covers the sites {meta['sites']}, but this run needs {list(sites)}."
        raise ValueError(msg)
    log.info(f"Using the pseudo-prior {path}")
    dim = meta["dim"]
    skeleton = standardized(flow_skeleton(dim), jnp.zeros(dim), jnp.ones(dim))
    return PseudoPrior(sites, eqx.tree_deserialise_leaves(path, skeleton))


def load_or_fit(source: str | Path, sites: dict[str, dist.Distribution], save_as: Path) -> PseudoPrior:
    """Fit the pseudo-prior over some sites to an earlier run, or load a saved fit.

    A new fit is saved as `save_as.{eqx,json}`. A fit already there is reused when it was
    made to the same sites and the same, unchanged chain files of `source`. A saved fit only
    has to cover the same sites: with any priors it stays a normalized density over the
    current ones, which keeps the Bayes factor exact and at worst lowers the switch rate.

    Parameters
    ----------
    source : str | Path
        Either a run's output directory, `out_dir/<model name>/`, to fit to, or the `.eqx`
        file of a fit saved by an earlier model selection run.
    sites : dict[str, numpyro.distributions.Distribution]
        The names of the sites and their priors, in a fixed order.
    save_as : Path
        Where to save a new fit, without a suffix.

    Returns
    -------
    pseudo_prior : PseudoPrior
        The pseudo-prior over `sites`.

    Raises
    ------
    ValueError
        If a saved fit covers different sites, the chain in `source` has no samples of
        some of them or none inside their priors, or the fit fails.

    """
    source = Path(source)
    if source.is_file():
        return load(source, sites)

    eqx_path, json_path = save_as.with_suffix(".eqx"), save_as.with_suffix(".json")
    current = {"source": str(source.resolve()), "sites": list(sites), "chain_files": chain_files(source)}
    if eqx_path.is_file() and json_path.is_file():
        saved = json.loads(json_path.read_text())
        changed = [key for key, val in current.items() if saved.get(key) != val]
        if not changed:
            return load(eqx_path, sites)
        log.info(f"Refitting the pseudo-prior {eqx_path}: its {', '.join(changed)} changed")

    u = site_samples(read_chain(source), sites)
    # a sample outside this run's priors (e.g. after narrowing one) is not finite in their
    # unconstrained space. Fitting to the others still gives a normalized density over the
    # priors, so they are left out
    outside = np.any([~np.isfinite(x).all(axis=1) for x in u.values()], axis=0)
    if outside.any():
        outside_sites = [site for site, x in u.items() if not np.isfinite(x).all()]
        if outside.all():
            msg = (f"All samples of the pseudo-prior chain {source} lie outside this run's priors of "
                   f"{outside_sites}: it has to be a run with priors that overlap these.")
            raise ValueError(msg)
        log.warning(
            f"{outside.sum()} of {len(outside)} samples of the pseudo-prior chain {source} lie outside "
            f"this run's priors of {outside_sites}; fitting to the other {(~outside).sum()}",
        )
    samples = np.concatenate(list(u.values()), axis=1)[~outside]

    log.info(f"Fitting the pseudo-prior of {list(sites)} to {len(samples)} samples from {source}")
    flow, val_loss = fit(samples, source)
    save_as.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(eqx_path, flow)
    meta = current | {"dim": samples.shape[1], "n_samples": len(samples), "val_loss": val_loss}
    json_path.write_text(json.dumps(meta))
    log.info(f"Saved the pseudo-prior as {eqx_path}")
    return PseudoPrior(sites, flow)


def log_ratios(
    config: ModuleType,
    model_sites: dict[str, dist.Distribution],
    null_sites: dict[str, dist.Distribution],
    save_dir: Path,
) -> tuple[Callable[[dict[str, jax.Array]], jax.Array], Callable[[dict[str, jax.Array]], jax.Array]]:
    """Load or fit the pseudo-priors of a model selection run.

    New fits are saved in `save_dir` as `pseudo_prior.{eqx,json}` and
    `null_pseudo_prior.{eqx,json}`, after the configuration keys that name their runs.

    Parameters
    ----------
    config : ModuleType
        The configuration file. Its `pseudo_prior` and `null_pseudo_prior` name the runs
        to fit to, or saved fits.
    model_sites : dict[str, numpyro.distributions.Distribution]
        The sites only the new-physics model uses, and their priors. Without any, the
        model's parameters keep their prior.
    null_sites : dict[str, numpyro.distributions.Distribution]
        The sites only the SMBHB-only null uses, and their priors.
    save_dir : Path
        Where new fits are saved.

    Returns
    -------
    model_log_ratio, null_log_ratio : Callable[[dict[str, jax.Array]], jax.Array]
        The log pseudo-prior minus the log prior of each model's own sites, see
        `PseudoPrior.log_ratio`.

    """
    if model_sites:
        model_log_ratio = load_or_fit(config.pseudo_prior, model_sites, save_dir / "pseudo_prior").log_ratio
    else:
        def model_log_ratio(_: dict[str, jax.Array]) -> jax.Array:
            # an array, as a pseudo-prior's log ratio, for the branches of lax.cond to match
            return jnp.zeros(())
    null_log_ratio = load_or_fit(config.null_pseudo_prior, null_sites, save_dir / "null_pseudo_prior").log_ratio
    return model_log_ratio, null_log_ratio

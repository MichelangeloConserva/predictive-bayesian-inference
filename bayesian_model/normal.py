from typing import NamedTuple

import tensorflow_probability.substrates.jax as tpf
from dynamax.utils.distributions import NormalInverseGamma

tfd = tpf.distributions


class NormalModel(NamedTuple):
    mu: float
    tau: float

    posterior_dist = tfd.Normal
    posterior_predictive_dist = tfd.StudentT

    class Hyperparameters(NamedTuple):
        mu_0: float
        nu: float
        alpha: int
        beta: int

    class SufficientStatistics(NamedTuple):
        sum_x: float
        sum_x2: float
        sample_size: int

    @property
    def likelihood(self):
        return tfd.Normal(self.mu, self.tau)

    def posterior_update(
        self, ss: SufficientStatistics, prior_hps: Hyperparameters
    ) -> Hyperparameters:

        # extract parameters of the prior distribution
        prior_loc, prior_precision, prior_df, prior_scale = prior_hps

        # unpack the sufficient statistics
        sum_x, sum_x2, n = ss

        # compute parameters of the posterior distribution
        posterior_precision = prior_precision + n
        posterior_df = prior_df + n / 2
        posterior_loc = (prior_precision * prior_loc + sum_x) / posterior_precision

        posterior_scale = prior_scale + 0.5 * (
            sum_x2
            + prior_precision * prior_loc**0.5
            - posterior_precision * posterior_loc**0.5
        )
        return NormalModel.Hyperparameters(
            posterior_loc, posterior_precision, posterior_df, posterior_scale
        )

    def posterior_predictive(self, posterior_hps: Hyperparameters):
        post_loc, post_precision, post_df, post_scale = posterior_hps
        return (
            2 * post_df,
            post_loc,
            (post_scale * (post_precision + 1)) / (post_precision * post_df),
        )

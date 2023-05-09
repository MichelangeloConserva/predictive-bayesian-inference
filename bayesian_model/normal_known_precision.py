from typing import NamedTuple

import distrax


class NormalModelKnownPrecision(NamedTuple):
    mu: float
    tau: float

    posterior_dist = distrax.Normal
    posterior_predictive_dist = distrax.Normal

    class Hyperparameters(NamedTuple):
        mean: float
        precision: float

    class SufficientStatistics(NamedTuple):
        sample_sum: float
        sample_size: int

    @property
    def likelihood(self):
        return distrax.Normal(self.mu, self.tau)

    def posterior_update(
        self, ss: SufficientStatistics, prior_hps: Hyperparameters
    ) -> Hyperparameters:

        sum_x, n = ss
        mu_0, tau_0 = prior_hps

        post_prec = (tau_0 + n * self.tau)
        return NormalModelKnownPrecision.Hyperparameters(
            (tau_0 * mu_0 + self.tau * sum_x) / post_prec, post_prec
        )

    def posterior_predictive(self, posterior_hps: Hyperparameters):
        mu_post, prec_post = posterior_hps
        return mu_post, (prec_post**-1 + self.tau**-1)**.5

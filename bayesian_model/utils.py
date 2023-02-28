from bayesian_model.normal_known_precision import NormalModelKnownPrecision


def get_kl_likelihood_posterior_predictive(
    model: NormalModelKnownPrecision,
    posterior_hps: NormalModelKnownPrecision.Hyperparameters,
):
    return model.likelihood.kl_divergence(
        model.posterior_predictive_dist(*posterior_hps)
    )

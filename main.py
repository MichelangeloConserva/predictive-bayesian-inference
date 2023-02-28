import matplotlib.pyplot as plt

import jax
import haiku as hk
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde, norm
from tqdm import trange

jax.config.update("jax_platform_name", "cpu")

from bayesian_model.normal import NormalModel
from bayesian_model.normal_known_precision import NormalModelKnownPrecision
from bayesian_model.utils import get_kl_likelihood_posterior_predictive

tau_star = 1.0
mu_star = 5.0
N = 1000
B = 500

rng = hk.PRNGSequence(42)

nkp = NormalModelKnownPrecision(mu_star, tau_star)
prior = nkp.Hyperparameters(0, 1)

final_posterior_mean = []
posterior_mean_trajs = []
samples = []

theta_0 = 15
sigma_squared_0 = 2.4
sigma_squared = 5
#%%
for b in trange(B):
    n = 0
    theta_n = theta_0
    sigma_squared_n = sigma_squared_0
    y_b = []

    posterior_mean_traj = []
    for i in range(N):
        y = np.random.normal(theta_n, (sigma_squared + sigma_squared_n) ** 0.5)
        y_b.append(y)
        n = len(y_b)
        sigma_squared_n = (1 / sigma_squared_0 + n / sigma_squared) ** -1
        theta_n = (
            theta_0 / sigma_squared_0 + sum(y_b) / sigma_squared
        ) * sigma_squared_n
        posterior_mean_traj.append(theta_n)
    final_posterior_mean.append(posterior_mean_traj[-1])
    posterior_mean_trajs.append(posterior_mean_traj)
    samples.append(y_b)


#%% Posterior sampling
fig, axes = plt.subplots(1, 2, figsize=plt.figaspect(0.5), sharey=True)
for t in posterior_mean_trajs:
    axes[0].plot(t)
axes[0].set_xlabel("Forward step")
axes[0].set_ylabel("Posterior mean")
nn = norm(theta_0, sigma_squared_0**0.5)
x = np.linspace(nn.ppf(0.0005), nn.ppf(1 - 0.0005), 200)
axes[1].set_xlabel("Density")
axes[1].plot(nn.pdf(x), x, color="black")
axes[1].plot(gaussian_kde(final_posterior_mean)(x), x)
plt.tight_layout()
plt.show()

#%% Empirical cumulative distributions

x = np.linspace(nn.ppf(0.0005), nn.ppf(1 - 0.0005), 200)
for y_b in samples:
    sns.ecdfplot(y_b)
plt.plot(x, nn.cdf(x), color="black", label="True posterior cumulative distribution")
plt.legend()
plt.tight_layout()
plt.show()


#%% Bayesian bootstrap
N = 10

y_n = np.random.normal(5, 1, N)
w = np.random.dirichlet([1] * len(y_n))
y = np.linspace(0, 10, 500)

plt.plot(
    y,
    (w.reshape(-1, 1) * (y_n.reshape(-1, 1) <= y.reshape(1, -1))).sum(0),
    label="Sample from the\nBayesian bootstrap posterior",
)
plt.plot(
    y, norm(5, 1).cdf(y), color="black", label="True posterior\ncumulative distribution"
)
plt.legend()
plt.show()

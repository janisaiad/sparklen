"""
===============================================================================
Inference of a Multivariate Hawkes Process (MHP)
===============================================================================

The following provides a comprehensive guide to using 
:class:`~sparkle.hawkes.inference.LearnerHawkesExp` to estimate a MHP.
"""

# Author: Romain E. Lacoste
# License: BSD-3-Clause

# Setup environment -----------------------------------------------------------

import numpy as np

from sparkle.hawkes.inference import LearnerHawkesExp
from sparkle.hawkes.simulation import SimuHawkesExp
from sparkle.plot import plot_values

# Set the true coefficients ---------------------------------------------------

d = 5
beta = 3.0

mu = np.array([0.6, 0.55, 0.6, 0.55, 0.6])

alpha = np.zeros((d,d))
alpha[:4, :4] += 0.1
alpha[2:, 2:] += 0.15

theta_star = np.hstack([np.reshape(mu, (d,-1)), alpha])

# Plot the true coefficients --------------------------------------------------

plot_values(theta_star)

# Simulate training data ------------------------------------------------------

T = 5.0
n = 1000

hawkes = SimuHawkesExp(
    mu=mu, alpha=alpha, beta=beta, 
    end_time=T, n_samples=n,
    random_state=4)

hawkes.simulate()
data = hawkes.timestamps

# Perform estimation ----------------------------------------------------------

learner = LearnerHawkesExp(
    decay=beta, loss="least-squares", penalty="none", 
    optimizer="agd", lr_scheduler="backtracking", 
    max_iter=200, tol=1e-5, 
    verbose_bar=True, verbose=True, 
    print_every=10, record_every=10)

learner.fit(data, T)
theta_hat = learner.estimated_params
print(theta_hat)

print(learner.score(data, T))

# Plot the estimated coefficients ---------------------------------------------

learner.plot_estimated_values()
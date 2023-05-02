"""
@gilbellosta, 2023-05-01
"""

import streamlit as st

import numpy as np
import pandas as pd

import scipy.stats as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.style.use('dark_background')

n_points = 1000

def loss(parms, q1, q2, p1, p2, qfoo):
    """
    Loss function. We want to reduce the (quadratic) error between the
    target quantiles and the quantiles of the distribution with the given
    parameters.
    """
    q_target     = np.array([q1, q2])
    q_candidates = qfoo((p1, p2), parms[0], parms[1])
    return np.square(q_candidates - q_target).sum()

def my_plot(x, y, q1, q2):
    fig, ax = plt.subplots()
    ax.tick_params(left = False, right = False , labelleft = False)
    ax.plot(x, y, color = 'red', alpha = .6)

    plt.axvline(x = q1, color = 'blue', alpha = 0.5)
    plt.axvline(x = q2, color = 'blue', alpha = 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title('Distribution parameters given quantiles')

st.sidebar.markdown("""
This application provides the parameters of a given distribution
—for a selection of them—
given its quantiles.

It is intended to facilitate the creation of informative priors in Bayesian modeling.
""")

select_distribution = st.sidebar.radio(
    'Select your distribution',
    ('Normal', 't', 'Gamma', 'Inverse Gamma', 'LogNormal', 'Beta')
)

p1_pct = st.sidebar.slider(
    'Select the lower quantile (%)',
    min_value = 0.0, max_value = 50.0,
    value = 5.0, step = .1
)

p2_pct = st.sidebar.slider(
    'Select the upper quantile (%)',
    min_value = 50.0, max_value = 100.0,
    value = 95.0, step = .1
)

p1 = p1_pct / 100.0
p2 = p2_pct / 100.0

# -----------------------------------------------------------------------------
# Central panel
# -----------------------------------------------------------------------------

if select_distribution == 'Normal':

    st.title('Normal distribution')

    st.markdown(r"""
    Standard normal distribution $N(\mu, \sigma)$.
    """)

    full_range = st.slider("Set the graph range",
        min_value = -50.0,
        max_value =  50.0,
        step = .01,
        value = [-5.0, 5.0]
    )

    my_range = st.slider("Set the quantile range",
        min_value = full_range[0],
        max_value = full_range[1],
        step = .01,
        value = [-2.0, 2.0])

    res = minimize(
            loss, np.ones(2), method='Nelder-Mead',
            tol=1e-6, args=(my_range[0], my_range[1], p1, p2, ss.norm.ppf))

    x = np.linspace(full_range[0], full_range[1], n_points)
    y = ss.norm.pdf(x, res['x'][0], scale = 1 / res['x'][1])

    st.pyplot(my_plot(x, y, my_range[0], my_range[1]))

    st.markdown(rf"""
        The parameters $\mu$ and $\sigma$ of the normal distribution are
        are:

        $\mu$: {res['x'][0]}

        $\sigma$: {res['x'][1]}
        """)

elif select_distribution == 't':

    st.title('t distribution')

    st.markdown(r"""
    The $\mu$ and $\sigma$ parameters are such that $\frac{X - \mu}{\sigma} \sim t(\nu)$.
    """)

    nu = st.slider(r"Set $\nu$, the number of degrees of freedom",
        min_value = 0.0,
        max_value = 50.0,
        step = .1,
        value = 1.0
    )

    def qt(p, mu, sigma, nu = nu):
        return ss.t.ppf(p, nu, mu, sigma)

    full_range = st.slider("Set the full graph range",
        min_value = -50.0,
        max_value =  50.0,
        step = .01,
        value = [-5.0, 5.0]
    )

    my_range = st.slider("Set the quantile range",
        min_value = full_range[0],
        max_value = full_range[1],
        step = .01,
        value = [-2.0, 2.0])

    res = minimize(
            loss, np.ones(2), method='Nelder-Mead',
            tol=1e-6, args=(my_range[0], my_range[1], p1, p2, qt))

    x = np.linspace(full_range[0], full_range[1], n_points)
    y = ss.t.pdf(x, nu, res['x'][0], scale = 1 / res['x'][1])

    st.pyplot(my_plot(x, y, my_range[0], my_range[1]))

    st.markdown(rf"""
        The parameters $\mu$ and $\sigma$ of the t distribution with {nu} degrees of freedom
        are:

        $\mu$: {res['x'][0]}

        $\sigma$: {res['x'][1]}
        """)

elif select_distribution == 'Gamma':

    st.title('Gamma distribution')

    st.markdown(r"""
    The $\alpha$ and $\beta$ parameters are such that the density function is $f(x) \sim x^{\alpha -1} \exp(-\beta x)$.
    """)

    def qgamma(p, alpha, beta):
        return ss.gamma.ppf(p, alpha, scale = 1 / beta)

    full_range = st.slider("Set the full figure range",
        min_value = 0.0,
        max_value = 100.0,
        step = .01,
        value = [0.0, 5.0]
    )

    my_range = st.slider("Set the quantile range",
        min_value = full_range[0],
        max_value = full_range[1],
        step = .01,
        value = [1.0, 4.0])

    res = minimize(
        loss, np.ones(2), method='Nelder-Mead',
        tol=1e-6, args=(my_range[0], my_range[1], p1, p2, qgamma))

    x = np.linspace(full_range[0], full_range[1], n_points)
    y = ss.gamma.pdf(x, res['x'][0], scale = 1 / res['x'][1])

    st.pyplot(my_plot(x, y, my_range[0], my_range[1]))

    st.markdown(rf"""
        The parameters $\alpha$ and $\beta$ of the
        [Gamma distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
        are:

        $\alpha$: {res['x'][0]}

        $\beta$: {res['x'][1]}
        """)

elif select_distribution == 'Inverse Gamma':

    st.title('Inverse Gamma distribution')

    st.markdown(r"""
    The $\alpha$ and $\beta$ parameters are such that $1/X \sim \Gamma(\alpha, \beta)$.
    """)

    def qgamma(p, alpha, beta):
        return ss.gamma.ppf(p, alpha, scale = 1 / beta)

    full_range = st.slider("Set the full figure range",
        min_value = 0.0,
        max_value = 100.0,
        step = .01,
        value = [0.0, 5.0]
    )

    my_range = st.slider("Set the quantile range",
        min_value = full_range[0],
        max_value = full_range[1],
        step = .01,
        value = [1.0, 4.0])

    res = minimize(
        loss, np.ones(2), method='Nelder-Mead',
        tol=1e-6, args=(1 / my_range[1], 1 / my_range[0], p1, p2, qgamma))

    x = np.linspace(full_range[0], full_range[1], n_points)
    y = ss.invgamma.pdf(x, res['x'][0], scale = res['x'][1])

    st.pyplot(my_plot(x, y, my_range[0], my_range[1]))

    st.markdown(rf"""
        The parameters $\alpha$ and $\beta$ of the
        [Gamma distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
        are:

        $\alpha$: {res['x'][0]}

        $\beta$: {res['x'][1]}
        """)

elif select_distribution == 'LogNormal':

    st.title('Lognormal distribution')

    st.markdown(r"""
    The fitted parameters are $\mu$, $\sigma$ such that $\log(X) \sim N(\mu, \sigma)$.
    """)

    def qnormal(p, mu, sigma):
        return ss.norm.ppf(p, mu, sigma)

    full_range = st.slider("Set the full range",
        min_value = 0.0,
        max_value = 100.0,
        step = .01,
        value = [0.0, 5.0]
    )

    r1, r2 = st.slider("Set the quantile range",
        min_value = full_range[0],
        max_value = full_range[1],
        step = .01,
        value = [1.0, 4.0])

    res = minimize(
            loss, np.ones(2), method='Nelder-Mead',
            tol=1e-6, args=(np.log(r1), np.log(r2), p1, p2, qnormal))

    mu = res['x'][0]

    x = np.linspace(full_range[0], full_range[1], n_points)
    y = ss.lognorm.pdf(x, res['x'][1], scale = np.exp(mu))

    st.pyplot(my_plot(x, y, r1, r2))

    st.markdown(rf"""
        The parameters $\mu$ and $\sigma$ of the lognormal distribution are:

        $\mu$: {res['x'][0]}

        $\sigma$: {res['x'][1]}
        """)

elif select_distribution == 'Beta':

    st.title('Beta distribution')

    st.markdown(r"""
    The fitted parameters are $\alpha$, $\beta$ such that the density
    function $f(x) \sim x^{\alpha -1} (1-x)^{\beta - 1}$.
    """)

    r1, r2 = st.slider("Set the quantile range",
        min_value = 0.0,
        max_value = 1.0,
        step = .01,
        value = [0.2, 0.8])

    res = [minimize(
                loss,
                np.array([a, b]),
                method='Nelder-Mead',
                tol=1e-6,
                args=(r1, r2, p1, p2, ss.beta.ppf),
                bounds = ((0, None), (0, None))
            )
            for a in [1.0, 10.0, 50.0]
            for b in [1.0, 10.0, 50.0]]

    res = min(res, key=lambda x: x.fun)

    x = np.linspace(0.0, 1.0, n_points)
    y = ss.beta.pdf(x, res['x'][0], res['x'][1])

    st.pyplot(my_plot(x, y, r1, r2))

    st.markdown(rf"""
        The parameters $\alpha$ and $\beta$ of the beta distribution are:

        $\alpha$: {res['x'][0]}

        $\beta$: {res['x'][1]}
        """)

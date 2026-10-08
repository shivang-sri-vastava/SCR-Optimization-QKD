#!/usr/bin/env python3
"""Trusted-receiver 64-QAM key-rate backend.

Quadrature convention: [x,p]=2i and vacuum variance 1 SNU.
The arbitrary-modulation covariance bound follows Denys, Brown and Leverrier.
Bob's calibrated receiver is represented by the trusted beam-splitter/EPR model
used in Aymeric and Roumestan.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, log2
from typing import Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr


@dataclass(frozen=True)
class TrustedKeyRateResult:
    V_A: float
    T_ch: float
    xi_excess_input: float
    xi_rec: float
    xi_detected: float
    eta_B: float
    beta: float
    mutual_information: float
    holevo_information: float
    secret_fraction: float
    C_tau: float
    w_tau: float
    Z_star: float
    nu_1: float
    nu_2: float
    nu_3: float
    nu_4: float
    nu_5: float


def binomial_64qam(V_A: float) -> Tuple[np.ndarray, np.ndarray]:
    """Coherent amplitudes and binomial probabilities for square 64-QAM."""
    if V_A <= 0:
        raise ValueError("V_A must be positive")
    m = 8
    alpha = np.sqrt(V_A / 2.0)
    scale = alpha * np.sqrt(2.0 / (m - 1))
    denominator = float(2 ** (2 * (m - 1)))
    points, probabilities = [], []
    for k in range(m):
        for ell in range(m):
            points.append(scale * ((k - 3.5) + 1j * (ell - 3.5)))
            probabilities.append(comb(7, k) * comb(7, ell) / denominator)
    return np.asarray(points, complex), np.asarray(probabilities, float)


def _coherent_state(alpha: complex, cutoff: int) -> np.ndarray:
    state = np.empty(cutoff, complex)
    state[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        state[n] = state[n - 1] * alpha / np.sqrt(float(n))
    return state


def constellation_quantities(V_A: float, cutoff: int = 60, eig_rtol: float = 1e-12) -> Tuple[float, float]:
    """Return C_tau=Tr(sqrt(tau)a sqrt(tau)a^dagger) and w."""
    points, probs = binomial_64qam(V_A)
    states = np.column_stack([_coherent_state(a, cutoff) for a in points])
    tau = (states * probs[np.newaxis, :]) @ states.conj().T
    tau = 0.5 * (tau + tau.conj().T)

    vals, vecs = eigh(tau, check_finite=False)
    vals = np.maximum(vals.real, 0.0)
    support = vals > max(vals.max() * eig_rtol, 1e-15)
    if not np.any(support):
        raise RuntimeError("Empty numerical support for tau")
    U, lam = vecs[:, support], vals[support]
    sqrt_tau = (U * np.sqrt(lam)[np.newaxis, :]) @ U.conj().T
    inv_sqrt_tau = (U * (1.0 / np.sqrt(lam))[np.newaxis, :]) @ U.conj().T

    a_op = np.zeros((cutoff, cutoff), complex)
    n = np.arange(1, cutoff)
    a_op[n - 1, n] = np.sqrt(n)
    adag = a_op.conj().T

    C_tau = float(np.trace(sqrt_tau @ a_op @ sqrt_tau @ adag).real)
    a_tau = sqrt_tau @ a_op @ inv_sqrt_tau
    ata = a_tau.conj().T @ a_tau
    w = 0.0
    for p, ket in zip(probs, states.T):
        mean = np.vdot(ket, a_tau @ ket)
        second = np.vdot(ket, ata @ ket).real
        w += float(p) * max(float(second - abs(mean) ** 2), 0.0)
    return C_tau, max(w, 0.0)


def _pam8_distribution() -> Tuple[np.ndarray, np.ndarray]:
    levels = np.arange(8, dtype=float) - 3.5
    probs = np.asarray([comb(7, k) / 128.0 for k in range(8)], float)
    variance = float(np.dot(probs, levels ** 2))
    return levels / np.sqrt(variance), probs


@lru_cache(maxsize=16)
def _gh_nodes(order: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(order)
    return nodes, weights / np.sqrt(np.pi)


def pam8_mutual_information_from_snr(snr: float, gh_order: int = 32) -> float:
    """MI for unit-variance binomial 8-PAM over Y=sqrt(snr)X+N, Var(N)=1."""
    if snr <= 0:
        return 0.0
    levels, probs = _pam8_distribution()
    means = np.sqrt(snr) * levels
    nodes, weights = _gh_nodes(gh_order)
    log_probs = np.log(probs)
    total = 0.0
    for mu_i, p_i in zip(means, probs):
        y = mu_i + np.sqrt(2.0) * nodes
        log_terms = log_probs[np.newaxis, :] - 0.5 * (y[:, None] - means[None, :]) ** 2
        log_py = logsumexp(log_terms, axis=1)
        log_pyi = -0.5 * (y - mu_i) ** 2
        total += p_i * float(np.dot(weights, (log_pyi - log_py) / np.log(2.0)))
    return max(total, 0.0)


def joint_quantized_heterodyne_information(
    V_A: float,
    T_ch: float,
    xi_excess_input: float,
    eta_B: float,
    xi_rec: float,
    bits_per_quadrature: int = 6,
    range_sigma: float = 4.0,
) -> Tuple[float, float]:
    """Return joint ``H(X_B)`` and ``I(X;X_B)`` for clipped I/Q quantization.

    Each heterodyne quadrature is quantized by an identical uniform scalar
    quantizer over ``[-range_sigma*sigma_B, range_sigma*sigma_B]``.  The two
    outer cells absorb the Gaussian tails.  Square 64-QAM and the detected
    channel factorize across quadratures, so the joint entropy and mutual
    information are twice their scalar values.
    """
    if V_A <= 0.0:
        raise ValueError("V_A must be positive")
    if not (0.0 < T_ch <= 1.0):
        raise ValueError("T_ch must be in (0,1]")
    if not (0.0 < eta_B < 1.0):
        raise ValueError("eta_B must be in (0,1)")
    if xi_excess_input < 0.0 or xi_rec < 0.0:
        raise ValueError("Noise variances must be non-negative")
    if bits_per_quadrature < 1 or range_sigma <= 0.0:
        raise ValueError("Invalid quantizer specification")

    levels, input_probabilities = _pam8_distribution()
    gain_squared = eta_B * T_ch / 2.0
    noise_variance = 1.0 + xi_rec + gain_squared * xi_excess_input
    noise_sigma = np.sqrt(noise_variance)
    conditional_means = np.sqrt(gain_squared * V_A) * levels
    output_sigma = np.sqrt(gain_squared * V_A + noise_variance)

    number_of_bins = 2 ** int(bits_per_quadrature)
    bin_width = 2.0 * range_sigma * output_sigma / number_of_bins
    internal_edges = (
        -range_sigma * output_sigma
        + bin_width * np.arange(1, number_of_bins, dtype=float)
    )
    edges = np.r_[-np.inf, internal_edges, np.inf]

    z_upper = (edges[1:, None] - conditional_means[None, :]) / noise_sigma
    z_lower = (edges[:-1, None] - conditional_means[None, :]) / noise_sigma
    conditional = np.maximum(ndtr(z_upper) - ndtr(z_lower), 0.0)
    conditional /= conditional.sum(axis=0, keepdims=True)
    output_probabilities = conditional @ input_probabilities

    positive_output = output_probabilities > 0.0
    scalar_entropy = -float(np.sum(
        output_probabilities[positive_output]
        * np.log2(output_probabilities[positive_output])
    ))

    ratio = np.divide(
        conditional,
        output_probabilities[:, None],
        out=np.ones_like(conditional),
        where=output_probabilities[:, None] > 0.0,
    )
    positive_joint = conditional > 0.0
    scalar_mutual_information = float(np.sum(
        conditional[positive_joint]
        * np.broadcast_to(input_probabilities[None, :], conditional.shape)[positive_joint]
        * np.log2(ratio[positive_joint])
    ))

    return 2.0 * scalar_entropy, max(2.0 * scalar_mutual_information, 0.0)


def g_bosonic(x: float) -> float:
    if x <= 1e-14:
        return 0.0
    return (x + 1.0) * log2(x + 1.0) - x * log2(x)


def symplectic_eigenvalues(Gamma: np.ndarray) -> np.ndarray:
    n = Gamma.shape[0] // 2
    omega = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    values = np.sort(np.abs(np.linalg.eigvals(1j * omega @ Gamma)))
    return values[::2].real


def evaluate_trusted_key_rate(
    V_A: float,
    T_ch: float,
    xi_excess_input: float,
    eta_B: float,
    xi_rec: float,
    beta: float = 0.95,
    cutoff: int = 60,
    gh_order: int = 32,
    C_tau: float | None = None,
    w_tau: float | None = None,
    mutual_information: float | None = None,
) -> TrustedKeyRateResult:
    if not (0 < T_ch <= 1):
        raise ValueError("T_ch must be in (0,1]")
    if not (0 < eta_B < 1):
        raise ValueError("eta_B must be in (0,1)")
    if xi_excess_input < 0 or xi_rec < 0:
        raise ValueError("xi_excess_input and xi_rec must be non-negative")
    if C_tau is None or w_tau is None:
        C_tau, w_tau = constellation_quantities(V_A, cutoff=cutoff)

    I2 = np.eye(2)
    sigma_z = np.diag([1.0, -1.0])
    V = V_A + 1.0
    # Only pre-detection, input-referred excess noise is included here.
    W_ch = 1.0 + T_ch * (V_A + xi_excess_input)
    Z = 2.0 * np.sqrt(T_ch) * C_tau - np.sqrt(
        max(2.0 * T_ch * xi_excess_input * w_tau, 0.0)
    )
    Gamma_AB1 = np.block([[V * I2, Z * sigma_z], [Z * sigma_z, W_ch * I2]])
    nu12 = symplectic_eigenvalues(Gamma_AB1)

    # xi_rec is the fixed, detector-referred electronic-noise variance per
    # measured quadrature and is purified inside Bob's trusted receiver.
    W_rec = 1.0 + 2.0 * xi_rec / (1.0 - eta_B)
    cross = np.sqrt(max(W_rec ** 2 - 1.0, 0.0)) * sigma_z
    Gamma_FG = np.block([[W_rec * I2, cross], [cross, W_rec * I2]])
    Gamma_0 = np.block([[Gamma_AB1, np.zeros((4, 4))], [np.zeros((4, 4)), Gamma_FG]])

    # Mode order in Gamma_0: A, B1, F', G. Mix B1 and F'.
    S = np.eye(8)
    se, sl = np.sqrt(eta_B), np.sqrt(1.0 - eta_B)
    for q in range(2):
        b, f = 2 + q, 4 + q
        S[b, b], S[b, f] = se, sl
        S[f, b], S[f, f] = -sl, se
    Gamma_after = S @ Gamma_0 @ S.T

    # Reorder to A,F,G,B and condition on heterodyne measurement of B.
    indices = []
    for mode in [0, 2, 3, 1]:
        indices.extend([2 * mode, 2 * mode + 1])
    Gamma = Gamma_after[np.ix_(indices, indices)]
    A, C, B = Gamma[:6, :6], Gamma[:6, 6:], Gamma[6:, 6:]
    Gamma_cond = A - C @ np.linalg.inv(B + I2) @ C.T
    nu345 = symplectic_eigenvalues(Gamma_cond)

    all_nu = np.r_[nu12, nu345]
    if np.min(all_nu) < 1.0 - 2e-6:
        xi_detected = xi_rec + eta_B * T_ch * xi_excess_input / 2.0
        return TrustedKeyRateResult(V_A, T_ch, xi_excess_input, xi_rec, xi_detected,
                                    eta_B, beta, np.nan, np.nan,
                                    -np.inf, C_tau, w_tau, Z,
                                    *[float(v) for v in all_nu])
    nu12 = np.maximum(nu12, 1.0)
    nu345 = np.maximum(nu345, 1.0)

    # Per-quadrature channel: y=sqrt(eta_B*T_ch/2)x+w,
    # Per-quadrature non-vacuum noise at Bob's measurement.
    xi_detected = xi_rec + eta_B * T_ch * xi_excess_input / 2.0
    snr = eta_B * T_ch * V_A / max(
        2.0 * (1.0 + xi_detected), 1e-15
    )
    Ixy = 2.0 * pam8_mutual_information_from_snr(float(snr), gh_order=gh_order) \
        if mutual_information is None else float(mutual_information)
    chi = sum(g_bosonic((v - 1.0) / 2.0) for v in nu12) \
        - sum(g_bosonic((v - 1.0) / 2.0) for v in nu345)
    secret = beta * Ixy - chi
    return TrustedKeyRateResult(
        V_A, T_ch, xi_excess_input, xi_rec, xi_detected, eta_B, beta,
        Ixy, chi, secret, C_tau, w_tau, Z,
        float(nu12[0]), float(nu12[1]), float(nu345[0]), float(nu345[1]), float(nu345[2]),
    )


class FastTrustedKeyRate:
    """Interpolation-assisted evaluator for design sweeps."""
    def __init__(self, va_min=0.5, va_max=12.0, va_step=0.05, cutoff=60,
                 gh_order=32, snr_max=10.0, snr_step=0.005):
        self.va_min, self.va_max = va_min, va_max
        self.cutoff, self.gh_order = cutoff, gh_order
        self.va_grid = np.arange(va_min, va_max + va_step / 2, va_step)
        C, w = np.empty_like(self.va_grid), np.empty_like(self.va_grid)
        for i, va in enumerate(self.va_grid):
            C[i], w[i] = constellation_quantities(float(va), cutoff=cutoff)
        self.C_interp = PchipInterpolator(self.va_grid, C, extrapolate=False)
        self.w_interp = PchipInterpolator(self.va_grid, w, extrapolate=False)
        self.snr_grid = np.arange(0.0, snr_max + snr_step / 2, snr_step)
        mi = np.asarray([2.0 * pam8_mutual_information_from_snr(float(s), gh_order) for s in self.snr_grid])
        self.mi_interp = PchipInterpolator(self.snr_grid, mi, extrapolate=True)

    def evaluate(self, V_A, T_ch, xi_excess_input, eta_B, xi_rec, beta=0.95):
        C = float(self.C_interp(V_A))
        w = float(self.w_interp(V_A))
        xi_detected = xi_rec + eta_B * T_ch * xi_excess_input / 2.0
        snr = eta_B * T_ch * V_A / max(
            2.0 * (1.0 + xi_detected), 1e-15
        )
        Ixy = float(self.mi_interp(snr))
        return evaluate_trusted_key_rate(V_A, T_ch, xi_excess_input, eta_B, xi_rec, beta,
                                         cutoff=self.cutoff, gh_order=self.gh_order,
                                         C_tau=C, w_tau=w, mutual_information=Ixy)

    def optimize(self, T_ch, xi_excess_input, eta_B, xi_rec, beta=0.95):
        def objective(va):
            r = self.evaluate(float(va), T_ch, xi_excess_input, eta_B, xi_rec, beta)
            return -r.secret_fraction if np.isfinite(r.secret_fraction) else 1e6
        opt = minimize_scalar(objective, bounds=(self.va_min, self.va_max), method="bounded",
                              options={"xatol": 2e-6, "maxiter": 150})
        return self.evaluate(float(opt.x), T_ch, xi_excess_input, eta_B, xi_rec, beta)

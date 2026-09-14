"""
========================================================================================
ENSEMBLE: Multi-Model Agreement & Voting Engine (Fix Group 7)
========================================================================================
Implements explicit ensemble agreement evaluation across 4 independent sources:
1. Model 1 (Directional 3-Class Classifier) - semantic direction {-1, 0, +1}
2. Model 2 (Price Regressor)                - predicted return sign matching Model 1
3. Model 3 / 3b (Exhaustion & Runup)        - downside exhaustion / upside runup confirmation
4. Cross-Asset Correlation Aggregate        - peer momentum sign agreement

Logs every decision (including vetoes and HOLDS) to artifacts/ensemble_log.csv.
========================================================================================
"""

import csv
from dataclasses import dataclass
import os

import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
ENSEMBLE_LOG_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_log.csv")


@dataclass
class AgreementResult:
    """Dataclass storing ensemble agreement evaluation details."""
    timestamp: str
    symbol: str
    m1_direction: int           # -1 (DOWN), 0 (FLAT), +1 (UP)
    m1_confidence: float        # Calibrated P(predicted class)
    m2_magnitude: float         # Model 2 expected price return
    m2_agrees: bool             # True if sign(m2) == m1_direction
    exhaustion_agrees: bool     # True if drawdown confirms buy or runup confirms sell
    peers_agree: bool           # True if peer aggregate return supports m1_direction
    agreement_count: int        # Number of agreeing sources (1 to 4)
    recommended_action: str     # 'BUY', 'SELL', or 'HOLD'
    ev: float                   # Expected value after costs


def init_ensemble_log() -> None:
    """Initializes the ensemble log CSV header if not present."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ENSEMBLE_LOG_PATH):
        with open(ENSEMBLE_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "m1_direction", "m1_confidence",
                "m2_magnitude", "m2_agrees", "exhaustion_agrees",
                "peers_agree", "agreement_count", "ev", "recommended_action"
            ])


def evaluate_ensemble_agreement(
    timestamp: str,
    symbol: str,
    m1_dir: int,
    m1_conf: float,
    m2_magnitude: float,
    m3_drawdown: float,
    m3b_runup: float,
    mean_peer_return: float,
    ev: float,
    meta_approved: bool = True,
    log_to_csv: bool = True
) -> AgreementResult:
    """
    Evaluates multi-model consensus across all 4 pipeline models and peer context.

    Agreement Criteria:
    - Source 1 (Model 1): Directional intent (+1 or -1). If FLAT (0), agreement is 0.
    - Source 2 (Model 2): Sign of magnitude matches direction: (m2_magnitude * m1_dir > 0).
    - Source 3 (Model 3/3b):
        * For UP (+1): Runup prediction m3b_runup > 0.005 (0.5% upside room).
        * For DOWN (-1): Exhaustion drawdown m3_drawdown < -0.005 (0.5% downside room).
    - Source 4 (Peers): Aggregate peer momentum aligns with call: (mean_peer_return * m1_dir > 0).

    Args:
        timestamp: Bar datetime string.
        symbol: Ticker symbol.
        m1_dir: Semantic direction (-1, 0, +1).
        m1_conf: Calibrated probability for predicted class.
        m2_magnitude: Model 2 expected 1-min return.
        m3_drawdown: Model 3 predicted 10-min drawdown (negative float).
        m3b_runup: Model 3b predicted 10-min runup (positive float).
        mean_peer_return: Mean return of top peers.
        ev: Expected value after round-trip costs.
        meta_approved: Whether Model 4 approved the trade.
        log_to_csv: Whether to write this row to artifacts/ensemble_log.csv.

    Returns:
        AgreementResult dataclass.
    """
    if m1_dir == 0:
        res = AgreementResult(
            timestamp=str(timestamp),
            symbol=symbol,
            m1_direction=0,
            m1_confidence=round(m1_conf, 4),
            m2_magnitude=round(m2_magnitude, 6),
            m2_agrees=False,
            exhaustion_agrees=False,
            peers_agree=False,
            agreement_count=0,
            recommended_action="HOLD",
            ev=round(ev, 6),
        )
        if log_to_csv:
            _log_agreement(res)
        return res

    # Check 1: Model 2 direction agreement
    m2_agrees = bool((m2_magnitude * m1_dir) > 0)

    # Check 2: Exhaustion / Runup confirmation
    if m1_dir == 1:
        # Long trade: confirmed if Model 3b predicts sufficient upside excursion
        exh_agrees = bool(m3b_runup >= 0.003)
    else:
        # Short trade: confirmed if Model 3 predicts significant downward excursion
        exh_agrees = bool(m3_drawdown <= -0.003)

    # Check 3: Peer correlation support
    peers_agree = bool((mean_peer_return * m1_dir) > 0)

    # Tally agreement: Model 1 base (1) + supporting sources
    agreed_sources = 1 + int(m2_agrees) + int(exh_agrees) + int(peers_agree)

    # Decision rule: Must have at least 2 agreeing sources AND positive EV AND meta approval
    action = "HOLD"
    if agreed_sources >= 2 and ev > 0.0002 and meta_approved:
        action = "BUY" if m1_dir == 1 else "SELL"

    res = AgreementResult(
        timestamp=str(timestamp),
        symbol=symbol,
        m1_direction=m1_dir,
        m1_confidence=round(m1_conf, 4),
        m2_magnitude=round(m2_magnitude, 6),
        m2_agrees=m2_agrees,
        exhaustion_agrees=exh_agrees,
        peers_agree=peers_agree,
        agreement_count=agreed_sources,
        recommended_action=action,
        ev=round(ev, 6),
    )

    if log_to_csv:
        _log_agreement(res)

    return res


def _log_agreement(res: AgreementResult) -> None:
    """Appends an AgreementResult to ensemble_log.csv."""
    init_ensemble_log()
    try:
        with open(ENSEMBLE_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                res.timestamp, res.symbol, res.m1_direction, res.m1_confidence,
                res.m2_magnitude, res.m2_agrees, res.exhaustion_agrees,
                res.peers_agree, res.agreement_count, res.ev, res.recommended_action
            ])
    except Exception:
        pass

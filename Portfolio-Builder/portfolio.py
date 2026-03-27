import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

LEG_COLORS = [
    "#378ADD", "#E24B4A", "#EF9F27", "#A855F7",
    "#EC4899", "#14B8A6", "#F97316", "#6366F1",
]

# ── Pricing layer ────────────────────────────────────────────────────────────

class Underlying:
    label = "Underlying"

    def __init__(self, qty=1):
        self.qty = qty

    def value(self, S, t=0):
        return S * self.qty

    def payoff(self, S):
        return S * self.qty

    def pnl(self, S, S0, t=None):
        return (S - S0) * self.qty

    def __str__(self):
        return f"Underlying  qty={self.qty:+g}"


class ZCB:
    label = "ZCB"

    def __init__(self, r, T, qty=1):
        self.r = r
        self.T = T
        self.qty = qty

    def value(self, S, t=0):
        return np.exp(-self.r * (self.T - t)) * self.qty * np.ones_like(S)

    def payoff(self, S):
        return np.ones_like(S) * self.qty

    def pnl(self, S, P0, t=None):
        t = self.T if t is None else t
        return self.value(S, t) - P0

    def __str__(self):
        return f"ZCB  r={self.r:.2%}  T={self.T}y  qty={self.qty:+g}"


class Option:
    label = "Option"

    def __init__(self, K, T, r, q, sigma, option_type="call", qty=1):
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type
        self.qty = qty

    def _d1(self, S, t):
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * (self.T - t)) / (
                self.sigma * np.sqrt(self.T - t)
            )
        return d1

    def _d2(self, S, t):
        return self._d1(S, t) - self.sigma * np.sqrt(self.T - t)

    def value(self, S, t=0):
        d1 = self._d1(S, t)
        d2 = self._d2(S, t)
        if self.option_type == "call":
            price = S * np.exp(-self.q * (self.T - t)) * norm.cdf(d1) - self.K * np.exp(
                -self.r * (self.T - t)
            ) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(-d2) - S * np.exp(
                -self.q * (self.T - t)
            ) * norm.cdf(-d1)
        return price * self.qty

    def payoff(self, S):
        sign = 1 if self.option_type == "call" else -1
        return np.maximum((S - self.K) * sign, 0) * self.qty

    def pnl(self, S, P0, t=1):
        return self.value(S, t) - P0

    def __str__(self):
        return (
            f"{self.option_type.capitalize()}  K={self.K}  σ={self.sigma:.0%}"
            f"  T={self.T}y  r={self.r:.2%}  qty={self.qty:+g}"
        )


class DigitalOption:
    label = "Digital"

    def __init__(self, K, T, r, q, sigma, qty=1, eps=1e-4):
        self.K = K
        self.T = T
        self.qty = qty
        self.eps = eps
        self._long = Option(K, T, r, q, sigma, "call", 1)
        self._short = Option(K + eps, T, r, q, sigma, "call", -1)

    def value(self, S, t=0):
        return (self._long.value(S, t) + self._short.value(S, t)) / self.eps * self.qty

    def payoff(self, S):
        return (self._long.payoff(S) + self._short.payoff(S)) / self.eps * self.qty

    def pnl(self, S, P0, t=None):
        t = self.T if t is None else t
        return self.value(S, t) - P0

    def __str__(self):
        return (
            f"Digital call  K={self.K}"
            f"  σ={self._long.sigma:.0%}  T={self.T}y  qty={self.qty:+g}"
        )


class Portfolio:
    def __init__(self):
        self.legs = []  # list of (instrument, cost_basis)

    def add(self, instrument):
        S_mid = 100.0  # placeholder spot for cost basis
        try:
            basis = instrument.value(np.array([S_mid]), t=0)[0]
        except Exception:
            basis = 0.0
        self.legs.append((instrument, basis))

    def payoff(self, S):
        out = np.zeros_like(S, dtype=float)
        for inst, _ in self.legs:
            out += inst.payoff(S)
        return out

    def pnl(self, S, t):
        out = np.zeros_like(S, dtype=float)
        for inst, basis in self.legs:
            out += inst.pnl(S, basis, t)
        return out


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("Derivatives portfolio builder")

# Session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()

portfolio: Portfolio = st.session_state.portfolio

# ── Sidebar: instrument builder ───────────────────────────────────────────────
with st.sidebar:
    st.header("Add instrument")

    inst_type = st.selectbox(
        "Instrument type",
        ["Option (call/put)", "Digital call", "Underlying", "ZCB"],
    )

    qty = st.number_input("qty (negative = short)", value=1, step=1)

    # Shared params
    needs_K     = inst_type in ("Option (call/put)", "Digital call")
    needs_sigma = inst_type in ("Option (call/put)", "Digital call")
    needs_T     = inst_type in ("Option (call/put)", "Digital call", "ZCB")
    needs_r     = inst_type in ("Option (call/put)", "Digital call", "ZCB")
    needs_q     = inst_type in ("Option (call/put)", "Digital call")

    if needs_K:
        K = st.number_input("K — strike", value=100.0, step=1.0)
    if needs_sigma:
        sigma = st.slider("σ — volatility", 0.05, 1.0, 0.20, 0.01, format="%.2f")
    if needs_T:
        T = st.number_input("T — maturity (years)", value=1.0, step=0.25, min_value=0.01)
    if needs_r:
        r = st.slider("r — risk-free rate", 0.0, 0.15, 0.03, 0.005, format="%.3f")
    if needs_q:
        q = st.slider("q — dividend yield", 0.0, 0.10, 0.0, 0.005, format="%.3f")
    if inst_type == "Option (call/put)":
        opt_type = st.radio("Option type", ["call", "put"], horizontal=True)

    if st.button("＋ Add to portfolio", use_container_width=True, type="primary"):
        match inst_type:
            case "Option (call/put)":
                inst = Option(K, T, r, q, sigma, opt_type, qty)
            case "Digital call":
                inst = DigitalOption(K, T, r, q, sigma, qty)
            case "Underlying":
                inst = Underlying(qty)
            case "ZCB":
                inst = ZCB(r, T, qty)
        portfolio.add(inst)
        st.rerun()

    # ── Portfolio legs ────────────────────────────────────────────────────────
    if portfolio.legs:
        st.divider()
        st.caption("Current legs")
        to_remove = None
        for i, (inst, _) in enumerate(portfolio.legs):
            col1, col2 = st.columns([5, 1])
            col1.caption(str(inst))
            if col2.button("✕", key=f"rm_{i}"):
                to_remove = i
        if to_remove is not None:
            portfolio.legs.pop(to_remove)
            st.rerun()

        if st.button("Clear all", use_container_width=True):
            portfolio.legs.clear()
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if not portfolio.legs:
    st.info("Add at least one instrument from the sidebar to start.")
    st.stop()

# Controls above chart
col_s, col_t = st.columns([2, 3])
with col_s:
    S0 = st.number_input("S₀ — current spot", value=100.0, step=1.0)
with col_t:
    t_frac = st.slider(
        "t — time elapsed (fraction of T)",
        0.0, 0.999, 0.5, 0.01,
        help="Used for P&L tab only. 0 = now, →1 = near expiry."
    )

# Wide default window; user zooms/pans directly on the Plotly chart
S_range = np.linspace(max(1.0, S0 * 0.3), S0 * 2.5, 2000)

tab_payoff, tab_pnl = st.tabs(["Payoff at expiry", "P&L (t < T)"])

# Helper: plotly figure factory
def make_figure(y_portfolio, y_legs, leg_labels, ylabel, title):
    fig = go.Figure()

    # Individual legs — dashed, semi-transparent
    for i, (y, lbl) in enumerate(zip(y_legs, leg_labels)):
        color = LEG_COLORS[i % len(LEG_COLORS)]
        fig.add_trace(go.Scatter(
            x=S_range, y=y,
            name=lbl,
            mode="lines",
            line=dict(color=color, width=1.4, dash="dash"),
            opacity=0.75,
            hovertemplate=f"<b>{lbl}</b><br>S=%{{x:.2f}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
        ))

    # Portfolio net — solid, prominent
    fig.add_trace(go.Scatter(
        x=S_range, y=y_portfolio,
        name="Portfolio",
        mode="lines",
        line=dict(color="#1D9E75", width=2.6),
        hovertemplate=f"<b>Portfolio</b><br>S=%{{x:.2f}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="#aaaaaa", width=0.8, dash="dot"))

    # Spot marker
    fig.add_vline(
        x=S0,
        line=dict(color="#888780", width=1.2, dash="dash"),
        annotation_text=f"S₀={S0}",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(text=title, font_size=14, x=0, xanchor="left", pad=dict(b=12)),
        xaxis=dict(title="S — underlying price", showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title=ylabel, showgrid=True, gridcolor="#eeeeee", tickformat=".2f"),
        legend=dict(
            orientation="v",
            xanchor="right", x=0.99,
            yanchor="top",   y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#dddddd",
            borderwidth=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        height=430,
        # Preserve zoom/pan state when Streamlit rerenders (e.g. t slider moves)
        uirevision="portfolio",
    )
    return fig

# ── Tab 1: Payoff ─────────────────────────────────────────────────────────────
with tab_payoff:
    y_net = portfolio.payoff(S_range)
    y_legs = [inst.payoff(S_range) for inst, _ in portfolio.legs]
    labels = [str(inst) for inst, _ in portfolio.legs]
    fig = make_figure(y_net, y_legs, labels, "Payoff", "Portfolio payoff at expiry")
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    idx0 = np.searchsorted(S_range, S0)
    mc1.metric("Payoff @ S₀", f"{y_net[idx0]:.4f}")
    mc2.metric("Max payoff", f"{y_net.max():.4f}")
    mc3.metric("Min payoff", f"{y_net.min():.4f}")
    breakeven = S_range[np.where(np.diff(np.sign(y_net)))[0]]
    be_str = "  |  ".join(f"{b:.2f}" for b in breakeven) if len(breakeven) else "—"
    mc4.metric("Break-even(s)", be_str)

# ── Tab 2: P&L ────────────────────────────────────────────────────────────────
with tab_pnl:
    # t as an actual time value: need the minimum T across legs that have one
    T_ref = 1.0  # fallback
    for inst, _ in portfolio.legs:
        if hasattr(inst, "T"):
            T_ref = inst.T
            break
    t_val = t_frac * T_ref

    y_net_pnl = portfolio.pnl(S_range, t_val)
    y_legs_pnl = [inst.pnl(S_range, basis, t_val) for inst, basis in portfolio.legs]
    labels = [str(inst) for inst, _ in portfolio.legs]

    fig2 = make_figure(
        y_net_pnl, y_legs_pnl, labels,
        "P&L",
        f"Portfolio P&L at t={t_val:.3f}y  (t/T = {t_frac:.0%})"
    )
    st.plotly_chart(fig2, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P&L @ S₀", f"{y_net_pnl[idx0]:.4f}")
    mc2.metric("Max P&L", f"{y_net_pnl.max():.4f}")
    mc3.metric("Min P&L", f"{y_net_pnl.min():.4f}")

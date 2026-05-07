import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

# ── Design tokens ─────────────────────────────────────────────────────────────
# Leg colours — vibrant, distinguishable
LEG_COLORS = [
    "#378ADD", "#E24B4A", "#EF9F27", "#A855F7",
    "#EC4899", "#14B8A6", "#F97316", "#6366F1",
]

# Fixed semantic colours for chart elements
C_PORTFOLIO  = "#1D9E75"                  # portfolio net line
C_ZERO       = "#aaaaaa"                  # zero line
C_S0         = "#555555"                  # S₀ vline
C_BARRIER    = "#E24B4A"                  # barrier vline
C_FILL_POS   = "rgba(26,158,117,0.15)"   # greek fill positive
C_FILL_NEG   = "rgba(226,75,74,0.15)"    # greek fill negative
C_GREEK_LINE = "#1982c4"                  # greek curves
BG           = "#FFFFFF"
GRID         = "#eeeeee"

# ── Pricing layer ─────────────────────────────────────────────────────────────

class Instrument:
    def delta(self, S, dS=0.01, t=0):        return np.zeros_like(S, dtype=float)
    def gamma(self, S, dS=0.01, t=0):        return np.zeros_like(S, dtype=float)
    def vega(self, S, dsigma=0.0001, t=0):   return np.zeros_like(S, dtype=float)
    def theta(self, S, dt=1/252, t=0):       return np.zeros_like(S, dtype=float)
    def volga(self, S, dsigma=0.0001, t=0):  return np.zeros_like(S, dtype=float)
    def rho(self, S, dr=0.0001, t=0):        return np.zeros_like(S, dtype=float)


class Underlying(Instrument):
    def __init__(self, qty=1, q=0.0, T=1.0, r=0.0):
        self.qty = qty
        self.q   = q
        self.T   = T
        self.r   = r

    def value(self, S, t=0):
        return S * np.exp(-self.q * (self.T - t)) * self.qty

    def payoff(self, S):
        return S * self.qty

    def pnl(self, S, P0, t=None):
        t = 0 if t is None else t
        return self.value(S, t) - P0

    def delta(self, S, dS=0.01, t=0):
        return np.ones_like(S, dtype=float) * self.qty

    def __str__(self):
        return f"Underlying  q={self.q:.2%}  qty={self.qty:+g}"


class ZCB(Instrument):
    def __init__(self, r, T, qty=1):
        self.r   = r
        self.T   = T
        self.qty = qty

    def value(self, S, t=0):
        return np.exp(-self.r * (self.T - t)) * self.qty * np.ones_like(S, dtype=float)

    def payoff(self, S):
        return np.ones_like(S, dtype=float) * self.qty

    def pnl(self, S, P0, t=None):
        t = self.T if t is None else t
        return self.value(S, t) - P0

    def theta(self, S, dt=1/252, t=0):
        return (ZCB(self.r, self.T - dt, self.qty).value(S, t) - self.value(S, t)) / dt

    def rho(self, S, dr=0.0001, t=0):
        up   = ZCB(self.r + dr, self.T, self.qty).value(S, t)
        down = ZCB(self.r - dr, self.T, self.qty).value(S, t)
        return (up - down) / (2 * dr)

    def __str__(self):
        return f"ZCB  r={self.r:.2%}  T={self.T}y  qty={self.qty:+g}"


class Option(Instrument):
    def __init__(self, K, T, r, q, sigma,
                 option_type="call", qty=1,
                 H=None, knock="out", d=True):
        self.K           = K
        self.T           = T
        self.r           = r
        self.q           = q
        self.sigma       = sigma
        self.option_type = option_type
        self.qty         = qty
        self.H           = H
        self.knock       = knock
        self.d           = d

    def _d1(self, S, t):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log(S / self.K) +
                    (self.r - self.q + 0.5 * self.sigma**2) * (self.T - t)
                    ) / (self.sigma * np.sqrt(self.T - t))

    def _d2(self, S, t):
        return self._d1(S, t) - self.sigma * np.sqrt(self.T - t)

    def _lam(self):
        return (self.r - self.q + 0.5 * self.sigma**2) / self.sigma**2

    def _y(self, S, t):
        l = self._lam()
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log(self.H**2 / (S * self.K)) /
                    (self.sigma * np.sqrt(self.T - t)) +
                    l * self.sigma * np.sqrt(self.T - t))

    def _x1(self, S, t):
        l = self._lam()
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log(S / self.H) /
                    (self.sigma * np.sqrt(self.T - t)) +
                    l * self.sigma * np.sqrt(self.T - t))

    def _y1(self, S, t):
        l = self._lam()
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log(self.H / S) /
                    (self.sigma * np.sqrt(self.T - t)) +
                    l * self.sigma * np.sqrt(self.T - t))

    def vanilla_value(self, S, t=0):
        d1 = self._d1(S, t);  d2 = self._d2(S, t)
        e_q = np.exp(-self.q * (self.T - t))
        e_r = np.exp(-self.r * (self.T - t))
        if self.option_type == "call":
            return S * e_q * norm.cdf(d1) - self.K * e_r * norm.cdf(d2)
        return -S * e_q * norm.cdf(-d1) + self.K * e_r * norm.cdf(-d2)

    def barrier_value(self, S, t=0):
        H, K      = self.H, self.K
        r, q      = self.r, self.q
        sigma, T  = self.sigma, self.T
        knock     = self.knock
        otype     = self.option_type
        l         = self._lam()
        y         = self._y(S, t)
        x1        = self._x1(S, t)
        y1        = self._y1(S, t)
        vanilla   = self.vanilla_value(S, t)
        N         = norm.cdf
        e_q       = np.exp(-q * (T - t))
        e_r       = np.exp(-r * (T - t))
        sq        = np.sqrt(T - t)

        with np.errstate(divide="ignore", invalid="ignore"):
            img = H / S

        down = H < K or (H == K and self.d)

        if down:
            if otype == "call":
                cdi = (S * e_q * img**(2*l) * N(y)
                       - K * e_r * img**(2*l-2) * N(y - sigma*sq))
                cdi = np.where(S < H, vanilla, cdi)
                value = cdi if knock == "in" else vanilla - cdi
            else:
                pdi = (-S * e_q * N(-x1) + K * e_r * N(-x1 + sigma*sq)
                       + S * e_q * img**(2*l) * (N(y) - N(y1))
                       - K * e_r * img**(2*l-2) * (N(y - sigma*sq) - N(y1 - sigma*sq)))
                pdi = np.where(S < H, vanilla, pdi)
                value = pdi if knock == "in" else vanilla - pdi
        else:
            if otype == "call":
                cui = (S * e_q * N(x1) - K * e_r * N(x1 - sigma*sq)
                       - S * e_q * img**(2*l) * (N(-y) - N(-y1))
                       + K * e_r * img**(2*l-2) * (N(-y + sigma*sq) - N(-y1 + sigma*sq)))
                cui = np.where(S >= H, vanilla, cui)
                value = cui if knock == "in" else vanilla - cui
            else:
                pui = (-S * e_q * img**(2*l) * N(-y)
                       + K * e_r * img**(2*l-2) * N(-y + sigma*sq))
                pui = np.where(S >= H, vanilla, pui)
                value = pui if knock == "in" else vanilla - pui

        return value

    def value(self, S, t=0):
        base = self.vanilla_value(S, t) if self.H is None else self.barrier_value(S, t)
        return base * self.qty

    def payoff(self, S):
        vanilla = np.maximum((S - self.K) * (1 if self.option_type == "call" else -1), 0)
        if self.H is None:
            return vanilla * self.qty
        down = self.H < self.K or (self.H == self.K and self.d)
        if self.knock == "out":
            pay = vanilla * (S > self.H) if down else vanilla * (S < self.H)
        else:
            pay = vanilla
        return pay * self.qty

    def pnl(self, S, P0, t=None):
        t = self.T if t is None else t
        return self.value(S, t) - P0

    def _clone(self, **kw):
        p = dict(K=self.K, T=self.T, r=self.r, q=self.q, sigma=self.sigma,
                 option_type=self.option_type, qty=self.qty,
                 H=self.H, knock=self.knock, d=self.d)
        p.update(kw)
        return Option(**p)

    def delta(self, S, dS=0.01, t=0):
        return (self.value(S+dS, t) - self.value(S-dS, t)) / (2*dS)

    def gamma(self, S, dS=0.01, t=0):
        return (self.value(S+dS, t) - 2*self.value(S, t) + self.value(S-dS, t)) / dS**2

    def vega(self, S, dsigma=0.0001, t=0):
        return (self._clone(sigma=self.sigma+dsigma).value(S, t)
                - self._clone(sigma=self.sigma-dsigma).value(S, t)) / (2*dsigma)

    def theta(self, S, dt=1/252, t=0):
        return (self._clone(T=self.T-dt).value(S, t) - self.value(S, t)) / dt

    def volga(self, S, dsigma=0.0001, t=0):
        return (self._clone(sigma=self.sigma+dsigma).value(S, t)
                - 2*self.value(S, t)
                + self._clone(sigma=self.sigma-dsigma).value(S, t)) / dsigma**2

    def rho(self, S, dr=0.0001, t=0):
        return (self._clone(r=self.r+dr).value(S, t)
                - self._clone(r=self.r-dr).value(S, t)) / (2*dr)

    def __str__(self):
        tag = f"{self.option_type.title()}  K={self.K}  σ={self.sigma:.0%}  T={self.T}y  qty={self.qty:+g}"
        if self.H is not None:
            down = self.H < self.K or (self.H == self.K and self.d)
            side = "Down" if down else "Up"
            tag  = f"{side}-and-{self.knock}  " + tag + f"  H={self.H}"
        return tag


class DigitalOption(Instrument):
    def __init__(self, K, T, r, q, sigma, qty=1, eps=1e-4):
        self.K     = K
        self.T     = T
        self.r     = r
        self.q     = q
        self.sigma = sigma
        self.qty   = qty
        self.eps   = eps
        self._lo   = Option(K,       T, r, q, sigma, "call", 1)
        self._hi   = Option(K + eps, T, r, q, sigma, "call", -1)

    def value(self, S, t=0):
        return (self._lo.value(S, t) + self._hi.value(S, t)) / self.eps * self.qty

    def payoff(self, S):
        return (self._lo.payoff(S) + self._hi.payoff(S)) / self.eps * self.qty

    def pnl(self, S, P0, t=None):
        t = self.T if t is None else t
        return self.value(S, t) - P0

    def _clone(self, **kw):
        p = dict(K=self.K, T=self.T, r=self.r, q=self.q,
                 sigma=self.sigma, qty=self.qty, eps=self.eps)
        p.update(kw)
        return DigitalOption(**p)

    def delta(self, S, dS=0.01, t=0):
        return (self.value(S+dS, t) - self.value(S-dS, t)) / (2*dS)

    def gamma(self, S, dS=0.01, t=0):
        return (self.value(S+dS, t) - 2*self.value(S, t) + self.value(S-dS, t)) / dS**2

    def vega(self, S, dsigma=0.0001, t=0):
        return (self._clone(sigma=self.sigma+dsigma).value(S, t)
                - self._clone(sigma=self.sigma-dsigma).value(S, t)) / (2*dsigma)

    def theta(self, S, dt=1/252, t=0):
        return (self._clone(T=self.T-dt).value(S, t) - self.value(S, t)) / dt

    def volga(self, S, dsigma=0.0001, t=0):
        return (self._clone(sigma=self.sigma+dsigma).value(S, t)
                - 2*self.value(S, t)
                + self._clone(sigma=self.sigma-dsigma).value(S, t)) / dsigma**2

    def rho(self, S, dr=0.0001, t=0):
        return (self._clone(r=self.r+dr).value(S, t)
                - self._clone(r=self.r-dr).value(S, t)) / (2*dr)

    def __str__(self):
        return f"Digital call  K={self.K}  σ={self.sigma:.0%}  T={self.T}y  qty={self.qty:+g}"


class Structure:
    def __init__(self):
        self.legs: list[tuple[Instrument, float]] = []

    def add(self, instrument: Instrument, S0: float = 100.0):
        try:
            basis = float(instrument.value(np.array([S0], dtype=float), t=0)[0])
        except Exception:
            basis = 0.0
        self.legs.append((instrument, basis))

    def _agg(self, method, S, **kw):
        out = np.zeros_like(S, dtype=float)
        for inst, _ in self.legs:
            out += getattr(inst, method)(S, **kw)
        return out

    def value(self, S, t=0):  return self._agg("value",  S, t=t)
    def payoff(self, S):      return self._agg("payoff", S)
    def delta(self, S, t=0):  return self._agg("delta",  S, t=t)
    def gamma(self, S, t=0):  return self._agg("gamma",  S, t=t)
    def vega(self, S, t=0):   return self._agg("vega",   S, t=t)
    def theta(self, S, t=0):  return self._agg("theta",  S, t=t)
    def volga(self, S, t=0):  return self._agg("volga",  S, t=t)
    def rho(self, S, t=0):    return self._agg("rho",    S, t=t)

    def pnl(self, S, t):
        out = np.zeros_like(S, dtype=float)
        for inst, basis in self.legs:
            out += inst.pnl(S, basis, t)
        return out


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("Derivatives portfolio builder")

if "structure" not in st.session_state:
    st.session_state.structure = Structure()
if "S0" not in st.session_state:
    st.session_state["S0"] = 100.0

structure: Structure = st.session_state.structure

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Add instrument")

    inst_type = st.selectbox("Instrument type",
        ["Option / Barrier", "Digital call", "Underlying", "ZCB"])

    qty = st.number_input("qty  (negative = short)", value=1, step=1)

    has_K   = inst_type in ("Option / Barrier", "Digital call")
    has_sig = inst_type in ("Option / Barrier", "Digital call")
    has_T   = inst_type in ("Option / Barrier", "Digital call", "ZCB", "Underlying")
    has_r   = inst_type in ("Option / Barrier", "Digital call", "ZCB", "Underlying")
    has_q   = inst_type in ("Option / Barrier", "Digital call", "Underlying")

    if has_K:   K     = st.number_input("K — strike", value=100.0, step=1.0)
    if has_sig: sigma = st.slider("σ — volatility", 0.05, 1.0, 0.20, 0.01, format="%.2f")
    if has_T:   T     = st.number_input("T — maturity (years)", value=1.0, step=0.25, min_value=0.01)
    if has_r:   r     = st.slider("r — risk-free rate", 0.0, 0.15, 0.03, 0.005, format="%.3f")
    if has_q:   q     = st.slider("q — dividend yield", 0.0, 0.10, 0.0, 0.005, format="%.3f")

    if inst_type == "Option / Barrier":
        opt_type = st.radio("Option type", ["call", "put"], horizontal=True)

    # Barrier
    H = None
    if inst_type == "Option / Barrier":
        use_barrier = st.toggle("Add barrier", value=False)
        if use_barrier:
            H     = st.number_input("H — barrier level", value=80.0, step=1.0)
            knock = st.radio("Knock", ["out", "in"], horizontal=True)
            d = True
            if abs(H - K) < 1e-8:
                d = st.radio("Direction (H=K)", ["down", "up"], horizontal=True) == "down"

    if st.button("＋ Add to portfolio", use_container_width=True, type="primary"):
        if inst_type == "Option / Barrier":
            inst = Option(K, T, r, q, sigma, opt_type, qty,
                          H=H, knock=(knock if H is not None else "out"),
                          d=(d if H is not None else True))
        elif inst_type == "Digital call":
            inst = DigitalOption(K, T, r, q, sigma, qty)
        elif inst_type == "Underlying":
            inst = Underlying(qty, q, T, r)
        else:
            inst = ZCB(r, T, qty)
        structure.add(inst, S0=st.session_state.S0)
        st.rerun()

    if structure.legs:
        st.divider()
        st.caption("Current legs")
        to_remove = None
        for i, (inst, _) in enumerate(structure.legs):
            c1, c2 = st.columns([5, 1])
            c1.caption(str(inst))
            if c2.button("✕", key=f"rm_{i}"):
                to_remove = i
        if to_remove is not None:
            structure.legs.pop(to_remove)
            st.rerun()
        if st.button("Clear all", use_container_width=True):
            structure.legs.clear()
            st.rerun()

# ── Main controls ─────────────────────────────────────────────────────────────
if not structure.legs:
    st.info("Add at least one instrument from the sidebar to start.")
    st.stop()

col_s, col_t, col_v = st.columns([2, 3, 1])
with col_s:
    S0 = st.number_input("S₀ — current spot", step=1.0, key="S0")
with col_t:
    t_frac = st.slider("t — time elapsed (fraction of T)",
                       0.0, 0.999, 0.5, 0.01,
                       help="Applies to P&L and Greeks tabs.")

# Dynamic S range
strikes  = [inst.K for inst, _ in structure.legs if hasattr(inst, "K")]
barriers = [inst.H for inst, _ in structure.legs
            if hasattr(inst, "H") and inst.H is not None]
anchors  = strikes + barriers + [S0]
s_hi     = max(anchors) * 1.3
S_range  = np.linspace(1e-6, s_hi, 2000)

T_ref = next((inst.T for inst, _ in structure.legs if hasattr(inst, "T")), 1.0)
t_val = t_frac * T_ref
idx0  = int(np.searchsorted(S_range, S0))

with col_v:
    current_value = float(structure.value(np.array([S0]), t=t_val)[0])
    st.metric("Value @ S₀", f"{current_value:.4f}")

# Barrier vlines reused across tabs
barrier_vlines = [
    (inst.H, f"H={inst.H}", "#E24B4A")
    for inst, _ in structure.legs
    if hasattr(inst, "H") and inst.H is not None
]

tab_payoff, tab_pnl, tab_greeks = st.tabs(["Payoff at expiry", "P&L (t < T)", "Greeks"])
show_legs = st.toggle("Show individual legs", value=True)

# ── Figure factory ────────────────────────────────────────────────────────────
def _base_layout(title, ylabel, height=420, uirev="portfolio"):
    return dict(
        title=dict(text=title, font_size=14, x=0, xanchor="left"),
        xaxis=dict(title="S — underlying price", showgrid=True, gridcolor=GRID),
        yaxis=dict(title=ylabel, showgrid=True, tickformat=".2f", gridcolor=GRID),
        legend=dict(
            orientation="v", xanchor="right", x=0.99, yanchor="top", y=0.99,
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#dddddd", borderwidth=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="x unified",
        plot_bgcolor=BG, paper_bgcolor=BG,
        height=height, uirevision=uirev,
    )

def make_figure(y_portfolio, y_legs, leg_labels, ylabel, title,
                show_legs=True, extra_vlines=None):
    fig = go.Figure()

    if show_legs:
        for i, (y, lbl) in enumerate(zip(y_legs, leg_labels)):
            fig.add_trace(go.Scatter(
                x=S_range, y=y, name=lbl, mode="lines",
                line=dict(color=LEG_COLORS[i % len(LEG_COLORS)], width=1.5, dash="dash"),
                opacity=0.8,
                hovertemplate=f"<b>{lbl}</b><br>S=%{{x:.2f}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=S_range, y=y_portfolio, name="Portfolio", mode="lines",
        line=dict(color=C_PORTFOLIO, width=2.5),
        hovertemplate=f"<b>Portfolio</b><br>S=%{{x:.2f}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=C_ZERO, width=1.0, dash="dot"))
    fig.add_vline(x=S0, line=dict(color=C_S0, width=1.0, dash="dash"),
                  annotation_text=f"S₀={S0}",
                  annotation_font=dict(color=C_S0, size=11),
                  annotation_position="top right")

    for xv, lbl, _ in (extra_vlines or []):
        fig.add_vline(x=xv, line=dict(color=C_BARRIER, width=1.0, dash="dot"),
                      annotation_text=lbl,
                      annotation_font=dict(color=C_BARRIER, size=11),
                      annotation_position="top left")

    fig.update_layout(**_base_layout(title, ylabel))
    return fig


# ── Tab 1: Payoff ─────────────────────────────────────────────────────────────
with tab_payoff:
    y_net  = structure.payoff(S_range)
    y_legs = [inst.payoff(S_range) for inst, _ in structure.legs]
    labels = [str(inst) for inst, _ in structure.legs]
    st.plotly_chart(
        make_figure(y_net, y_legs, labels, "Payoff",
                    "Portfolio payoff at expiry", show_legs, barrier_vlines),
        use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Payoff @ S₀",  f"{y_net[idx0]:.4f}")
    mc2.metric("Max payoff",   f"{y_net.max():.4f}")
    mc3.metric("Min payoff",   f"{y_net.min():.4f}")
    be = S_range[np.where(np.diff(np.sign(y_net)))[0]]
    mc4.metric("Break-even(s)", "  |  ".join(f"{b:.2f}" for b in be) if len(be) else "—")

# ── Tab 2: P&L ────────────────────────────────────────────────────────────────
with tab_pnl:
    y_pnl      = structure.pnl(S_range, t_val)
    y_legs_pnl = [inst.pnl(S_range, basis, t_val) for inst, basis in structure.legs]
    labels     = [str(inst) for inst, _ in structure.legs]
    st.plotly_chart(
        make_figure(y_pnl, y_legs_pnl, labels, "P&L",
                    f"Portfolio P&L  t={t_val:.3f}y  (t/T={t_frac:.0%})",
                    show_legs, barrier_vlines),
        use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P&L @ S₀", f"{y_pnl[idx0]:.4f}")
    mc2.metric("Max P&L",  f"{y_pnl.max():.4f}")
    mc3.metric("Min P&L",  f"{y_pnl.min():.4f}")

# ── Tab 3: Greeks ─────────────────────────────────────────────────────────────
with tab_greeks:
    GREEKS = [
        ("delta", "Δ Delta",  "dV/dS"),
        ("gamma", "Γ Gamma",  "d²V/dS²"),
        ("theta", "Θ Theta",  "dV/dt"),
        ("rho",   "ρ Rho",    "dV/dr"),
        ("vega",  "ν Vega",   "dV/dσ"),
        ("volga", "Volga",    "d²V/dσ²"),
    ]

    greek_values = {k: getattr(structure, k)(S_range, t=t_val) for k, *_ in GREEKS}

    fig_g = make_subplots(
        rows=3, cols=2,
        subplot_titles=[label for _, label, _ in GREEKS],
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    for idx, (key, label, ylabel) in enumerate(GREEKS):
        row = idx // 2 + 1
        col = idx %  2 + 1
        y   = greek_values[key]

        # Shaded fill above / below zero
        fig_g.add_trace(go.Scatter(
            x=S_range, y=np.where(y >= 0, y, 0),
            fill="tozeroy", mode="none",
            fillcolor=C_FILL_POS, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig_g.add_trace(go.Scatter(
            x=S_range, y=np.where(y < 0, y, 0),
            fill="tozeroy", mode="none",
            fillcolor=C_FILL_NEG, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        # Main line
        fig_g.add_trace(go.Scatter(
            x=S_range, y=y, mode="lines", name=label,
            line=dict(color=C_GREEK_LINE, width=1.8),
            hovertemplate=f"<b>{label}</b><br>S=%{{x:.2f}}<br>%{{y:.5f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

        # S₀ and barrier vlines
        fig_g.add_vline(x=S0, line=dict(color=C_S0, width=0.8, dash="dash"),
                        row=row, col=col)
        for xv, _, _ in barrier_vlines:
            fig_g.add_vline(x=xv, line=dict(color=C_BARRIER, width=0.8, dash="dot"),
                            row=row, col=col)

    fig_g.update_layout(
        height=700,
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=50, r=20, t=60, b=40),
        uirevision="greeks",
    )
    fig_g.update_xaxes(showgrid=True, gridcolor=GRID, title_text="S")
    fig_g.update_yaxes(showgrid=True, gridcolor=GRID, tickformat=".3f")

    st.plotly_chart(fig_g, use_container_width=True)

    # Point values at S₀
    st.caption(f"Greeks at S₀={S0},  t={t_val:.3f}y")
    gcols = st.columns(6)
    for i, (key, label, _) in enumerate(GREEKS):
        gcols[i].metric(label, f"{greek_values[key][idx0]:.5f}")

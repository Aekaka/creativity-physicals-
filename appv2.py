# football_creativity_dashboard_v3.py
# ----------------------------------------------------------
# AC Horsens Creativity + Physical Index (Humanised, Full App)
# - Multi-match Opta event upload (JSON)
# - Creativity per-90 (0–10)
# - Physical contribution per-90 (0–10) from CSV or detected columns
# - In-app sliders to blend creativity vs physical + physical sub-weights
# - Radar charts, xT heatmaps, passing network (arrows optional), shot map, assist chains
# - Player profile + CSV exports
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
try:
    from mplsoccer import Pitch
except Exception:
    Pitch = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, mean_squared_error
except Exception:
    LogisticRegression = None
    GradientBoostingRegressor = None
    GradientBoostingClassifier = None
    train_test_split = None
    roc_auc_score = None
    mean_squared_error = None

# --------------------------
# App setup
# --------------------------
st.set_page_config(layout="wide")
st.title("🌟 AC Horsens — Creativity + Physical Contribution Dashboard")
st.caption("Human-friendly analytics: blend chance creation (xT) with physical output, all per-90 for fair comparisons.")

# --------------------------
# Sidebar — explanations & upload
# --------------------------
st.sidebar.header("📂 Upload & Options")

st.sidebar.markdown(
    """
    **Quick metric guide**  
    - **xT (Expected Threat)**: how much an action increases scoring probability given pitch location.  
    - **Creativity Score (0–10)**: weighted mix of xT, progressive passes, and involvement — **per-90**.  
    - **Physical Score (0–10)**: per-90 distance, high-speed running, sprint distance, high-intensity distance.  
    - **Overall Score (0–10)**: your in-house blend of creativity and physical, set by the sliders below.
    """
)

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more Opta event JSON files",
    type="json",
    accept_multiple_files=True
)

team_filter_option = st.sidebar.radio("Players to analyse", ["All players", "AC Horsens only"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Physical data (optional)")
phys_csv = st.sidebar.file_uploader(
    "Upload player physicals CSV (any of: total_distance, hsr, sprint_distance, high_intensity)",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tracking data (optional)")
tracking_file = st.sidebar.file_uploader(
    "Upload tracking JSON (ndjson accepted)",
    type=["json"]
)
learned_epv_file = st.sidebar.file_uploader(
    "Upload learned EPV surface (npy/csv/json)",
    type=["npy", "csv", "json"]
)
rr_classifier_file = st.sidebar.file_uploader(
    "Upload RR classifier bins (CSV with teamId,len_bin,ang_bin,comp_prob)",
    type=["csv"]
)
rr_model_file = st.sidebar.file_uploader(
    "Upload RR classifier model (pickle: predict_proba)",
    type=["pkl", "pickle"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Weighting (adjust live)")
physical_weight = st.sidebar.slider("Physical weight (0 = only creativity, 1 = only physical)", 0.0, 1.0, 0.30, 0.05)
creativity_weight = 1.0 - physical_weight
st.sidebar.write(f"Creativity weight = **{creativity_weight:.2f}**")

st.sidebar.markdown("#### Physical component weights (auto-normalised)")
pd_w_dist = st.sidebar.slider("Total distance weight", 0.0, 1.0, 0.25)
pd_w_hsr = st.sidebar.slider("HSR weight", 0.0, 1.0, 0.25)
pd_w_sprint = st.sidebar.slider("Sprint distance weight", 0.0, 1.0, 0.25)
pd_w_hid = st.sidebar.slider("High-intensity distance weight", 0.0, 1.0, 0.25)

# normalise physical component sliders so they sum to 1
phys_comp_sum = pd_w_dist + pd_w_hsr + pd_w_sprint + pd_w_hid
if phys_comp_sum == 0:
    pd_w_dist = pd_w_hsr = pd_w_sprint = pd_w_hid = 0.25
    phys_comp_sum = 1.0
pd_w_dist /= phys_comp_sum
pd_w_hsr /= phys_comp_sum
pd_w_sprint /= phys_comp_sum
pd_w_hid /= phys_comp_sum

st.sidebar.markdown("---")
st.sidebar.markdown("### Data quality guardrails")
min_minutes = st.sidebar.slider("Minimum minutes to include a player", 0, 120, 30)
show_arrows = st.sidebar.checkbox("Show arrowheads on passing network (heavier)", value=False)
max_arrows = st.sidebar.slider("Max arrows to draw (performance)", 50, 1000, 250, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("### Game Intelligence (proxies)")
st.sidebar.caption(
    "Final third entries, box entries, switches, through-ball proxies, zone 14 and one-touch shot setups per-90."
)
gi_w_final = st.sidebar.slider("GI weight — Final third entries", 0.0, 1.0, 1.0, 0.05)
gi_w_box = st.sidebar.slider("GI weight — Box entries", 0.0, 1.0, 1.0, 0.05)
gi_w_switch = st.sidebar.slider("GI weight — Switches", 0.0, 1.0, 1.0, 0.05)
gi_w_through = st.sidebar.slider("GI weight — Through-pass proxy", 0.0, 1.0, 1.0, 0.05)
gi_w_zone14 = st.sidebar.slider("GI weight — Zone 14 entries", 0.0, 1.0, 1.0, 0.05)
gi_w_one_touch = st.sidebar.slider("GI weight — One-touch shot setups", 0.0, 1.0, 1.0, 0.05)

# normalise GI weights
gi_sum = gi_w_final + gi_w_box + gi_w_switch + gi_w_through + gi_w_zone14 + gi_w_one_touch
if gi_sum == 0:
    gi_w_final = gi_w_box = gi_w_switch = gi_w_through = gi_w_zone14 = gi_w_one_touch = 1.0
    gi_sum = 6.0
gi_w_final /= gi_sum
gi_w_box /= gi_sum
gi_w_switch /= gi_sum
gi_w_through /= gi_sum
gi_w_zone14 /= gi_sum
gi_w_one_touch /= gi_sum

st.sidebar.markdown("#### EPV weighting (optional)")
use_epv = st.sidebar.checkbox("Use EPV weighting for GI metrics", value=False)
epv_beta_x = st.sidebar.slider("EPV: forward importance (beta_x)", 0.01, 0.25, 0.08, 0.01)
epv_center_pref = st.sidebar.slider("EPV: central channel preference", 0.0, 2.0, 0.8, 0.1)

st.sidebar.markdown("#### EPV surface from tracking (experimental)")
build_epv_from_tracking = st.sidebar.checkbox("Build EPV surface from tracking", value=False)
epv_grid_x = st.sidebar.slider("EPV grid X bins", 10, 100, 50, 5)
epv_grid_y = st.sidebar.slider("EPV grid Y bins", 8, 68, 34, 2)
epv_use_density = st.sidebar.checkbox("Include player density term", value=True)
epv_attack_density_w = st.sidebar.slider("Attack density weight", 0.0, 2.0, 0.3, 0.05)
epv_defense_density_w = st.sidebar.slider("Defense density weight", 0.0, 2.0, 0.3, 0.05)
epv_flip_x = st.sidebar.checkbox("Flip EPV x-axis (if teams attack opposite)", value=False)
epv_auto_flip = st.sidebar.checkbox("Auto flip by period (home perspective)", value=False)
epv_first_half_dir = st.sidebar.selectbox("1st-half home attack direction", ["left-to-right", "right-to-left"], index=0)

st.sidebar.markdown("#### Risk–Reward and Tracking mapping")
use_rr = st.sidebar.checkbox("Compute pass Risk–Reward (EPV)", value=False)
rr_risk_lambda = st.sidebar.slider("Risk weight (lambda)", 0.0, 2.0, 0.6, 0.05)
track_pitch_len = st.sidebar.number_input("Tracking pitch length (m)", 90.0, 120.0, 105.0, 0.5)
track_pitch_wid = st.sidebar.number_input("Tracking pitch width (m)", 60.0, 80.0, 68.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Train in-app models")
train_epv_btn = st.sidebar.button("Train learned EPV surface (sklearn)")
train_rr_btn = st.sidebar.button("Train pass completion classifier (sklearn)")
use_learned_epv = st.sidebar.checkbox("Use learned EPV surface after training", value=False)
use_learned_rr = st.sidebar.checkbox("Use learned RR classifier after training", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Technical weights (adjust granular)")
tw_pass_acc = st.sidebar.slider("Pass accuracy weight", 0.0, 1.0, 0.25)
tw_prog = st.sidebar.slider("Progressive actions weight", 0.0, 1.0, 0.25)
tw_shot = st.sidebar.slider("Shooting accuracy weight", 0.0, 1.0, 0.25)
tw_cross = st.sidebar.slider("Crossing activity weight", 0.0, 1.0, 0.25)
tw_sum = tw_pass_acc + tw_prog + tw_shot + tw_cross
if tw_sum == 0:
    tw_pass_acc = tw_prog = tw_shot = tw_cross = 0.25
    tw_sum = 1.0
tw_pass_acc /= tw_sum; tw_prog /= tw_sum; tw_shot /= tw_sum; tw_cross /= tw_sum

st.sidebar.markdown("### Tactical weights (adjust granular)")
tac_final = st.sidebar.slider("Final third/box recognition", 0.0, 1.0, 0.4)
tac_switch = st.sidebar.slider("Switching play", 0.0, 1.0, 0.2)
tac_zone14 = st.sidebar.slider("Zone 14 usage", 0.0, 1.0, 0.2)
tac_link = st.sidebar.slider("Shot chain link-ups", 0.0, 1.0, 0.2)
tac_sum = tac_final + tac_switch + tac_zone14 + tac_link
if tac_sum == 0:
    tac_final = 0.4; tac_switch = tac_zone14 = tac_link = 0.2
    tac_sum = 1.0
tac_final /= tac_sum; tac_switch /= tac_sum; tac_zone14 /= tac_sum; tac_link /= tac_sum

# --------------------------
# Helpers: parse & load
# --------------------------
@st.cache_data
def load_json_file(f):
    return json.load(f)

def parse_events(data):
    """
    Convert a single Opta event JSON (match) into a tidy DataFrame.
    Adds helpful columns: playerName, minute, xT, endX, endY, teamId, teamName.
    """
    events = data.get("liveData", {}).get("event", [])
    if not events:
        return pd.DataFrame()

    df = pd.json_normalize(events)

    # try build a team id -> team name map from matchInfo
    team_map = {}
    for c in data.get("matchInfo", {}).get("contestant", []):
        cid = c.get("id")
        name = c.get("name") or c.get("officialName") or c.get("shortName")
        if cid:
            team_map[cid] = name

    # Player name + minute stamp
    if "playerName" not in df.columns:
        df["playerName"] = "Unknown"
    else:
        df["playerName"] = df["playerName"].fillna("Unknown")

    df["minute"] = df.get("timeMin", 0).astype(float) + df.get("timeSec", 0).astype(float) / 60.0

    # Helper to safely pull qualifier values
    def get_qualifier_value(q, qid):
        if isinstance(q, list):
            for item in q:
                if isinstance(item, dict) and item.get("qualifierId") == qid:
                    val = item.get("value", 0)
                    try:
                        return float(val)
                    except Exception:
                        return 0.0
        return 0.0

    # Extract expected threat + pass end coords
    df["xT"] = df["qualifier"].apply(lambda q: get_qualifier_value(q, 318))
    df["endX"] = df["qualifier"].apply(lambda q: get_qualifier_value(q, 140))
    df["endY"] = df["qualifier"].apply(lambda q: get_qualifier_value(q, 141))

    # team id/name mapping
    if "contestantId" in df.columns:
        df["teamId"] = df["contestantId"]
        df["teamName"] = df["teamId"].map(team_map).fillna(df["teamId"])
    else:
        df["teamId"] = df.get("contestantId", None)
        df["teamName"] = df["teamId"].map(team_map).fillna("Unknown")

    # ensure expected base columns exist
    for col in ["id", "eventId", "typeId", "outcome", "x", "y"]:
        if col not in df.columns:
            df[col] = np.nan

    return df

@st.cache_data
def load_multiple_files(file_list):
    dfs = []
    for f in file_list:
        try:
            data = load_json_file(f)
            match_df = parse_events(data)
            if not match_df.empty:
                dfs.append(match_df)
        except Exception as e:
            st.warning(f"Could not parse a file: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def read_tracking_json(uploaded):
    try:
        uploaded.seek(0)
        raw = uploaded.read()
        # Support both JSON array and ndjson
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                frames = [data]
            elif isinstance(data, list):
                frames = data
            else:
                frames = []
        except Exception:
            # try ndjson
            lines = raw.decode("utf-8").strip().splitlines()
            frames = [json.loads(l) for l in lines if l.strip()]
        return frames
    except Exception as e:
        st.warning(f"Could not parse tracking file: {e}")
        return []

def load_learned_epv_surface(file_handle):
    try:
        name = file_handle.name.lower()
        file_handle.seek(0)
        if name.endswith('.npy'):
            arr = np.load(file_handle)
            # assume arr shape [y_bins, x_bins]
            y_bins, x_bins = arr.shape
            xs = np.linspace(0, 100, x_bins)
            ys = np.linspace(0, 100, y_bins)
            X, Y = np.meshgrid(xs, ys)
            return X, Y, arr
        elif name.endswith('.csv'):
            df = pd.read_csv(file_handle)
            # expect columns x,y,value or wide grid
            if {'x','y','value'}.issubset(set(df.columns)):
                pv = df.pivot(index='y', columns='x', values='value').sort_index().sort_index(axis=1)
                xs = pv.columns.to_numpy()
                ys = pv.index.to_numpy()
                X, Y = np.meshgrid(xs, ys)
                return X, Y, pv.to_numpy()
            else:
                arr = df.to_numpy()
                y_bins, x_bins = arr.shape
                xs = np.linspace(0, 100, x_bins)
                ys = np.linspace(0, 100, y_bins)
                X, Y = np.meshgrid(xs, ys)
                return X, Y, arr
        elif name.endswith('.json'):
            data = json.load(file_handle)
            arr = np.array(data)
            y_bins, x_bins = arr.shape
            xs = np.linspace(0, 100, x_bins)
            ys = np.linspace(0, 100, y_bins)
            X, Y = np.meshgrid(xs, ys)
            return X, Y, arr
    except Exception as e:
        st.warning(f"Could not load learned EPV surface: {e}")
    return None

def build_epv_surface_from_tracking(frames, x_bins=50, y_bins=34, use_density=True, attack_w=0.3, defense_w=0.3, flip_x=False, auto_flip=False, first_half_dir="left-to-right"):
    """
    Construct a simple EPV-like surface from tracking frames.
    Heuristic: combine field tilt (closer to goal more value) + central preference and optional density contrasts.
    Returns: (X_grid, Y_grid, EPV_grid) on 0..100 pitch scale.
    """
    if not frames:
        return None

    # grids
    xs = np.linspace(0, 100, x_bins)
    ys = np.linspace(0, 100, y_bins)
    X, Y = np.meshgrid(xs, ys)

    # base EPV from location only (forward + central)
    beta_x = 0.08
    center_pref = 0.8
    base = 1.0 / (1.0 + np.exp(-beta_x * ((100 - X) if flip_x else (X - 70))))
    center = np.exp(-((Y - 50.0) ** 2) / (2 * (15.0 ** 2))) ** center_pref
    epv = base * center

    if use_density:
        # crude density maps from first N frames
        n_use = min(len(frames), 200)
        atk_density = np.zeros_like(epv, dtype=float)
        def_density = np.zeros_like(epv, dtype=float)

        for i in range(n_use):
            fr = frames[i]
            period = fr.get("period", 1)
            # Determine flip per frame if auto_flip enabled (home perspective)
            flip_frame = False
            if auto_flip:
                if first_half_dir == "left-to-right":
                    # 1st half attack to x+, 2nd half to x- => flip in 2nd half
                    flip_frame = (period >= 2)
                else:
                    # 1st half attack to x-, 2nd half to x+ => flip in 1st half
                    flip_frame = (period == 1)
            home = fr.get("homePlayers", [])
            away = fr.get("awayPlayers", [])
            # assume xyz scaled around [-? .. ?], map to 0..100 using minmax of plausible bounds
            # We treat x as longitudinal, y as lateral; adjust if needed.
            def norm(p):
                x, y = p[0], p[1]
                # map roughly from [-50, 50] to [0, 100]
                nx = (x + 50) * 1.0
                ny = (y + 50) * 1.0
                nx = np.clip(nx, 0, 100)
                ny = np.clip(ny, 0, 100)
                if flip_frame:
                    nx = 100 - nx
                return nx, ny

            for pl in home:
                nx, ny = norm(pl.get("xyz", [0, 0, 0]))
                ix = np.searchsorted(xs, nx) - 1
                iy = np.searchsorted(ys, ny) - 1
                if 0 <= ix < x_bins and 0 <= iy < y_bins:
                    atk_density[iy, ix] += 1.0
            for pl in away:
                nx, ny = norm(pl.get("xyz", [0, 0, 0]))
                ix = np.searchsorted(xs, nx) - 1
                iy = np.searchsorted(ys, ny) - 1
                if 0 <= ix < x_bins and 0 <= iy < y_bins:
                    def_density[iy, ix] += 1.0

        if atk_density.max() > 0:
            atk_density /= atk_density.max()
        if def_density.max() > 0:
            def_density /= def_density.max()

        epv = np.clip(epv + attack_w * atk_density - defense_w * def_density, 0.0, 1.0)

    return X, Y, epv

def bilinear_sample(grid, xs, ys, grid_xs, grid_ys):
    """
    Bilinear sample from grid at points (xs, ys) where xs, ys in [0,100].
    grid is shaped [y_bins, x_bins], with coordinates grid_xs (1D), grid_ys (1D).
    Returns array of sampled values.
    """
    x_idx = np.searchsorted(grid_xs, xs) - 1
    y_idx = np.searchsorted(grid_ys, ys) - 1
    x_idx = np.clip(x_idx, 0, len(grid_xs) - 2)
    y_idx = np.clip(y_idx, 0, len(grid_ys) - 2)

    x0 = grid_xs[x_idx]
    x1 = grid_xs[x_idx + 1]
    y0 = grid_ys[y_idx]
    y1 = grid_ys[y_idx + 1]
    # avoid divide by zero
    wx = np.divide(xs - x0, (x1 - x0), out=np.zeros_like(xs), where=(x1 - x0) != 0)
    wy = np.divide(ys - y0, (y1 - y0), out=np.zeros_like(ys), where=(y1 - y0) != 0)

    v00 = grid[y_idx, x_idx]
    v10 = grid[y_idx, x_idx + 1]
    v01 = grid[y_idx + 1, x_idx]
    v11 = grid[y_idx + 1, x_idx + 1]

    v0 = v00 * (1 - wx) + v10 * wx
    v1 = v01 * (1 - wx) + v11 * wx
    v = v0 * (1 - wy) + v1 * wy
    return v

@st.cache_resource(show_spinner=False)
def train_epv_model_from_data(frames, events_df, x_bins=50, y_bins=34, horizon_s=10.0, auto_flip=False, first_half_dir="left-to-right"):
    """
    Train a lightweight learned EPV surface using grid labels = probability of shot within horizon while keeping possession.
    Features: just location (x,y). Model: GradientBoostingRegressor over (x,y) → EPV.
    """
    if frames is None or len(frames) == 0 or events_df is None or events_df.empty:
        return None
    if GradientBoostingRegressor is None:
        st.warning("scikit-learn not available.")
        return None

    # Build labels per grid cell using events only: mark cells where subsequent shot occurred soon after possession moves there.
    # Simplification: use end locations of successful passes as states; label 1 if a shot by same team occurs within horizon in event order.
    ev = events_df.sort_values("eventId")
    ev = ev[(ev.get("typeId") == 1) & (ev.get("outcome") == 1)].copy()
    if ev.empty:
        return None
    shots = events_df[events_df.get("typeId") == 13][["eventId", "teamId"]].sort_values("eventId")
    shot_ids = shots["eventId"].to_numpy()

    ev["label"] = 0.0
    ev_ids = ev["eventId"].to_numpy()
    ev_team = ev["teamId"].to_numpy()
    for i in range(len(ev)):
        eid = ev_ids[i]
        tid = ev_team[i]
        # find next shot id and team
        idx = np.searchsorted(shot_ids, eid, side='right')
        if idx < len(shots):
            shot_eid = shots.iloc[idx]["eventId"]
            shot_tid = shots.iloc[idx]["teamId"]
            if shot_tid == tid and (shot_eid - eid) <= 50:  # proxy for short horizon in event steps
                ev.iat[i, ev.columns.get_loc("label")] = 1.0

    X = ev[["endX", "endY"]].fillna(0).to_numpy()
    y = ev["label"].to_numpy()
    model = GradientBoostingRegressor(random_state=42)
    try:
        model.fit(X, y)
    except Exception:
        return None

    xs = np.linspace(0, 100, x_bins)
    ys = np.linspace(0, 100, y_bins)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    pred = model.predict(grid_pts)
    V = pred.reshape(Yg.shape)
    V = np.clip(V, 0.0, 1.0)
    return Xg, Yg, V

@st.cache_resource(show_spinner=False)
def train_rr_model_from_data(events_df):
    if events_df is None or events_df.empty:
        return None
    if LogisticRegression is None:
        st.warning("scikit-learn not available.")
        return None
    passes = events_df[events_df.get("typeId") == 1].copy()
    passes = passes.dropna(subset=["x", "y", "endX", "endY", "outcome"]) 
    if passes.empty:
        return None
    dx = passes["endX"].to_numpy() - passes["x"].to_numpy()
    dy = passes["endY"].to_numpy() - passes["y"].to_numpy()
    length = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.degrees(np.arctan2(dy, dx))
    start_zone14 = ((passes["x"] >= 78) & (passes["x"] <= 88) & (passes["y"] >= 42) & (passes["y"] <= 58)).astype(int)
    end_zone14 = ((passes["endX"] >= 78) & (passes["endX"] <= 88) & (passes["endY"] >= 42) & (passes["endY"] <= 58)).astype(int)
    X = np.column_stack([length, angle, start_zone14, end_zone14])
    y = (passes["outcome"] == 1).astype(int).to_numpy()
    model = GradientBoostingClassifier(random_state=42)
    try:
        model.fit(X, y)
    except Exception:
        return None
    return model
def compute_pass_risk_reward(df, epv_surface=None, lambda_risk=0.6, rr_bins=None):
    """
    Compute per-pass reward = EPV(end) - EPV(start), and risk proxy based on interception/turnover likelihood.
    Risk proxy (simple): inverse of pass completion probability approximation = 1 - completion_prob.
    Here we approximate completion_prob by average team success rate for similar pass length and angle buckets.
    Returns per-player aggregates per-90.
    """
    if df.empty:
        return pd.DataFrame()

    passes = df[(df.get("typeId") == 1)].copy()
    for col in ["x", "y", "endX", "endY", "outcome"]:
        if col not in passes.columns:
            passes[col] = np.nan
    passes = passes.dropna(subset=["x", "y", "endX", "endY"]).copy()

    # EPV sampling
    if epv_surface is not None:
        Xg, Yg, Vg = epv_surface
        gx = Xg[0, :]
        gy = Yg[:, 0]
        start_epv = bilinear_sample(Vg, passes["x"].to_numpy(), passes["y"].to_numpy(), gx, gy)
        end_epv = bilinear_sample(Vg, passes["endX"].to_numpy(), passes["endY"].to_numpy(), gx, gy)
    else:
        # fallback: analytic EPV
        def f(x, y):
            return 1.0 / (1.0 + np.exp(-0.08 * (x - 70))) * np.exp(-((y - 50.0) ** 2) / (2 * (15.0 ** 2)))
        start_epv = passes.apply(lambda r: f(r["x"], r["y"]), axis=1).to_numpy()
        end_epv = passes.apply(lambda r: f(r["endX"], r["endY"]), axis=1).to_numpy()

    passes["epv_reward"] = end_epv - start_epv

    # Buckets for pass difficulty (length and angle)
    dx = passes["endX"].to_numpy() - passes["x"].to_numpy()
    dy = passes["endY"].to_numpy() - passes["y"].to_numpy()
    length = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.degrees(np.arctan2(dy, dx))
    length_bin = np.digitize(length, bins=np.linspace(0, 100, 8))
    angle_bin = np.digitize(angle, bins=np.linspace(-180, 180, 12))
    passes["len_bin"] = length_bin
    passes["ang_bin"] = angle_bin

    # Completion by team and bins
    if rr_bins is not None and not rr_bins.empty:
        comp = rr_bins.copy()
    else:
        comp = passes.assign(success=(passes["outcome"] == 1).astype(float)).groupby(["teamId", "len_bin", "ang_bin"])\
            ["success"].mean().rename("comp_prob").reset_index()
    passes = passes.merge(comp, on=["teamId", "len_bin", "ang_bin"], how="left")
    passes["comp_prob"] = passes["comp_prob"].fillna(passes["outcome"].mean())
    passes["risk"] = 1.0 - passes["comp_prob"]
    passes["rr_value"] = passes["epv_reward"] - lambda_risk * passes["risk"]

    # Per-player per-90
    minutes_played = df.groupby("playerName")["minute"].agg(lambda x: max(x) - min(x) + 1).rename("minutes_played")
    agg = passes.groupby("playerName").agg(
        epv_reward_sum=("epv_reward", "sum"),
        risk_sum=("risk", "sum"),
        rr_sum=("rr_value", "sum"),
        pass_count=("id", "count")
    ).reset_index()
    out = agg.merge(minutes_played.reset_index(), on="playerName", how="left")
    out["minutes_played"] = out["minutes_played"].replace({0: 1}).fillna(1)
    for c in ["epv_reward_sum", "risk_sum", "rr_sum", "pass_count"]:
        out[f"{c}_per90"] = (out[c] / out["minutes_played"]) * 90
    return out.sort_values("rr_sum_per90", ascending=False).reset_index(drop=True)

def compute_offball_epv(frames, surface, pitch_len_m=105.0, pitch_wid_m=68.0, auto_flip=False, first_half_dir="left-to-right"):
    """
    Off-ball EPV proxy: for each frame, sample EPV at each player's location; average per player.
    Assumes `frames` tracking with Opta-like xyz in meters relative to pitch center; maps to 0..100 pitch.
    """
    if not frames or surface is None:
        return pd.DataFrame()
    Xg, Yg, Vg = surface
    gx = Xg[0, :]
    gy = Yg[:, 0]

    # accumulate per player
    values = {}
    counts = {}

    def to_pitch01(x_m, y_m):
        # map meters to 0..100 assuming center (0,0)
        x01 = (x_m / (pitch_len_m / 2.0)) * 50 + 50
        y01 = (y_m / (pitch_wid_m / 2.0)) * 50 + 50
        return float(np.clip(x01, 0, 100)), float(np.clip(y01, 0, 100))

    for fr in frames:
        period = fr.get("period", 1)
        flip_frame = False
        if auto_flip:
            if first_half_dir == "left-to-right":
                flip_frame = (period >= 2)
            else:
                flip_frame = (period == 1)
        for side in ["homePlayers", "awayPlayers"]:
            for pl in fr.get(side, []):
                pid = pl.get("optaId") or pl.get("playerId")
                x_m, y_m = pl.get("xyz", [0, 0, 0])[0], pl.get("xyz", [0, 0, 0])[1]
                x01, y01 = to_pitch01(x_m, y_m)
                if flip_frame:
                    x01 = 100 - x01
                v = bilinear_sample(Vg, np.array([x01]), np.array([y01]), gx, gy)[0]
                values[pid] = values.get(pid, 0.0) + float(v)
                counts[pid] = counts.get(pid, 0) + 1

    rows = []
    for pid, val in values.items():
        cnt = counts.get(pid, 1)
        rows.append({"playerId": pid, "offball_epv": val / cnt})
    return pd.DataFrame(rows).sort_values("offball_epv", ascending=False).reset_index(drop=True)

# --------------------------
# Core: compute creativity (per-90 adjusted)
# --------------------------
def compute_creativity_metrics(df):
    if df.empty:
        return pd.DataFrame()

    # successful passes only
    passes = df[(df["typeId"] == 1) & (df["outcome"] == 1)].copy()

    # minutes played per player (rough proxy from first to last event minute)
    minutes_played = df.groupby("playerName")["minute"].agg(lambda x: max(x) - min(x) + 1).rename("minutes_played")

    creativity = passes.groupby("playerName").agg(
        total_passes=("id", "count"),
        xT_sum=("xT", "sum"),
        progressive_passes=("x", lambda s: (s.diff() > 10).sum()),  # ≥10 units closer to goal
        avg_xT=("xT", "mean"),
    ).reset_index()

    creativity = creativity.merge(minutes_played.reset_index(), on="playerName", how="left")
    creativity["minutes_played"] = creativity["minutes_played"].replace({0: 1}).fillna(1)

    # per-90 conversions for totals
    for col in ["xT_sum", "progressive_passes", "total_passes"]:
        creativity[f"{col}_per90"] = (creativity[col] / creativity["minutes_played"]) * 90

    # normalise metrics to 0..1 (top performer = 1)
    norm_cols = ["xT_sum_per90", "avg_xT", "progressive_passes_per90", "total_passes_per90"]
    for col in norm_cols:
        if col in creativity.columns:
            max_val = creativity[col].max()
            creativity[f"{col}_norm"] = creativity[col] / max_val if max_val and max_val > 0 else 0.0
        else:
            creativity[f"{col}_norm"] = 0.0

    # creativity score on 0–10
    creativity["creativity_score"] = (
        creativity["xT_sum_per90_norm"] * 0.4
        + creativity["avg_xT_norm"] * 0.2
        + creativity["progressive_passes_per90_norm"] * 0.2
        + creativity["total_passes_per90_norm"] * 0.2
    ) * 10

    # drop very short cameos if needed
    creativity = creativity[creativity["minutes_played"] >= min_minutes]

    return creativity.sort_values("creativity_score", ascending=False).reset_index(drop=True)

# --------------------------
# Physical metrics handling
# --------------------------
def canonicalise_physical_columns(df_phys_like):
    """
    Accepts any reasonable column names and maps to canonical:
      total_distance, hsr, sprint_distance, high_intensity
    """
    if df_phys_like is None or df_phys_like.empty:
        return pd.DataFrame()

    df = df_phys_like.copy()
    if "playerName" not in df.columns:
        st.warning("Physical CSV must contain 'playerName'. Ignoring.")
        return pd.DataFrame()

    # common variants
    variants = {
        "total_distance": ["total_distance", "totalDistance", "distance", "distance_total", "total_dist"],
        "hsr": ["hsr", "high_speed_running", "highSpeedRunning", "high_speed", "hsr_distance"],
        "sprint_distance": ["sprint_distance", "sprintDistance", "sprints", "sprint"],
        "high_intensity": ["high_intensity", "highIntensity", "high_intensity_distance", "high_intensity_dist", "hid"],
    }

    # create canonical columns if any variant exists; leave zeros otherwise
    for canon, opts in variants.items():
        found = None
        for c in opts:
            if c in df.columns:
                found = c
                break
        if found:
            df = df.rename(columns={found: canon})
        else:
            if canon not in df.columns:
                df[canon] = 0.0

    keep_cols = ["playerName", "total_distance", "hsr", "sprint_distance", "high_intensity"]
    return df[keep_cols]

def detect_and_aggregate_physical_from_events(df):
    """
    Try to detect physical columns in the event dataframe and aggregate per player.
    Returns canonicalised per-player totals if something was found; otherwise empty.
    """
    if df.empty:
        return pd.DataFrame()

    possible = {
        "total_distance": ["total_distance", "totalDistance", "distance", "distance_total"],
        "hsr": ["hsr", "high_speed_running", "highSpeedRunning"],
        "sprint_distance": ["sprint_distance", "sprintDistance", "sprint"],
        "high_intensity": ["high_intensity", "highIntensity", "high_intensity_distance"],
    }

    found_cols = {}
    for canon, opts in possible.items():
        for c in opts:
            if c in df.columns:
                found_cols[canon] = c
                break

    if not found_cols:
        return pd.DataFrame()

    agg = df.groupby("playerName").agg({src: "sum" for src in found_cols.values()}).reset_index()
    # rename back to canonical
    for canon, src in found_cols.items():
        agg = agg.rename(columns={src: canon})
    # ensure all canonical exist
    for c in ["total_distance", "hsr", "sprint_distance", "high_intensity"]:
        if c not in agg.columns:
            agg[c] = 0.0
    return agg

def compute_physical_scores(creativity_df, physical_source_df):
    """
    Merge creativity_df (has minutes_played) with physical_source_df (canonical cols),
    compute per-90 + normalised metrics, and flag availability.
    """
    if creativity_df.empty:
        return creativity_df.assign(physical_available=False)

    df = creativity_df.copy()

    # merge if provided
    if physical_source_df is not None and not physical_source_df.empty:
        df = df.merge(physical_source_df, on="playerName", how="left")
    else:
        # ensure columns exist
        for c in ["total_distance", "hsr", "sprint_distance", "high_intensity"]:
            df[c] = 0.0

    # per-90 for physicals
    for c in ["total_distance", "hsr", "sprint_distance", "high_intensity"]:
        df[f"{c}_per90"] = (df[c] / df["minutes_played"]) * 90

    # normalise 0..1
    for c in ["total_distance", "hsr", "sprint_distance", "high_intensity"]:
        col = f"{c}_per90"
        maxv = df[col].max()
        df[f"{c}_norm"] = df[col] / maxv if maxv and maxv > 0 else 0.0

    df["physical_available"] = (df[["total_distance", "hsr", "sprint_distance", "high_intensity"]].sum(axis=1) > 0)

    return df

def compute_technical_metrics(df):
    if df.empty:
        return pd.DataFrame()
    # base aggregations
    passes = df[df.get("typeId") == 1].copy()
    shots = df[df.get("typeId") == 13].copy()
    # pass accuracy
    pass_acc = passes.groupby("playerName").agg(
        passes_attempted=("id", "count"),
        passes_completed=("outcome", lambda s: (s == 1).sum())
    ).reset_index()
    pass_acc["pass_accuracy"] = pass_acc["passes_completed"] / pass_acc["passes_attempted"].replace({0: np.nan})
    pass_acc["pass_accuracy"] = pass_acc["pass_accuracy"].fillna(0)
    # progressive actions: reuse progressive_passes from creativity + add carries proxy (dx>10 on any event)
    df["dx"] = df.get("endX", df.get("x", 0)) - df.get("x", 0)
    prog_actions = df.groupby("playerName").agg(
        progressive_actions=("dx", lambda s: (s > 10).sum())
    ).reset_index()
    # shooting accuracy
    shot_acc = shots.groupby("playerName").agg(
        shots=("id", "count"),
        shots_success=("outcome", lambda s: (s == 1).sum())
    ).reset_index()
    shot_acc["shot_accuracy"] = shot_acc["shots_success"] / shot_acc["shots"].replace({0: np.nan})
    shot_acc["shot_accuracy"] = shot_acc["shot_accuracy"].fillna(0)
    # crossing activity proxy: passes from wide channels
    crosses = passes.copy()
    crosses["is_wide"] = crosses.apply(lambda r: (r.get("y", 50) <= 20) or (r.get("y", 50) >= 80), axis=1)
    cross_agg = crosses.groupby("playerName").agg(
        crosses_attempted=("is_wide", "sum")
    ).reset_index()

    # minutes
    minutes_played = df.groupby("playerName")["minute"].agg(lambda x: max(x) - min(x) + 1).rename("minutes_played").reset_index()

    out = minutes_played
    for part in [pass_acc, prog_actions, shot_acc, cross_agg]:
        out = out.merge(part, on="playerName", how="left")
    out = out.fillna(0)
    # per-90 rates where relevant
    for col in ["progressive_actions", "crosses_attempted", "passes_attempted", "passes_completed", "shots", "shots_success"]:
        if col in out.columns:
            out[f"{col}_per90"] = (out[col] / out["minutes_played"]) * 90
    # normalize and score 0–10 using sidebar weights
    def norm(col):
        m = out[col].max()
        return out[col] / m if m and m > 0 else 0.0
    out["tech_norm"] = (
        norm("pass_accuracy") * tw_pass_acc
        + norm("progressive_actions_per90") * tw_prog
        + norm("shot_accuracy") * tw_shot
        + norm("crosses_attempted_per90") * tw_cross
    )
    out["technical_score"] = out["tech_norm"] * 10.0
    return out.sort_values("technical_score", ascending=False).reset_index(drop=True)

def compute_tactical_metrics(df, gi_df):
    if df.empty:
        return pd.DataFrame()
    # Use GI components as tactical proxies + assist chain involvement
    gi = gi_df.copy() if gi_df is not None and not gi_df.empty else pd.DataFrame()
    # assist chain links: count times a player is the last passer before shot
    chains = []
    if "eventId" in df.columns:
        shots = df[df.get("typeId") == 13]
        for _, shot in shots.iterrows():
            prev_pass = df[(df["eventId"] < shot["eventId"]) & (df["typeId"] == 1) & (df["teamId"] == shot.get("teamId"))].tail(1)
            if not prev_pass.empty:
                chains.append(prev_pass[["playerName"]])
    chain_df = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame(columns=["playerName"])
    chain_agg = chain_df.groupby("playerName").size().rename("shot_linkups").reset_index()

    minutes_played = df.groupby("playerName")["minute"].agg(lambda x: max(x) - min(x) + 1).rename("minutes_played").reset_index()
    out = minutes_played.merge(chain_agg, on="playerName", how="left").fillna(0)
    out["shot_linkups_per90"] = (out["shot_linkups"] / out["minutes_played"]) * 90

    if not gi.empty:
        keep = [
            "playerName",
            "final_third_entries_per90", "box_entries_per90", "switches_per90",
            "zone14_entries_per90"
        ]
        gi_k = gi[[c for c in keep if c in gi.columns]].copy()
        out = out.merge(gi_k, on="playerName", how="left").fillna(0)

    def norm(s, col):
        m = s[col].max()
        s[col+"_norm"] = s[col] / m if m and m > 0 else 0.0
        return s
    for col in ["final_third_entries_per90", "box_entries_per90", "switches_per90", "zone14_entries_per90", "shot_linkups_per90"]:
        if col in out.columns:
            out = norm(out, col)

    out["tactical_norm"] = (
        out.get("final_third_entries_per90_norm", 0) * tac_final
        + out.get("switches_per90_norm", 0) * tac_switch
        + out.get("zone14_entries_per90_norm", 0) * tac_zone14
        + out.get("shot_linkups_per90_norm", 0) * tac_link
    )
    out["tactical_score"] = out["tactical_norm"] * 10.0
    return out.sort_values("tactical_score", ascending=False).reset_index(drop=True)

# --------------------------
# Game Intelligence metrics (proxies inspired by SkillCorner article)
# --------------------------
def compute_game_intelligence_metrics(df, weights=None, epv_params=None):
    """
    Computes proxy metrics per player (as passer) approximating aspects of "Game Intelligence":
      - final_third_entries: successful passes ending in final third (x >= 66)
      - box_entries: successful passes ending inside penalty box (x >= 84, 18 <= y <= 82)
      - switches: lateral + forward passes (|dy| >= 30 and dx >= 20)
      - through_pass_proxy: forward vertical passes into space (dx >= 25 and |dy| <= 10 and start x < 60)
      - zone14_entries: successful passes into Zone 14 (central outside box)
      - one_touch_shot_setups: last successful pass immediately prior to a shot by same team

    Returns per-90 counts, normalised 0..1 columns, and a composite gi_score on 0–10.
    """
    if df.empty:
        return pd.DataFrame()

    passes = df[(df.get("typeId") == 1) & (df.get("outcome") == 1)].copy()
    for col in ["x", "y", "endX", "endY"]:
        if col not in passes.columns:
            passes[col] = np.nan
    passes = passes.dropna(subset=["x", "y", "endX", "endY"]).copy()

    minutes_played = df.groupby("playerName")["minute"].agg(lambda x: max(x) - min(x) + 1).rename("minutes_played")

    def is_final_third(row):
        return row["endX"] >= 66

    def is_box_entry(row):
        return (row["endX"] >= 84) and (row["endY"] >= 18) and (row["endY"] <= 82)

    def is_switch(row):
        dx = row["endX"] - row["x"]
        dy = abs(row["endY"] - row["y"])
        return (dy >= 30) and (dx >= 20)

    def is_through_proxy(row):
        dx = row["endX"] - row["x"]
        dy = abs(row["endY"] - row["y"])
        return (row["x"] < 60) and (dx >= 25) and (dy <= 10)

    # Zone 14 proxy on 0..100 pitch
    def is_zone14(row):
        return (row["endX"] >= 78) and (row["endX"] <= 88) and (row["endY"] >= 42) and (row["endY"] <= 58)

    passes["final_third"] = passes.apply(is_final_third, axis=1)
    passes["box_entry"] = passes.apply(is_box_entry, axis=1)
    passes["switch"] = passes.apply(is_switch, axis=1)
    passes["through_proxy"] = passes.apply(is_through_proxy, axis=1)
    passes["zone14_entry"] = passes.apply(is_zone14, axis=1)

    shot_events = df[df.get("typeId") == 13]
    one_touch_ids = set()
    if not shot_events.empty and "eventId" in df.columns:
        for _, shot in shot_events.iterrows():
            prev = df[(df["eventId"] < shot["eventId"]) & (df["teamId"] == shot.get("teamId"))]
            prev_pass = prev[prev["typeId"] == 1].tail(1)
            if not prev_pass.empty:
                eid = prev_pass.iloc[0].get("id")
                if pd.notna(eid):
                    one_touch_ids.add(eid)
    passes["one_touch_shot_setup"] = passes["id"].isin(one_touch_ids)

    agg = passes.groupby("playerName").agg(
        final_third_entries=("final_third", "sum"),
        box_entries=("box_entry", "sum"),
        switches=("switch", "sum"),
        through_passes_proxy=("through_proxy", "sum"),
        zone14_entries=("zone14_entry", "sum"),
        one_touch_shot_setups=("one_touch_shot_setup", "sum"),
    ).reset_index()

    gi = agg.merge(minutes_played.reset_index(), on="playerName", how="left")
    gi["minutes_played"] = gi["minutes_played"].replace({0: 1}).fillna(1)

    count_cols = [
        "final_third_entries", "box_entries", "switches",
        "through_passes_proxy", "zone14_entries", "one_touch_shot_setups"
    ]
    for col in count_cols:
        gi[f"{col}_per90"] = (gi[col] / gi["minutes_played"]) * 90

    # Optional EPV weighting of counts based on pass end location
    def simple_epv(x, y, beta_x=0.08, center_pref=0.8):
        # x,y on 0..100; more value closer to goal (x) and more central (y near 50)
        forward_term = 1.0 / (1.0 + np.exp(-beta_x * (x - 70)))  # rise near final third
        center_term = np.exp(-((y - 50.0) ** 2) / (2 * (15.0 ** 2))) ** center_pref
        return np.clip(forward_term * center_term, 0.0, 1.0)

    if epv_params and not passes.empty:
        # If an EPV surface is provided, prefer bilinear sampling; else fallback to analytic EPV
        epv_surface = epv_params.get("surface")
        if epv_surface is not None:
            grid_X, grid_Y, grid_V = epv_surface
            grid_xs = grid_X[0, :]
            grid_ys = grid_Y[:, 0]
            xs = passes["endX"].to_numpy()
            ys = passes["endY"].to_numpy()
            passes["epv_w"] = bilinear_sample(grid_V, xs, ys, grid_xs, grid_ys)
        else:
            bx = float(epv_params.get("beta_x", 0.08))
            cp = float(epv_params.get("center_pref", 0.8))
            passes["epv_w"] = passes.apply(lambda r: simple_epv(r["endX"], r["endY"], bx, cp), axis=1)
    else:
        passes["epv_w"] = 1.0

    weighted_agg = passes.groupby("playerName").agg(
        final_third_entries_w=("final_third", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
        box_entries_w=("box_entry", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
        switches_w=("switch", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
        through_passes_proxy_w=("through_proxy", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
        zone14_entries_w=("zone14_entry", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
        one_touch_shot_setups_w=("one_touch_shot_setup", lambda s: float(np.dot(s.astype(float), passes.loc[s.index, "epv_w"]))),
    ).reset_index()
    gi = gi.merge(weighted_agg, on="playerName", how="left")

    # Normalise 0..1 for raw per-90 counts and EPV-weighted per-90 counts
    for col in [f"{c}_per90" for c in count_cols]:
        maxv = gi[col].max()
        gi[f"{col}_norm"] = gi[col] / maxv if maxv and maxv > 0 else 0.0
    for col in [
        "final_third_entries_w", "box_entries_w", "switches_w",
        "through_passes_proxy_w", "zone14_entries_w", "one_touch_shot_setups_w"
    ]:
        gi[f"{col}_per90"] = (gi[col].fillna(0) / gi["minutes_played"]) * 90
        maxv = gi[f"{col}_per90"].max()
        gi[f"{col}_per90_norm"] = gi[f"{col}_per90"] / maxv if maxv and maxv > 0 else 0.0

    # Weighted GI score
    default_w = {
        "final": 1/6,
        "box": 1/6,
        "switch": 1/6,
        "through": 1/6,
        "zone14": 1/6,
        "one_touch": 1/6,
    }
    w = weights or default_w
    # Choose EPV-normalised channels if EPV weighting is enabled
    use_epv_norm = bool(epv_params)
    gi["gi_norm_weighted"] = (
        w.get("final", default_w["final"]) * gi.get("final_third_entries_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
        + w.get("box", default_w["box"]) * gi.get("box_entries_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
        + w.get("switch", default_w["switch"]) * gi.get("switches_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
        + w.get("through", default_w["through"]) * gi.get("through_passes_proxy_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
        + w.get("zone14", default_w["zone14"]) * gi.get("zone14_entries_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
        + w.get("one_touch", default_w["one_touch"]) * gi.get("one_touch_shot_setups_{}per90_norm".format("w_" if use_epv_norm else ""), 0)
    )
    gi["gi_score"] = gi["gi_norm_weighted"] * 10.0

    gi = gi[gi["minutes_played"] >= min_minutes]

    return gi.sort_values("gi_score", ascending=False).reset_index(drop=True)

# --------------------------
# Visualisations (Plotly)
# --------------------------
def radar_plot(cre_df, player1, player2):
    metrics_norm = [
        "xT_sum_per90_norm", "avg_xT_norm",
        "progressive_passes_per90_norm", "total_passes_per90_norm"
    ]
    labels = ["xT/90", "avg xT", "prog passes/90", "passes/90"]

    missing = [m for m in metrics_norm if m not in cre_df.columns]
    if missing:
        st.info(f"Radar needs columns {missing} — compute creativity first.")
        return

    if player1 not in cre_df["playerName"].values or player2 not in cre_df["playerName"].values:
        st.info("Select two players present in the table.")
        return

    vals1 = cre_df[cre_df["playerName"] == player1][metrics_norm].iloc[0].fillna(0).tolist()
    vals2 = cre_df[cre_df["playerName"] == player2][metrics_norm].iloc[0].fillna(0).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals1, theta=labels, fill="toself", name=player1))
    fig.add_trace(go.Scatterpolar(r=vals2, theta=labels, fill="toself", name=player2))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

def xt_heatmap(df):
    st.subheader("🌍 Creativity Map — xT heatmap")
    passes = df[(df["typeId"] == 1)]
    if passes.empty:
        st.info("No pass events found to build the xT heatmap.")
        return
    fig = px.density_heatmap(
        passes, x="x", y="y", z="xT",
        nbinsx=12, nbinsy=8, color_continuous_scale="Reds",
        labels={"x": "Pitch X", "y": "Pitch Y", "xT": "Expected Threat"}
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), height=450)
    st.plotly_chart(fig, use_container_width=True)

def passing_network(df, show_arrows=False, max_arrows=250):
    st.subheader("🧭 Passing network")
    passes = df[(df["typeId"] == 1) & (df["outcome"] == 1)]
    if passes.empty:
        st.info("No successful passes in this dataset.")
        return

    # player average locations (as nodes)
    node_pos = passes.groupby("playerName")[["x", "y"]].mean().reset_index()
    # pass edges aggregated by passer -> receiver end location (proxy: towards endX,endY)
    # Note: Without explicit receiver name, we visualize pass vectors only.
    fig = go.Figure()

    # draw nodes
    fig.add_trace(go.Scatter(
        x=node_pos["x"], y=node_pos["y"], mode="markers+text",
        marker=dict(size=10),
        text=node_pos["playerName"], textposition="top center",
        name="Players"
    ))

    # draw pass vectors (may be many; limit arrows if requested)
    draw_count = 0
    for _, r in passes.iterrows():
        x0, y0, x1, y1 = r.get("x"), r.get("y"), r.get("endX"), r.get("endY")
        if pd.notna(x0) and pd.notna(y0) and pd.notna(x1) and pd.notna(y1):
            # line
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color="rgba(0,100,200,0.3)", width=1),
                showlegend=False
            ))
            # arrowhead (annotation)
            if show_arrows and draw_count < max_arrows:
                fig.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, opacity=0.5
                )
                draw_count += 1

    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def shot_map(df):
    st.subheader("🎯 Shot map")
    shots = df[df["typeId"] == 13]
    if shots.empty:
        st.info("No shots in this dataset.")
        return
    fig = go.Figure()
    for _, r in shots.iterrows():
        color = "green" if r.get("outcome") == 1 else "red"
        fig.add_trace(go.Scatter(
            x=[r.get("x")], y=[r.get("y")],
            mode="markers",
            marker=dict(size=8, color=color),
            name=r.get("playerName")
        ))
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), height=450)
    st.plotly_chart(fig, use_container_width=True)

def assist_chains(df):
    st.subheader("🔗 Assist chains (last pass before shot)")
    if "eventId" not in df.columns:
        st.info("eventId not present; cannot compute assist chains.")
        return
    shots = df[df["typeId"] == 13]
    if shots.empty:
        st.info("No shots — no assist chains.")
        return
    chains = []
    for _, shot in shots.iterrows():
        prev_pass = df[(df["eventId"] < shot["eventId"]) & (df["typeId"] == 1) & (df["teamId"] == shot["teamId"])].tail(1)
        if not prev_pass.empty:
            chains.append(prev_pass.assign(shot_player=shot.get("playerName"), shot_minute=shot.get("minute")))
    if chains:
        chains_df = pd.concat(chains, ignore_index=True)
        show_cols = ["playerName", "xT", "minute", "shot_player", "shot_minute"]
        existing = [c for c in show_cols if c in chains_df.columns]
        st.dataframe(chains_df[existing], use_container_width=True)
        st.download_button("Download assist chains CSV", chains_df[existing].to_csv(index=False), "assist_chains.csv")
    else:
        st.info("No pre-shot passes found for the shots in this dataset.")

def player_profile(df, creativity_df, overall_df, gi_df=None, tech_df=None, tac_df=None):
    st.subheader("🧑‍💻 Player profile")
    options = sorted(creativity_df["playerName"].unique().tolist()) if not creativity_df.empty else []
    if not options:
        st.info("No players available.")
        return
    player = st.selectbox("Select player", options)

    p_events = df[df["playerName"] == player]
    p_cre = creativity_df[creativity_df["playerName"] == player]
    p_over = overall_df[overall_df["playerName"] == player] if not overall_df.empty else pd.DataFrame()
    p_gi = gi_df[gi_df["playerName"] == player] if gi_df is not None and not gi_df.empty else pd.DataFrame()
    p_tech = tech_df[tech_df["playerName"] == player] if tech_df is not None and not tech_df.empty else pd.DataFrame()
    p_tac = tac_df[tac_df["playerName"] == player] if tac_df is not None and not tac_df.empty else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minutes played (approx.)", f"{p_cre['minutes_played'].iloc[0]:.0f}" if not p_cre.empty else "—")
        st.metric("Total passes", f"{p_cre['total_passes'].iloc[0]:.0f}" if "total_passes" in p_cre.columns and not p_cre.empty else "—")
    with col2:
        st.metric("xT sum /90", f"{p_cre['xT_sum_per90'].iloc[0]:.2f}" if "xT_sum_per90" in p_cre.columns and not p_cre.empty else "—")
        st.metric("Avg xT/pass", f"{p_cre['avg_xT'].iloc[0]:.3f}" if "avg_xT" in p_cre.columns and not p_cre.empty else "—")
    with col3:
        st.metric("Creativity (0–10)", f"{p_cre['creativity_score'].iloc[0]:.2f}" if "creativity_score" in p_cre.columns and not p_cre.empty else "—")
        if not p_over.empty and "overall_score" in p_over.columns:
            st.metric("Overall (0–10)", f"{p_over['overall_score'].iloc[0]:.2f}")

    if not p_gi.empty:
        st.markdown("**Game Intelligence (per-90)**")
        gi_show = [
            "final_third_entries_per90", "box_entries_per90", "switches_per90",
            "through_passes_proxy_per90", "zone14_entries_per90", "one_touch_shot_setups_per90", "gi_score"
        ]
        gi_existing = [c for c in gi_show if c in p_gi.columns]
        st.dataframe(p_gi[gi_existing], use_container_width=True)

    if not p_tech.empty or not p_tac.empty:
        colA, colB = st.columns(2)
        with colA:
            if not p_tech.empty:
                st.metric("Technical (0–10)", f"{p_tech['technical_score'].iloc[0]:.2f}")
        with colB:
            if not p_tac.empty:
                st.metric("Tactical (0–10)", f"{p_tac['tactical_score'].iloc[0]:.2f}")

    st.markdown("**Recent actions**")
    show_cols = [c for c in ["minute", "typeId", "x", "y", "endX", "endY", "xT", "outcome"] if c in p_events.columns]
    st.dataframe(p_events[show_cols].tail(30), use_container_width=True)

# --------------------------
# Main UI behaviour
# --------------------------
if uploaded_files:
    df = load_multiple_files(uploaded_files)
    if df.empty:
        st.error("No valid events parsed from uploaded files.")
    else:
        # optional team filter
        if team_filter_option == "AC Horsens only":
            df = df[df["teamName"].astype(str).str.contains("Horsens", na=False)]

        st.success(f"Loaded {len(uploaded_files)} file(s). Events: {len(df)}")

        # player + minute filters
        all_players = sorted(df["playerName"].unique().tolist())
        selected_players = st.sidebar.multiselect("Select players (empty = all)", all_players, default=all_players[:5])
        minute_range = st.sidebar.slider("Minute range", 0, 120, (0, 90))

        df_filt = df.copy()
        if selected_players:
            df_filt = df_filt[df_filt["playerName"].isin(selected_players)]
        df_filt = df_filt[(df_filt["minute"] >= minute_range[0]) & (df_filt["minute"] <= minute_range[1])]

        # compute creativity (per-90)
        creativity_df = compute_creativity_metrics(df_filt)

        # physicals: from CSV or auto-detected in events
        physical_source_df = pd.DataFrame()
        if phys_csv:
            try:
                phys_csv.seek(0)
                uploaded_phys = pd.read_csv(phys_csv)
                physical_source_df = canonicalise_physical_columns(uploaded_phys)
                st.sidebar.success("Using uploaded physical CSV.")
            except Exception as e:
                st.sidebar.warning(f"Could not read physical CSV: {e}")
        else:
            auto_phys = detect_and_aggregate_physical_from_events(df_filt)
            if not auto_phys.empty:
                physical_source_df = canonicalise_physical_columns(auto_phys)
                st.sidebar.success("Detected physical metrics in event data.")
            else:
                st.sidebar.info("No player-level physicals provided/found. Physical score will be zero.")

        # merge + compute per-90 physicals + normals
        cre_phys = compute_physical_scores(creativity_df, physical_source_df)

        # overall score (blend)
        c = cre_phys.copy()
        # combined physical norm via sub-weights
        c["physical_norm_combined"] = (
            c.get("total_distance_norm", 0) * pd_w_dist
            + c.get("hsr_norm", 0) * pd_w_hsr
            + c.get("sprint_distance_norm", 0) * pd_w_sprint
            + c.get("high_intensity_norm", 0) * pd_w_hid
        )
        # creativity norm (0..1)
        c["creativity_norm"] = c["creativity_score"].fillna(0) / 10.0
        # overall norm (0..1) and 0..10
        c["overall_norm"] = creativity_weight * c["creativity_norm"] + physical_weight * c["physical_norm_combined"]
        c["overall_score"] = c["overall_norm"] * 10.0

        # Display: combined index
        st.subheader("🏷 Player Index — Overall (Creativity + Physical)")
        display_cols = [
            "playerName",
            "overall_score",
            "creativity_score",
            "creativity_norm",
            "physical_norm_combined",
            "minutes_played",
            "xT_sum", "xT_sum_per90", "avg_xT",
            "progressive_passes", "progressive_passes_per90",
            "total_passes", "total_passes_per90",
            "total_distance_per90", "hsr_per90", "sprint_distance_per90", "high_intensity_per90"
        ]
        existing = [col for col in display_cols if col in c.columns]
        table_overall = c[existing].sort_values("overall_score", ascending=False).reset_index(drop=True)
        st.dataframe(table_overall, use_container_width=True)
        st.download_button("⬇️ Download overall CSV", c.to_csv(index=False), "creativity_physical_index.csv")

        # Creativity-only (pretty) table for clarity
        st.subheader("🎯 Player Creativity Metrics (Per-90, 0–10)")
        pretty_cols = [
            "playerName", "creativity_score", "minutes_played",
            "xT_sum", "xT_sum_per90", "avg_xT",
            "progressive_passes", "progressive_passes_per90",
            "total_passes", "total_passes_per90"
        ]
        st.dataframe(creativity_df[[c for c in pretty_cols if c in creativity_df.columns]], use_container_width=True)
        st.download_button("⬇️ Download creativity CSV", creativity_df.to_csv(index=False), "creativity_metrics.csv")

        # Radar comparison (creativity dimensions)
        st.subheader("🏆 Compare players — creativity radar")
        players_list = c["playerName"].tolist()
        if len(players_list) >= 2:
            col_a, col_b = st.columns(2)
            with col_a:
                p1 = st.selectbox("Player 1", players_list, index=0, key="radar_p1")
            with col_b:
                p2 = st.selectbox("Player 2", players_list, index=1, key="radar_p2")
            radar_plot(c, p1, p2)
        else:
            st.info("Not enough players to compare.")

        # Game Intelligence table
        st.subheader("🧠 Game Intelligence — proxy metrics (Per-90, 0–10)")
        gi_weights = {
            "final": gi_w_final,
            "box": gi_w_box,
            "switch": gi_w_switch,
            "through": gi_w_through,
            "zone14": gi_w_zone14,
            "one_touch": gi_w_one_touch,
        }
        epv_params = {"beta_x": epv_beta_x, "center_pref": epv_center_pref}
        # Build or load an EPV surface if enabled
        epv_surface = None
        if learned_epv_file is not None:
            epv_loaded = load_learned_epv_surface(learned_epv_file)
            if epv_loaded is not None:
                epv_surface = epv_loaded
        if use_epv and build_epv_from_tracking and tracking_file is not None:
            frames = read_tracking_json(tracking_file)
            surf = build_epv_surface_from_tracking(
                frames,
                x_bins=epv_grid_x,
                y_bins=epv_grid_y,
                use_density=epv_use_density,
                attack_w=epv_attack_density_w,
                defense_w=epv_defense_density_w,
                flip_x=epv_flip_x,
                auto_flip=epv_auto_flip,
                first_half_dir=epv_first_half_dir,
            )
            if surf is not None:
                epv_surface = surf
        if use_epv:
            epv_params["surface"] = epv_surface
        else:
            epv_params = None
        gi_df = compute_game_intelligence_metrics(df_filt, weights=gi_weights, epv_params=epv_params)
        if not gi_df.empty:
            gi_display_cols = [
                "playerName", "gi_score", "minutes_played",
                "final_third_entries_per90", "box_entries_per90", "switches_per90",
                "through_passes_proxy_per90", "zone14_entries_per90", "one_touch_shot_setups_per90",
                # EPV-weighted per-90s if available
                "final_third_entries_w_per90", "box_entries_w_per90", "switches_w_per90",
                "through_passes_proxy_w_per90", "zone14_entries_w_per90", "one_touch_shot_setups_w_per90"
            ]
            st.dataframe(gi_df[[c for c in gi_display_cols if c in gi_df.columns]], use_container_width=True)
            st.download_button("⬇️ Download GI CSV", gi_df.to_csv(index=False), "game_intelligence_metrics.csv")
        else:
            st.info("Not enough data to compute Game Intelligence metrics.")

        # Technical and Tactical tables
        st.subheader("🛠️ Technical Index (Per-90, 0–10)")
        tech_df = compute_technical_metrics(df_filt)
        if not tech_df.empty:
            tech_cols = [
                "playerName", "technical_score", "minutes_played",
                "pass_accuracy", "progressive_actions_per90", "shot_accuracy", "crosses_attempted_per90"
            ]
            st.dataframe(tech_df[[c for c in tech_cols if c in tech_df.columns]], use_container_width=True)
            st.download_button("⬇️ Download Technical CSV", tech_df.to_csv(index=False), "technical_metrics.csv")
        else:
            st.info("Not enough data for technical metrics.")

        st.subheader("🧭 Tactical Index (Per-90, 0–10)")
        tac_df = compute_tactical_metrics(df_filt, gi_df)
        if not tac_df.empty:
            tac_cols = [
                "playerName", "tactical_score", "minutes_played",
                "final_third_entries_per90", "box_entries_per90", "switches_per90", "zone14_entries_per90", "shot_linkups_per90"
            ]
            st.dataframe(tac_df[[c for c in tac_cols if c in tac_df.columns]], use_container_width=True)
            st.download_button("⬇️ Download Tactical CSV", tac_df.to_csv(index=False), "tactical_metrics.csv")
        else:
            st.info("Not enough data for tactical metrics.")

        # Train buttons (executed on demand)
        learned_surface = None
        if train_epv_btn and tracking_file is not None:
            frames_for_train = read_tracking_json(tracking_file)
            learned_surface = train_epv_model_from_data(frames_for_train, df_filt, x_bins=epv_grid_x, y_bins=epv_grid_y, auto_flip=epv_auto_flip, first_half_dir=epv_first_half_dir)
            if learned_surface is not None:
                st.success("Learned EPV surface trained.")
        learned_rr_model = None
        if train_rr_btn:
            learned_rr_model = train_rr_model_from_data(df_filt)
            if learned_rr_model is not None:
                st.success("Pass completion classifier trained.")

        # Risk–Reward table
        if use_rr:
            st.subheader("⚖️ Pass Risk–Reward (EPV delta minus risk)")
            rr_surface = None
            if use_learned_epv and learned_surface is not None:
                rr_surface = learned_surface
            elif use_epv and epv_surface is not None:
                rr_surface = epv_surface
            rr_bins_df = None
            if rr_classifier_file is not None:
                try:
                    rr_classifier_file.seek(0)
                    rr_bins_df = pd.read_csv(rr_classifier_file)
                except Exception as e:
                    st.warning(f"Could not read RR classifier file: {e}")
            # Optional model-based risk prediction (if supplied)
            rr_model = None
            if use_learned_rr and learned_rr_model is not None:
                rr_model = learned_rr_model
            if rr_model_file is not None:
                try:
                    import pickle
                    rr_model_file.seek(0)
                    rr_model = pickle.load(rr_model_file)
                except Exception as e:
                    st.warning(f"Could not load RR classifier model: {e}")
            rr_df = compute_pass_risk_reward(df_filt, epv_surface=rr_surface, lambda_risk=rr_risk_lambda, rr_bins=rr_bins_df)
            if not rr_df.empty:
                rr_cols = [
                    "playerName", "rr_sum_per90", "epv_reward_sum_per90", "risk_sum_per90", "pass_count_per90"
                ]
                st.dataframe(rr_df[[c for c in rr_cols if c in rr_df.columns]], use_container_width=True)
                st.download_button("⬇️ Download Risk–Reward CSV", rr_df.to_csv(index=False), "risk_reward.csv")
            else:
                st.info("Not enough data to compute risk–reward.")

        # EPV Heatmap (if built and mplsoccer available)
        if use_epv and build_epv_from_tracking and tracking_file is not None:
            frames = read_tracking_json(tracking_file)
            surf = build_epv_surface_from_tracking(
                frames,
                x_bins=epv_grid_x,
                y_bins=epv_grid_y,
                use_density=epv_use_density,
                attack_w=epv_attack_density_w,
                defense_w=epv_defense_density_w,
                flip_x=epv_flip_x,
            )
            if surf is not None and Pitch is not None:
                st.subheader("🔥 EPV surface (tracking-derived)")
                Xg, Yg, Vg = surf
                fig, ax = plt.subplots(figsize=(8, 5))
                pitch = Pitch(pitch_type='opta', line_zorder=2)
                pitch.draw(ax=ax)
                pcm = ax.pcolormesh(Xg, Yg, Vg, cmap='Reds', shading='auto')
                fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)
            elif surf is not None:
                st.info("mplsoccer not available; EPV surface heatmap skipped.")

        # Off-ball EPV table
        if use_epv and build_epv_from_tracking and tracking_file is not None and epv_surface is not None:
            frames = read_tracking_json(tracking_file)
            offball_df = compute_offball_epv(
                frames, epv_surface,
                pitch_len_m=track_pitch_len,
                pitch_wid_m=track_pitch_wid,
                auto_flip=epv_auto_flip,
                first_half_dir=epv_first_half_dir,
            )
            if not offball_df.empty:
                st.subheader("🛰️ Off-ball EPV (tracking-derived)")
                st.dataframe(offball_df, use_container_width=True)
                st.download_button("⬇️ Download Off-ball EPV CSV", offball_df.to_csv(index=False), "offball_epv.csv")

        # Visualisations (pitch views)
        st.subheader("🗺️ Pitch Visualisations")
        col1, col2 = st.columns(2)
        with col1:
            xt_heatmap(df_filt)
        with col2:
            passing_network(df_filt, show_arrows=show_arrows, max_arrows=max_arrows)

        col3, col4 = st.columns(2)
        with col3:
            shot_map(df_filt)
        with col4:
            assist_chains(df_filt)

        # Player profile
        player_profile(df_filt, creativity_df, c, gi_df, tech_df, tac_df)

        # Optional tracking preview
        if tracking_file is not None:
            st.subheader("📹 Tracking preview")
            frames = read_tracking_json(tracking_file)
            if frames:
                st.caption("Showing first 1–3 frames for sanity check")
                st.json(frames[0])
                if len(frames) > 1:
                    st.json(frames[1])
                if len(frames) > 2:
                    st.json(frames[2])
            else:
                st.info("No frames parsed from tracking file.")

else:
    st.info("📥 Upload one or more Opta event JSON files to begin analysis.")

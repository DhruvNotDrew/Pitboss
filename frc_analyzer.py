"""
FRC WPILog Deep Analysis App
- Python event detection with exact timestamps.
- Multi-match comparison (load multiple CSVs)
- Auto/Teleop/Endgame period detection
- 20+ event detectors: battery, CAN, vision, drive, motors, loops, pneumatics, etc.
- Every event shows exact key name + timestamp + match period
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import stats
import os
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import threading
import re

# ── Palette ───────────────────────────────────────────────────────────────────
BG_DARK  = "#0a0e1a"; BG_PANEL = "#111827"; BG_CARD  = "#1a2235"; BG_CARD2 = "#1e2a3a"
AC_CYAN  = "#00d4ff"; AC_GREEN = "#00ff88"; AC_RED   = "#ff4444"
AC_WARN  = "#ffaa00"; AC_PURP  = "#a855f7"; AC_BLUE  = "#3b82f6"
TEXT_PRIM= "#e8f4fd"; TEXT_SEC = "#8899aa"; TEXT_DIM = "#445566"; BORDER = "#1e3a5f"
CHART_COLORS = ["#00d4ff","#00ff88","#a855f7","#ffaa00","#ff6b6b",
                "#4ecdc4","#ffd93d","#ff6b9d","#c084fc","#34d399","#fb923c","#38bdf8"]
SEV_COLOR = {"CRITICAL": "#ff4444", "WARNING": "#ffaa00", "INFO": "#00d4ff", "GOOD": "#00ff88"}
PERIOD_COLOR = {"auto": "#00ff88", "teleop": "#00d4ff", "endgame": "#a855f7", "disabled": "#445566", "unknown": "#8899aa"}


# ═════════════════════════════════════════════════════════════════════════════
#  EVENT DATA CLASS
# ═════════════════════════════════════════════════════════════════════════════

class MatchEvent:
    __slots__ = ("time","end_time","severity","category","key","description","detail","period","match_label")

    def __init__(self, time, severity, category, key, description, detail="", end_time=None, period="unknown", match_label=""):
        self.time        = round(float(time), 3)
        self.end_time    = round(float(end_time), 3) if end_time is not None else None
        self.severity    = severity        # CRITICAL / WARNING / INFO / GOOD
        self.category    = category        # battery / vision / drive / can / etc.
        self.key         = key             # EXACT column name that triggered this
        self.description = description     # short label
        self.detail      = detail          # verbose explanation
        self.period      = period          # auto / teleop / endgame / disabled
        self.match_label = match_label     # which match file this came from

    def dur(self):
        if self.end_time is not None:
            return round(self.end_time - self.time, 2)
        return None

    def dur_str(self):
        d = self.dur()
        return f"{d:.2f}s" if d is not None else ""

    def time_str(self):
        if self.end_time is not None:
            return f"t={self.time:.2f}s – {self.end_time:.2f}s"
        return f"t={self.time:.2f}s"

    def __repr__(self):
        return f"[{self.severity}][{self.period}] {self.time_str()} [{self.category}] {self.key}: {self.description}"


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class FRCAnalyzer:

    KEYWORDS = {
        "battery":     ["battery","voltage","vbus","pdh","busvolts"],
        "current":     ["current","ampere","amps","amp","stator","supply","outputcurrent"],
        "drive":       ["drive","chassis","wheel","swerve","module","velocity",
                        "encoder","distance","heading","yaw","gyro","navx",
                        "pigeon","imu","odometry","pose","translation","rotation"],
        "can":         ["canutilization","canbus","can_util","busutilization"],
        "intake":      ["intake","index","feeder","hopper","roller","kicker","agitator","collect"],
        "shooter":     ["shoot","flywheel","launch","rpm","hood","turret","flywheelspeed"],
        "climber":     ["climb","hook","winch","hang","ratchet","telescope"],
        "arm":         ["arm","shoulder","elbow","wrist","elevator","lift","pivot","extend","manipulator"],
        "pneumatics":  ["pressure","solenoid","pneum","compressor","piston","airpressure"],
        "vision":      ["vision","camera","limelight","target","pipeline","apriltag",
                        "photon","hastarget","tv","tx","ty","ta","latency","heartbeat","detected"],
        "match":       ["match","auto","teleop","enabled","mode","alliance","game","state","period","fms","isenable","isenabled","isauto","isteleop","isendgame"],
        "temperature": ["temp","temperature","celsius","fahrenheit","thermal","heat","motortemp"],
        "loop":        ["looptime","loopruntime","dt","cycletime","schedulertime","rioloop","loopduration"],
        "pdp":         ["pdp","channel","breaker","faultcurrent"],
        "gyro":        ["gyro","yaw","pitch","roll","navx","pigeon","angle"],
    }

    def __init__(self, df: pd.DataFrame, label: str = "Match", key_config=None):
        self.df           = df.copy()
        self.label        = label
        self.key_config   = key_config or {}
        self.time_col     = self._detect_time_col()
        self.numeric_cols = self._safe_numeric_cols()
        self.all_cols     = df.columns.tolist()
        self._cat_cache   = None
        self.events       = []
        self.periods      = []   # list of period names indexed by row (auto/teleop/endgame/disabled)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def update_key_config(self, key_config):
        self.key_config = key_config or {}
        self._cat_cache = None

    def _mapped_signal_key(self, signal_name):
        mapping = (self.key_config or {}).get("signal_map", {})
        col = mapping.get(signal_name)
        return col if col in self.df.columns else None

    def _mapped_signal_keys(self, signal_name):
        mapping = (self.key_config or {}).get("signal_map", {})
        raw = mapping.get(signal_name)
        if isinstance(raw, list):
            return [c for c in raw if c in self.df.columns]
        if isinstance(raw, str) and raw in self.df.columns:
            return [raw]
        return []

    def _is_probably_operator_input(self, col):
        cl = col.lower()
        bad_words = ["joystick", "controller", "xbox", "pov", "button", "trigger", "dpad", "stick"]
        return any(w in cl for w in bad_words)

    def _safe_numeric_cols(self):
        """Return only columns that are truly numeric after coercion."""
        result = []
        for c in self.df.columns:
            try:
                coerced = pd.to_numeric(self.df[c], errors="coerce")
                if coerced.notna().sum() > len(self.df) * 0.1:
                    result.append(c)
            except Exception:
                pass
        return result

    def _detect_time_col(self):
        # 1. Look for known headers
        for name in ["timestamp","time","Time","Timestamp","t","T","elapsed","Elapsed","matchtime"]:
            if name in self.df.columns:
                return name
        # 2. Look for ANY numeric column that is increasing (typical for time)
        for c in self.df.columns:
            try:
                s = pd.to_numeric(self.df[c], errors="coerce").dropna()
                if len(s) > 20 and s.is_monotonic_increasing:
                    return c
            except Exception:
                pass
        # 3. Fallback to the first column (risky, but prevents crash)
        return self.df.columns[0]

    def _t(self):
        return pd.to_numeric(self.df[self.time_col], errors="coerce")

    def _dur(self):
        t = self._t().dropna()
        return float(t.max() - t.min()) if len(t) > 1 else 0.0

    def _col_numeric(self, col):
        """Return a numeric Series for a column, coercing strings."""
        return pd.to_numeric(self.df[col], errors="coerce")

    def _col_bool(self, col):
        """Return a boolean Series: True=active/on."""
        raw = self.df[col]
        # already bool
        if raw.dtype == bool:
            return raw
        # numeric
        try:
            n = pd.to_numeric(raw, errors="coerce")
            if n.notna().sum() > 0:
                return n.fillna(0) > 0
        except Exception:
            pass
        # string
        return raw.astype(str).str.lower().isin(["true","1","1.0","yes","on","active","enabled"])

    def categorize_columns(self):
        if self._cat_cache:
            return self._cat_cache
        cats = defaultdict(list)
        forced = (self.key_config or {}).get("column_category", {})
        for col in self.df.columns:
            if col in forced and forced[col] in self.KEYWORDS:
                cats[forced[col]].append(col)
                continue
            cl = col.lower().replace("/","_").replace(" ","_").replace(".","_")
            placed = False
            for cat, words in self.KEYWORDS.items():
                if any(w in cl for w in words):
                    cats[cat].append(col); placed = True; break
            if not placed:
                cats["other"].append(col)
        self._cat_cache = dict(cats)
        return self._cat_cache

    def _base_stats(self, col):
        s = self._col_numeric(col).dropna()
        if len(s) == 0: return {}
        return dict(
            mean=round(float(s.mean()),4), std=round(float(s.std()),4),
            min=round(float(s.min()),4),  max=round(float(s.max()),4),
            p5=round(float(np.percentile(s,5)),4),
            p95=round(float(np.percentile(s,95)),4),
            zeros_pct=round(float((s==0).mean()*100),2)
        )

    # ── Period detection ──────────────────────────────────────────────────────

    def detect_match_periods(self):
        """
        Build a per-row period map: auto / teleop / endgame / disabled.
        Returns dict {row_index: period_name}.
        Match structure: auto ~0-15s, teleop ~15-105s, endgame ~105-135s.
        Uses enabled/auto/teleop columns if available; falls back to time-based.
        """
        t = self._t().values
        n = len(t)
        row_period = ["unknown"] * n
        cats = self.categorize_columns()

        # Try to find enable/auto/teleop boolean columns
        match_cols = cats.get("match", [])
        auto_col    = None
        teleop_col  = None
        enabled_col = self._mapped_signal_key("enabled_state")
        endgame_col = None

        for c in match_cols:
            cl = c.lower()
            if any(k in cl for k in ["isauto","isautonomous","auto_mode","automode"]) and auto_col is None:
                auto_col = c
            elif any(k in cl for k in ["isteleop","teleop_mode","teleopmode"]) and teleop_col is None:
                teleop_col = c
            elif any(k in cl for k in ["isenabled","enabled","robotenabled"]) and enabled_col is None:
                enabled_col = c
            elif any(k in cl for k in ["isendgame","endgame"]) and endgame_col is None:
                endgame_col = c

        t0 = float(t[0]) if n > 0 else 0

        for i in range(n):
            ti = float(t[i]) if not np.isnan(t[i]) else t0
            elapsed = ti - t0

            # Use boolean signals if available
            is_enabled  = True
            is_auto     = False
            is_teleop   = False
            is_endgame  = False

            if enabled_col:
                is_enabled = bool(self._col_bool(enabled_col).iloc[i])
            if auto_col:
                is_auto = bool(self._col_bool(auto_col).iloc[i])
            if teleop_col:
                is_teleop = bool(self._col_bool(teleop_col).iloc[i])
            if endgame_col:
                is_endgame = bool(self._col_bool(endgame_col).iloc[i])

            # If we have explicit columns, trust them
            if auto_col or teleop_col or enabled_col:
                if not is_enabled:
                    row_period[i] = "disabled"
                elif is_auto:
                    row_period[i] = "auto"
                elif is_endgame:
                    row_period[i] = "endgame"
                elif is_teleop:
                    row_period[i] = "teleop"
                else:
                    row_period[i] = "disabled"
            else:
                # Fall back to time-based (standard FRC match timing)
                if elapsed < 0:
                    row_period[i] = "disabled"
                elif elapsed < 15:
                    row_period[i] = "auto"
                elif elapsed < 105:
                    row_period[i] = "teleop"
                elif elapsed < 135:
                    row_period[i] = "endgame"
                else:
                    row_period[i] = "disabled"

        # Sanity fallback:
        # If explicit match-state columns produce a single dominant period for long logs,
        # they are likely the wrong signal. Fall back to standard FRC time windows.
        unique_periods = set(row_period)
        duration_guess = self._dur()
        if (auto_col or teleop_col or enabled_col) and duration_guess > 60 and len(unique_periods) <= 1:
            row_period = ["unknown"] * n
            for i in range(n):
                ti = float(t[i]) if not np.isnan(t[i]) else t0
                elapsed = ti - t0
                if elapsed < 0:
                    row_period[i] = "disabled"
                elif elapsed < 15:
                    row_period[i] = "auto"
                elif elapsed < 105:
                    row_period[i] = "teleop"
                elif elapsed < 135:
                    row_period[i] = "endgame"
                else:
                    row_period[i] = "disabled"

        self.periods = row_period
        return row_period

    def _period_at(self, idx):
        """Return period name at integer row index."""
        if self.periods and idx < len(self.periods):
            return self.periods[idx]
        return "unknown"

    # ── Run-detection core ────────────────────────────────────────────────────

    def _runs(self, bool_array, min_dur=0.0, min_rows=1):
        """
        Find contiguous True runs in bool_array (aligned with self.df rows).
        Returns list of (start_iloc, end_iloc, t_start, t_end, duration).
        bool_array must be a numpy bool array of length == len(self.df).
        """
        t   = self._t().values
        arr = np.asarray(bool_array, dtype=bool)
        assert len(arr) == len(t), f"bool_array length {len(arr)} != time length {len(t)}"

        runs = []
        in_run = False; s_i = 0
        for i, v in enumerate(arr):
            if v and not in_run:
                s_i = i; in_run = True
            elif (not v) and in_run:
                if (i - s_i) >= min_rows:
                    ts = float(t[s_i]); te = float(t[i-1])
                    dur = te - ts
                    if dur >= min_dur:
                        runs.append((s_i, i-1, ts, te, dur))
                in_run = False
        if in_run:
            rows = len(arr) - s_i
            if rows >= min_rows:
                ts = float(t[s_i]); te = float(t[-1])
                dur = te - ts
                if dur >= min_dur:
                    runs.append((s_i, len(arr)-1, ts, te, dur))
        return runs

    def _make_event(self, ts, te, severity, category, key, description, detail="", si=0):
        period = self._period_at(si)
        return MatchEvent(ts, severity, category, key, description, detail,
                          end_time=te, period=period, match_label=self.label)

    def _normalize_desc(self, text):
        """Normalize event text so similar messages can be merged."""
        t = (text or "").lower()
        t = re.sub(r"\d+(\.\d+)?", "#", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _post_process_events(self, evts):
        """
        Reduce noisy output:
        - merge duplicate events that happen at same time window across keys
        - soften persistent whole-match warnings that are usually not actionable
        """
        if not evts:
            return evts

        match_dur = max(self._dur(), 0.0)
        merged = {}
        merged_keys = {}
        order = []

        for e in evts:
            dur = e.dur()
            # Long, persistent warnings are often telemetry-noise conditions.
            if (
                dur is not None
                and e.severity == "WARNING"
                and e.category in {"vision", "drive", "match"}
                and dur > max(120.0, match_dur * 0.8)
            ):
                e.severity = "INFO"
                if "most of match" not in e.description.lower():
                    e.description = f"{e.description} (most of match)"

            t0 = round(e.time, 1)
            t1 = round(e.end_time if e.end_time is not None else e.time, 1)
            norm_desc = self._normalize_desc(e.description)
            merge_key = (e.severity, e.category, e.period, norm_desc, t0, t1)

            if merge_key not in merged:
                merged[merge_key] = e
                merged_keys[merge_key] = {e.key}
                order.append(merge_key)
            else:
                merged_keys[merge_key].add(e.key)

        out = []
        for mk in order:
            evt = merged[mk]
            keys = sorted(merged_keys[mk])
            if len(keys) > 1:
                shown = ", ".join(keys[:4])
                extra = f" (+{len(keys)-4} more)" if len(keys) > 4 else ""
                evt.key = f"{len(keys)} keys"
                evt.detail = f"{evt.detail} | Also triggered by: {shown}{extra}".strip()
            out.append(evt)

        out.sort(key=lambda x: x.time)
        return out

    # ═══════════════════════════════════════════════════════════════════════════
    #  MASTER EVENT DETECTOR
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_all_events(self):
        evts = []
        cats = self.categorize_columns()
        
        # Verify time column exists and is numeric
        try:
            t = self._t().values
            n = len(t)
        except Exception as e:
            print(f"Error accessing time column: {e}")
            return []

        self.detect_match_periods()

        
        try: # ── 1. BROWNOUT (voltage < 6.8V) ─────────────────────────────────────
            for col in cats.get("battery", []):
                cl = col.lower()
                if not any(k in cl for k in ["voltage","battery","vbus","busvolts"]): continue
                s = self._col_numeric(col).ffill().fillna(0).values
                mask = s < 6.8
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.02, min_rows=1):
                    vmin = float(np.nanmin(s[si:ei+1]))
                    evts.append(self._make_event(ts, te, "CRITICAL", "battery", col,
                        f"BROWNOUT <6.8V for {dur:.2f}s (min {vmin:.2f}V)",
                        f"Key: {col} | min={vmin:.2f}V | dur={dur:.2f}s", si=si))
        except Exception as e:
            pass  # Skip if brownout detection fails

        try: # ── 2. LOW VOLTAGE <11.5V ─────────────────────────────────────────────
            for col in cats.get("battery", []):
                cl = col.lower()
                if not any(k in cl for k in ["voltage","battery","vbus","busvolts"]): continue
                s = self._col_numeric(col).ffill().fillna(13).values
                mask = s < 11.5
                for si,ei,ts,te,dur in self._runs(mask, min_dur=1.0, min_rows=5):
                    vmin = float(np.nanmin(s[si:ei+1]))
                    evts.append(self._make_event(ts, te, "WARNING", "battery", col,
                        f"Low voltage <11.5V for {dur:.1f}s (min {vmin:.2f}V)",
                        f"Key: {col} | min={vmin:.2f}V | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if low voltage detection fails

        try: # ── 3. VOLTAGE SAG (large drop in one step) ───────────────────────────
            for col in cats.get("battery", []):
                cl = col.lower()
                if not any(k in cl for k in ["voltage","battery","vbus","busvolts"]): continue
                s = self._col_numeric(col).ffill().fillna(13)
                diff = s.diff()
                for i in range(1, n):
                    dv = diff.iloc[i]
                    if pd.notna(dv) and float(dv) < -1.5:
                        ts2 = float(t[i]) if i < n else float(t[-1])
                        evts.append(self._make_event(ts2, None, "WARNING", "battery", col,
                            f"Voltage sag {float(dv):.2f}V in one step",
                            f"Key: {col} | drop={float(dv):.2f}V", si=i))
        except Exception as e:
            pass  # Skip if voltage sag detection fails

        try: # ── 4. HIGH CURRENT ───────────────────────────────────────────────────
            for col in cats.get("current", []) + cats.get("pdp", []):
                cl = col.lower()
                s  = self._col_numeric(col).fillna(0).values
                threshold = 60 if "climb" in cl else 80
                sev       = "CRITICAL" if "climb" in cl else "WARNING"
                mask = s > threshold
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.1, min_rows=2):
                    peak = float(np.nanmax(s[si:ei+1]))
                    evts.append(self._make_event(ts, te, sev, "current", col,
                        f"High current >{threshold}A for {dur:.2f}s (peak {peak:.1f}A)",
                        f"Key: {col} | peak={peak:.1f}A | threshold={threshold}A | dur={dur:.2f}s", si=si))
        except Exception as e:
            pass  # Skip if high current detection fails

        try: # ── 5. CAN BUS SATURATION ─────────────────────────────────────────────
            for col in cats.get("can", []):
                s = self._col_numeric(col).fillna(0).values
                for thresh, sev in [(95,"CRITICAL"), (80,"WARNING")]:
                    mask = s > thresh
                    for si,ei,ts,te,dur in self._runs(mask, min_dur=0.2, min_rows=3):
                        peak = float(np.nanmax(s[si:ei+1]))
                        evts.append(self._make_event(ts, te, sev, "can", col,
                            f"CAN bus {peak:.0f}% (>{thresh}%) for {dur:.2f}s",
                            f"Key: {col} | peak={peak:.1f}% | threshold={thresh}% | dur={dur:.2f}s", si=si))
                    break  # only fire highest threshold hit
        except Exception as e:
            pass  # Skip if CAN detection fails

        try: # ── 6. VISION / CAMERA NOT DETECTING TARGET ───────────────────────────
            vision_cols = self._mapped_signal_keys("vision_has_target")
            if not vision_cols:
                for c in cats.get("vision", []):
                    cl = c.lower()
                    if any(k in cl for k in ["hastarget","tv","detected","valid"]) and "latency" not in cl:
                        vision_cols.append(c)
            vision_cols = [c for c in vision_cols if not self._is_probably_operator_input(c)]
            if vision_cols:
                camera_on = []
                for col in vision_cols:
                    camera_on.append(self._col_bool(col).fillna(False).values.astype(bool))

                # For multiple cameras, only fire "no vision" when ALL cameras are off.
                combined_no_target = np.logical_not(np.logical_or.reduce(camera_on))
                label = vision_cols[0] if len(vision_cols) == 1 else f"{len(vision_cols)} cameras"
                for si,ei,ts,te,dur in self._runs(combined_no_target, min_dur=1.0, min_rows=5):
                    sev = "CRITICAL" if dur > 10 else "WARNING"
                    used = ", ".join(vision_cols[:4])
                    more = f" (+{len(vision_cols)-4} more)" if len(vision_cols) > 4 else ""
                    evts.append(self._make_event(ts, te, sev, "vision", label,
                        f"All mapped cameras lost target for {dur:.1f}s",
                        f"Keys: {used}{more} | no target t={ts:.2f}s–{te:.2f}s | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if vision detection fails

        try: # ── 7. VISION LATENCY HIGH ────────────────────────────────────────────
            for col in cats.get("vision", []):
                if "latency" not in col.lower(): continue
                s = self._col_numeric(col).fillna(0).values
                mask = s > 100  # >100ms is bad
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.5, min_rows=3):
                    peak = float(np.nanmax(s[si:ei+1]))
                    evts.append(self._make_event(ts, te, "WARNING", "vision", col,
                        f"Vision latency high {peak:.0f}ms for {dur:.1f}s",
                        f"Key: {col} | peak={peak:.0f}ms | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if vision latency detection fails

        try: # ── 8. ROBOT STUCK (pose / odometry not changing) ─────────────────────
            preferred = self._mapped_signal_key("robot_pose")
            pose_cols = [preferred] if preferred else []
            if not pose_cols:
                for c in cats.get("drive", []):
                    cl = c.lower()
                    if (
                        any(k in cl for k in ["pose2d", "pose", "odometry", "distance", "position", "translation"])
                        and not self._is_probably_operator_input(c)
                    ):
                        pose_cols.append(c)
            for col in pose_cols:
                cl = col.lower()
                s = self._col_numeric(col).ffill().dropna()
                if len(s) < 20: continue
                win = max(5, min(50, len(s)//30))
                rolling_std = s.rolling(win, min_periods=3).std().reindex(self.df.index).fillna(0)
                mask = (rolling_std < 0.005).values
                for si,ei,ts,te,dur in self._runs(mask, min_dur=3.0, min_rows=10):
                    if ts < 2.0: continue  # ignore pre-match
                    period_here = self._period_at(si)
                    if period_here not in {"teleop", "endgame"}:
                        continue
                    evts.append(self._make_event(ts, te, "WARNING", "drive", col,
                        f"Robot stuck/stationary — pose unchanged for {dur:.1f}s",
                        f"Key: {col} | no change t={ts:.2f}s–{te:.2f}s | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if robot stuck detection fails

        try: # ── 9. DRIVE VELOCITY ZERO (robot parked) ─────────────────────────────
            preferred = self._mapped_signal_key("robot_velocity")
            vel_cols = [preferred] if preferred else []
            if not vel_cols:
                for c in cats.get("drive", []):
                    cl = c.lower()
                    if any(k in cl for k in ["velocity","speed","mps","chassis"]) and not self._is_probably_operator_input(c):
                        vel_cols.append(c)
            for col in vel_cols:
                cl = col.lower()
                s = self._col_numeric(col).fillna(0).abs().values
                mask = s < 0.05
                for si,ei,ts,te,dur in self._runs(mask, min_dur=5.0, min_rows=10):
                    if ts < 3.0: continue
                    if self._period_at(si) not in {"teleop", "endgame"}:
                        continue
                    evts.append(self._make_event(ts, te, "INFO", "drive", col,
                        f"Drive velocity near zero for {dur:.1f}s (parked / disabled?)",
                        f"Key: {col} | stationary t={ts:.2f}s–{te:.2f}s | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if drive velocity detection fails

        try: # ── 10. MOTOR STALL (high current, near-zero velocity) ────────────────
            # Detect stalls by pairing current and velocity columns per subsystem
            for curr_col in cats.get("current", []):
                cl = curr_col.lower()
                # Find matching velocity for this subsystem
                vel_col = None
                for vc in cats.get("drive", []) + cats.get("shooter", []) + cats.get("arm", []) + cats.get("intake", []):
                    if any(k in vc.lower() for k in ["velocity","speed","rpm"]):
                        vel_col = vc; break
                if vel_col is None: continue
                sc = self._col_numeric(curr_col).fillna(0).abs().values
                sv = self._col_numeric(vel_col).fillna(0).abs().values
                if len(sc) != n or len(sv) != n: continue
                mean_c = float(np.nanmean(sc))
                stall_mask = (sc > max(mean_c * 2.5, 20)) & (sv < 0.1)
                for si,ei,ts,te,dur in self._runs(stall_mask, min_dur=0.3, min_rows=3):
                    peak_c = float(np.nanmax(sc[si:ei+1]))
                    evts.append(self._make_event(ts, te, "WARNING", "drive", curr_col,
                        f"Motor stall detected {dur:.2f}s — high current ({peak_c:.1f}A) + near-zero velocity",
                        f"Current key: {curr_col} | Velocity key: {vel_col} | peak={peak_c:.1f}A | dur={dur:.2f}s", si=si))
        except Exception as e:
            pass  # Skip if motor stall detection fails

        try: # ── 11. MOTOR TEMPERATURE ─────────────────────────────────────────────
            for col in cats.get("temperature", []):
                s = self._col_numeric(col).fillna(0).values
                for thresh, sev, lbl in [(80,"CRITICAL","MOTOR OVERHEAT"), (60,"WARNING","High motor temp")]:
                    mask = s > thresh
                    for si,ei,ts,te,dur in self._runs(mask, min_dur=0.5, min_rows=3):
                        peak = float(np.nanmax(s[si:ei+1]))
                        evts.append(self._make_event(ts, te, sev, "temperature", col,
                            f"{lbl}: {peak:.0f}°C for {dur:.1f}s",
                            f"Key: {col} | peak={peak:.1f}°C | threshold={thresh}°C | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if motor temperature detection fails

        try: # ── 12. LOOP OVERRUN ──────────────────────────────────────────────────
            for col in cats.get("loop", []):
                s = self._col_numeric(col).fillna(0).values
                # Loop time in seconds: >25ms is overrun; >50ms is critical
                for thresh, sev in [(0.050,"CRITICAL"), (0.025,"WARNING")]:
                    mask = s > thresh
                    for si,ei,ts,te,dur in self._runs(mask, min_dur=0.02, min_rows=1):
                        peak_ms = float(np.nanmax(s[si:ei+1])) * 1000
                        evts.append(self._make_event(ts, te, sev, "loop", col,
                            f"Loop overrun {peak_ms:.0f}ms (>{thresh*1000:.0f}ms) for {dur:.2f}s",
                            f"Key: {col} | peak={peak_ms:.1f}ms | threshold={thresh*1000:.0f}ms | dur={dur:.2f}s", si=si))
                    break
        except Exception as e:
            pass  # Skip if loop overrun detection fails

        try: # ── 13. LOW PNEUMATIC PRESSURE ────────────────────────────────────────
            for col in cats.get("pneumatics", []):
                if "pressure" not in col.lower(): continue
                s = self._col_numeric(col).fillna(999).values
                for thresh, sev in [(40,"WARNING"), (20,"CRITICAL")]:
                    mask = s < thresh
                    for si,ei,ts,te,dur in self._runs(mask, min_dur=0.3, min_rows=3):
                        vmin = float(np.nanmin(s[si:ei+1]))
                        evts.append(self._make_event(ts, te, sev, "pneumatics", col,
                            f"Low pressure {vmin:.0f} PSI (<{thresh} PSI) for {dur:.1f}s",
                            f"Key: {col} | min={vmin:.0f} PSI | threshold={thresh} PSI | dur={dur:.1f}s", si=si))
                    break
        except Exception as e:
            pass  # Skip if pneumatic pressure detection fails

        try: # ── 14. FLYWHEEL / SHOOTER INSTABILITY ────────────────────────────────
            for col in cats.get("shooter", []):
                cl = col.lower()
                if not any(k in cl for k in ["rpm","velocity","speed","flywheel"]): continue
                s = self._col_numeric(col).dropna()
                if len(s) < 10: continue
                z = np.abs(stats.zscore(s.values))
                spike_mask_full = np.zeros(n, dtype=bool)
                for idx_loc, (orig_idx, _) in enumerate(zip(s.index, s.values)):
                    if z[idx_loc] > 3.5:
                        row_loc = self.df.index.get_loc(orig_idx) if orig_idx in self.df.index else 0
                        if row_loc < n:
                            spike_mask_full[row_loc] = True
                for si,ei,ts,te,dur in self._runs(spike_mask_full, min_rows=1, min_dur=0):
                    val = float(self._col_numeric(col).iloc[si])
                    evts.append(self._make_event(ts, None, "WARNING", "shooter", col,
                        f"Flywheel instability spike: {val:.0f}",
                        f"Key: {col} | value={val:.1f} | z-score>3.5", si=si))
        except Exception as e:
            pass  # Skip if flywheel instability detection fails

        try: # ── 15. ARM / ELEVATOR POSITION SPIKE ────────────────────────────────
            for col in cats.get("arm", []):
                cl = col.lower()
                if any(k in cl for k in ["current","amp","temp"]): continue
                s = self._col_numeric(col).dropna()
                if len(s) < 10: continue
                z = np.abs(stats.zscore(s.values))
                for idx_loc, (orig_idx, _) in enumerate(zip(s.index, s.values)):
                    if z[idx_loc] > 4.0:
                        row_loc = self.df.index.get_loc(orig_idx) if orig_idx in self.df.index else 0
                        if row_loc < n:
                            ts2 = float(t[row_loc])
                            val = float(s.iloc[idx_loc])
                            evts.append(self._make_event(ts2, None, "WARNING", "arm", col,
                                f"Arm/elevator position spike: {val:.3f}",
                                f"Key: {col} | value={val:.3f} | z-score>4.0", si=row_loc))
        except Exception as e:
            pass  # Skip if arm spike detection fails

        try: # ── 16. ROBOT DISABLED MID-MATCH ─────────────────────────────────────
            preferred = self._mapped_signal_key("enabled_state")
            enabled_cols = [preferred] if preferred else []
            if not enabled_cols:
                enabled_cols = [c for c in cats.get("match", []) if any(k in c.lower() for k in ["enabled","isenabled","robotenabled"])]
            for col in enabled_cols:
                cl = col.lower()
                b = self._col_bool(col)
                dis = (~b).values
                for si,ei,ts,te,dur in self._runs(dis, min_dur=0.5, min_rows=3):
                    if ts < 3.0: continue
                    evts.append(self._make_event(ts, te, "CRITICAL", "match", col,
                        f"Robot DISABLED for {dur:.1f}s (e-stop / comm loss / brownout?)",
                        f"Key: {col} | disabled t={ts:.2f}s–{te:.2f}s | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if robot disabled detection fails

        try: # ── 17. GYRO / HEADING JUMP ───────────────────────────────────────────
            for col in cats.get("gyro", []):
                cl = col.lower()
                if "yaw" not in cl and "angle" not in cl and "heading" not in cl: continue
                s = self._col_numeric(col).ffill().fillna(0)
                diff = s.diff().abs()
                for i in range(1, n):
                    dv = diff.iloc[i]
                    if pd.notna(dv) and float(dv) > 30:
                        ts2 = float(t[i])
                        evts.append(self._make_event(ts2, None, "WARNING", "gyro", col,
                            f"Gyro heading jump {float(dv):.1f}° in one step (sensor glitch?)",
                            f"Key: {col} | jump={float(dv):.1f}° | possible IMU disconnect", si=i))
        except Exception as e:
            pass  # Skip if gyro jump detection fails

        try: # ── 18. CLIMBER ACTIVITY + HIGH CURRENT ──────────────────────────────
            for col in cats.get("climber", []):
                s = self._col_numeric(col).fillna(0).values
                # Activity
                mask = np.abs(s) > 0.05
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.3, min_rows=3):
                    peak = float(np.nanmax(np.abs(s[si:ei+1])))
                    evts.append(self._make_event(ts, te, "INFO", "climber", col,
                        f"Climber active {dur:.1f}s (peak {peak:.2f})",
                        f"Key: {col} | active t={ts:.2f}s–{te:.2f}s | peak={peak:.2f}", si=si))
                # High current climber
                cl = col.lower()
                if any(k in cl for k in ["current","amp"]):
                    mask2 = s > 60
                    for si,ei,ts,te,dur in self._runs(mask2, min_dur=0.1, min_rows=2):
                        peak = float(np.nanmax(s[si:ei+1]))
                        evts.append(self._make_event(ts, te, "CRITICAL", "climber", col,
                            f"Climber overcurrent {peak:.1f}A for {dur:.2f}s",
                            f"Key: {col} | peak={peak:.1f}A | threshold=60A | dur={dur:.2f}s", si=si))
        except Exception as e:
            pass  # Skip if climber detection fails

        try: # ── 19. INTAKE JAM (current spike) ────────────────────────────────────
            for col in cats.get("intake", []):
                cl = col.lower()
                if not any(k in cl for k in ["current","amp"]): continue
                s = self._col_numeric(col).fillna(0).values
                mean_v = float(np.nanmean(s)); std_v = float(np.nanstd(s))
                if std_v < 0.1: continue
                mask = s > mean_v + 3*std_v
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.1, min_rows=2):
                    peak = float(np.nanmax(s[si:ei+1]))
                    evts.append(self._make_event(ts, te, "WARNING", "intake", col,
                        f"Intake jam? Current spike {peak:.1f}A (>{mean_v+3*std_v:.1f}A) for {dur:.2f}s",
                        f"Key: {col} | peak={peak:.1f}A | mean={mean_v:.1f}A | dur={dur:.2f}s", si=si))
        except Exception as e:
            pass  # Skip if intake jam detection fails

        try: # ── 20. COMPRESSOR NOT RUNNING WHEN PRESSURE LOW ──────────────────────
            pressure_cols  = [c for c in cats.get("pneumatics",[]) if "pressure" in c.lower()]
            compressor_cols= [c for c in cats.get("pneumatics",[]) if "compressor" in c.lower()]
            if pressure_cols and compressor_cols:
                sp = self._col_numeric(pressure_cols[0]).fillna(999).values
                sc = self._col_bool(compressor_cols[0]).values
                mask = (sp < 60) & (~sc)
                for si,ei,ts,te,dur in self._runs(mask, min_dur=2.0, min_rows=5):
                    evts.append(self._make_event(ts, te, "WARNING", "pneumatics", pressure_cols[0],
                        f"Low pressure + compressor off for {dur:.1f}s (compressor fault?)",
                        f"Pressure key: {pressure_cols[0]} | Compressor key: {compressor_cols[0]} | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if compressor detection fails

        try: # ── 21. PDP BREAKER TRIP (channel goes to 0 mid-match) ────────────────
            for col in cats.get("pdp", []):
                cl = col.lower()
                if "current" not in cl and "amp" not in cl: continue
                s = self._col_numeric(col).fillna(0).values
                mean_v = float(np.nanmean(s))
                if mean_v < 1.0: continue  # channel not in use
                # A breaker trip looks like current suddenly going to 0 after being active
                mask = s < 0.5
                for si,ei,ts,te,dur in self._runs(mask, min_dur=0.5, min_rows=3):
                    if si == 0: continue
                    before = float(np.nanmean(s[max(0,si-10):si]))
                    if before > 2.0:
                        evts.append(self._make_event(ts, te, "CRITICAL", "pdp", col,
                            f"PDP channel may have tripped (current dropped from {before:.1f}A to 0) for {dur:.1f}s",
                            f"Key: {col} | before={before:.1f}A | dur={dur:.1f}s", si=si))
        except Exception as e:
            pass  # Skip if PDP breaker detection fails

        try: # ── 22. AUTO PERIOD — did robot move? ─────────────────────────────────
            # Look for completely motionless auto
            auto_rows = [i for i,p in enumerate(self.periods) if p == "auto"]
            if len(auto_rows) > 10:
                vel_cols = [c for c in cats.get("drive",[]) if any(k in c.lower() for k in ["velocity","speed"])]
                if vel_cols:
                    vc = vel_cols[0]
                    s  = self._col_numeric(vc).fillna(0).abs().values
                    auto_v = [s[i] for i in auto_rows if i < n]
                    if auto_v and float(np.mean(auto_v)) < 0.01:
                        ts2 = float(t[auto_rows[0]]); te2 = float(t[auto_rows[-1]])
                        evts.append(self._make_event(ts2, te2, "WARNING", "drive", vc,
                            "Robot did NOT move during AUTO (zero velocity entire auto period)",
                            f"Key: {vc} | avg velocity during auto = {float(np.mean(auto_v)):.4f}", si=auto_rows[0]))
        except Exception as e:
            pass  # Skip if auto movement detection fails

        try: # ── 23. TELEOP STARTED WITHOUT VISION LOCK ────────────────────────────
            teleop_rows = [i for i,p in enumerate(self.periods) if p == "teleop"]
            if teleop_rows:
                tele_start = teleop_rows[0]
                vision_cols = [c for c in cats.get("vision",[]) if any(k in c.lower() for k in ["hastarget","tv","detected"])]
                for vc in vision_cols:
                    check_rows = range(tele_start, min(tele_start+30, n))
                    s = self._col_bool(vc).values
                    early_vals = [s[i] for i in check_rows if i < n]
                    if early_vals and not any(early_vals):
                        ts2 = float(t[tele_start])
                        evts.append(self._make_event(ts2, None, "WARNING", "vision", vc,
                            "No vision target at teleop start (first ~30 rows of teleop)",
                            f"Key: {vc} | camera not locked at teleop begin t={ts2:.2f}s", si=tele_start))
        except Exception as e:
            pass  # Skip if teleop vision detection fails
            
        self.events = self._post_process_events(evts)
        return self.events

    # ── Summary stats per period ──────────────────────────────────────────────
    def period_summary(self):
        """Returns dict of period -> {events, critical, warning, info}"""
        summary = defaultdict(lambda: {"events":0,"critical":0,"warning":0,"info":0})
        for e in self.events:
            p = e.period
            summary[p]["events"] += 1
            summary[p][e.severity.lower()] = summary[p].get(e.severity.lower(), 0) + 1
        return dict(summary)

    def top_correlations(self, cols, n=12):
        valid = [c for c in cols if c in self.numeric_cols]
        if len(valid) < 2: return []
        try:
            df_num = self.df[valid].copy()
            for c in valid:
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
            corr = df_num.dropna().corr()
        except Exception:
            return []
        pairs, seen = [], set()
        for i,c1 in enumerate(corr.columns):
            for j,c2 in enumerate(corr.columns):
                if i>=j: continue
                if (c1,c2) in seen: continue
                seen.add((c1,c2))
                v = corr.loc[c1,c2]
                if not np.isnan(v): pairs.append((c1,c2,v))
        pairs.sort(key=lambda x:abs(x[2]),reverse=True)
        return pairs[:n]


# ═════════════════════════════════════════════════════════════════════════════
#  MULTI-MATCH STORE
# ═════════════════════════════════════════════════════════════════════════════

class MatchStore:
    """Holds multiple loaded matches."""
    def __init__(self):
        self.matches = []   # list of FRCAnalyzer

    def add(self, analyzer):
        self.matches.append(analyzer)

    def remove(self, idx):
        if 0 <= idx < len(self.matches):
            self.matches.pop(idx)

    def clear(self):
        self.matches.clear()

    def all_events(self):
        evts = []
        for m in self.matches:
            evts.extend(m.events)
        return sorted(evts, key=lambda e: (e.match_label, e.time))

    def new_errors_vs_previous(self):
        """
        Compare consecutive matches. Return dict of match_label -> list of event descriptions
        that did NOT appear in the previous match.
        """
        if len(self.matches) < 2:
            return {}
        result = {}
        for i in range(1, len(self.matches)):
            prev_descs = set(e.description for e in self.matches[i-1].events)
            curr        = self.matches[i]
            new_evts    = [e for e in curr.events if e.description not in prev_descs]
            if new_evts:
                result[curr.label] = new_evts
        return result

    def event_counts_by_match(self):
        """Returns list of (label, total, critical, warning, info) per match."""
        rows = []
        for m in self.matches:
            total = len(m.events)
            crit  = sum(1 for e in m.events if e.severity=="CRITICAL")
            warn  = sum(1 for e in m.events if e.severity=="WARNING")
            info  = sum(1 for e in m.events if e.severity=="INFO")
            rows.append((m.label, total, crit, warn, info))
        return rows


# ═════════════════════════════════════════════════════════════════════════════
#  WIDGET HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def mk_btn(parent, text, cmd=None, color=AC_CYAN, **kw):
    b = tk.Button(parent, text=text, command=cmd, bg=BG_CARD2, fg=color,
                  relief="flat", activebackground=color, activeforeground=BG_DARK,
                  font=("Courier New",10,"bold"), cursor="hand2",
                  padx=12, pady=5, bd=0, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=color,    fg=BG_DARK))
    b.bind("<Leave>", lambda e: b.config(bg=BG_CARD2, fg=color))
    return b

def mk_lbl(parent, text, size=9, color=TEXT_PRIM, bold=False, bg=BG_PANEL):
    return tk.Label(parent, text=text, bg=bg, fg=color,
                    font=("Courier New",size,"bold" if bold else "normal"))

def sec_hdr(parent, text):
    f = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(f, text=text, bg=BG_PANEL, fg=AC_CYAN,
             font=("Courier New",10,"bold")).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8,0))
    return f

def embed_fig(fig, frame, toolbar=False):
    canvas = FigureCanvasTkAgg(fig, master=frame); canvas.draw()
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, frame)
        tb.configure(bg=BG_PANEL); tb.update()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

class FRCApp(tk.Tk):
    SIGNAL_FIELDS = [
        ("robot_pose", "Robot pose/odometry signal"),
        ("robot_velocity", "Robot velocity signal"),
        ("vision_has_target", "Vision has-target signal"),
        ("enabled_state", "Robot enabled signal"),
        ("battery_voltage", "Battery voltage signal"),
        ("can_utilization", "CAN utilization signal"),
        ("loop_time", "Loop time signal"),
        ("pneumatic_pressure", "Pneumatic pressure signal"),
        ("compressor_state", "Compressor state signal"),
        ("gyro_heading", "Gyro heading signal"),
    ]
    CATEGORY_FIELDS = [
        "battery", "current", "drive", "can", "intake", "shooter", "climber", "arm",
        "pneumatics", "vision", "match", "temperature", "loop", "pdp", "gyro", "other"
    ]

    def __init__(self):
        super().__init__()
        self.title("Pitboss  ◈  FRC Match Analyzer")
        self.geometry("1520x940"); self.configure(bg=BG_DARK); self.minsize(1100,700)

        self.store         = MatchStore()
        self.active_idx    = None   # which match is shown in single-match tabs
        self.selected_cols = []
        self._all_events_for_list = []
        self._detection_running = False
        self.key_config_path = os.path.join(os.path.dirname(__file__), "frc_key_config.json")
        self.key_config = self._load_key_config()
        self._signal_vars = {}
        self._signal_combos = {}
        self._vision_add_var = tk.StringVar(value="")
        self._selected_key_var = tk.StringVar(value="No key selected")
        self._selected_key_category_var = tk.StringVar(value="drive")
        self._loading_win = None
        self._loading_var = tk.StringVar(value="Starting Pitboss...")
        self._icon_path = self._find_icon_path()

        self._set_app_icon()
        self._show_loading_screen()
        self._set_loading_message("Building interface...")

        self._build_ui()
        self._set_loading_message("Applying styles...")
        self._style_ttk()
        self._set_loading_message("Ready")
        self.after(120, self._close_loading_screen)

    # ── TTK theme ─────────────────────────────────────────────────────────────
    def _style_ttk(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TNotebook", background=BG_DARK, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_SEC,
                    font=("Courier New",10,"bold"), padding=[14,6], borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected",BG_PANEL)], foreground=[("selected",AC_CYAN)])
        s.configure("Treeview", background=BG_CARD, fieldbackground=BG_CARD,
                    foreground=TEXT_PRIM, font=("Courier New",9), rowheight=26)
        s.configure("Treeview.Heading", background=BG_CARD2, foreground=AC_CYAN,
                    font=("Courier New",9,"bold"))
        s.map("Treeview", background=[("selected",AC_CYAN)], foreground=[("selected",BG_DARK)])
        for sb in ("Vertical.TScrollbar","Horizontal.TScrollbar"):
            s.configure(sb, background=BG_CARD2, troughcolor=BG_DARK, arrowcolor=AC_CYAN)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self, bg=BG_PANEL, height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="⬡  PITBOSS", bg=BG_PANEL, fg=AC_CYAN,
                 font=("Courier New",14,"bold")).pack(side="left", padx=20, pady=12)
        self.status_var = tk.StringVar(value="Load a CSV to begin")
        tk.Label(top, textvariable=self.status_var, bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Courier New",9)).pack(side="right", padx=20)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        paned = tk.PanedWindow(self, orient="horizontal", bg=BG_DARK, sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(paned, bg=BG_PANEL, width=310)
        paned.add(self.sidebar, minsize=260)
        self.content = tk.Frame(paned, bg=BG_DARK)
        paned.add(self.content, minsize=700)
        self._build_sidebar()
        self._build_tabs()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        # File management
        sec_hdr(self.sidebar," MATCH FILES").pack(fill="x",padx=14,pady=(10,3))
        mk_btn(self.sidebar,"▲  Add CSV Match",cmd=self.load_csv,
               color=AC_CYAN).pack(fill="x",padx=14,pady=2)

        # Match list
        lf = tk.Frame(self.sidebar,bg=BG_CARD)
        lf.pack(fill="x",padx=14,pady=4)
        ysb = tk.Scrollbar(lf,bg=BG_CARD2,troughcolor=BG_DARK,bd=0)
        ysb.pack(side="right",fill="y")
        self.match_lb = tk.Listbox(lf,bg=BG_CARD,fg=TEXT_PRIM,
                                    selectbackground=AC_CYAN,selectforeground=BG_DARK,
                                    font=("Courier New",8),relief="flat",bd=0,
                                    activestyle="none",yscrollcommand=ysb.set,height=6)
        self.match_lb.pack(side="left",fill="both",expand=True)
        ysb.config(command=self.match_lb.yview)
        self.match_lb.bind("<<ListboxSelect>>", self._on_match_select)

        bf = tk.Frame(self.sidebar,bg=BG_PANEL); bf.pack(fill="x",padx=14,pady=2)
        mk_btn(bf,"✕ Remove",cmd=self.remove_match,color=AC_RED).pack(side="left",expand=True,fill="x",padx=(0,2))
        mk_btn(bf,"✕✕ Clear All",cmd=self.clear_matches,color=AC_RED).pack(side="left",expand=True,fill="x",padx=(2,0))

        # Stats
        sf = tk.Frame(self.sidebar,bg=BG_CARD); sf.pack(fill="x",padx=14,pady=4)
        self._stat = {}
        for lbl in ["Rows","Columns","Duration","Period","Events"]:
            r = tk.Frame(sf,bg=BG_CARD); r.pack(fill="x",padx=8,pady=1)
            tk.Label(r,text=lbl,bg=BG_CARD,fg=TEXT_SEC,font=("Courier New",8)).pack(side="left")
            v = tk.Label(r,text="—",bg=BG_CARD,fg=AC_CYAN,font=("Courier New",8,"bold"))
            v.pack(side="right"); self._stat[lbl] = v

        # Detect button
        tk.Frame(self.sidebar,bg=BORDER,height=1).pack(fill="x",padx=14,pady=4)
        self.detect_btn = mk_btn(self.sidebar,"⚡ Detect All Events",cmd=self.run_event_detection,
               color=AC_WARN)
        self.detect_btn.pack(fill="x",padx=14,pady=2)
        self.detect_all_btn = mk_btn(self.sidebar,"⚡ Detect ALL Matches",cmd=self.run_event_detection_all,
               color=AC_BLUE)
        self.detect_all_btn.pack(fill="x",padx=14,pady=(2,4))

        # Column selector
        tk.Frame(self.sidebar,bg=BORDER,height=1).pack(fill="x",padx=14,pady=4)
        sec_hdr(self.sidebar," COLUMN SELECTOR").pack(fill="x",padx=14)
        self.col_filter = tk.Entry(self.sidebar,bg=BG_CARD2,fg=TEXT_SEC,
                                    insertbackground=AC_CYAN,
                                    font=("Courier New",9),relief="flat",bd=4)
        self.col_filter.insert(0,"Filter columns...")
        self.col_filter.bind("<FocusIn>",self._on_col_filter_focus_in)
        self.col_filter.bind("<FocusOut>",self._on_col_filter_focus_out)
        self.col_filter.pack(fill="x",padx=14,pady=4)
        self.col_filter.bind("<KeyRelease>",self._filter_cols)
        lb_wrap = tk.Frame(self.sidebar,bg=BG_CARD)
        lb_wrap.pack(fill="both",expand=True,padx=14,pady=(0,4))
        ysb2 = tk.Scrollbar(lb_wrap,bg=BG_CARD2,troughcolor=BG_DARK,bd=0)
        ysb2.pack(side="right",fill="y")
        self.col_lb = tk.Listbox(lb_wrap,selectmode="extended",
                                  bg=BG_CARD,fg=TEXT_PRIM,
                                  selectbackground=AC_CYAN,selectforeground=BG_DARK,
                                  font=("Courier New",8),relief="flat",bd=0,
                                  activestyle="none",yscrollcommand=ysb2.set)
        self.col_lb.pack(side="left",fill="both",expand=True)
        ysb2.config(command=self.col_lb.yview)
        self._lb_cols = []
        mk_btn(self.sidebar,"✓  Apply Selection",cmd=self._apply_selection,
               color=AC_GREEN).pack(fill="x",padx=14,pady=(2,14))

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.nb = ttk.Notebook(self.content); self.nb.pack(fill="both",expand=True)
        # Per-match tabs
        self.t_timeline = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_timeline, text="  EVENT TIMELINE  ")
        self.t_events   = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_events,   text="  EVENT LIST  ")
        self.t_keymap   = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_keymap,   text="  KEY MAPPING  ")
        self.t_periods  = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_periods,  text="  PERIODS  ")
        self.t_ts       = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_ts,       text="  TIME SERIES  ")
        self.t_batt     = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_batt,     text="  BATTERY / CAN  ")
        self.t_data     = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_data,     text="  DATA PREVIEW  ")
        # Multi-match tabs
        self.t_multi    = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_multi,    text="  MULTI-MATCH  ")
        self.t_newbugs  = tk.Frame(self.nb,bg=BG_PANEL); self.nb.add(self.t_newbugs,  text="  NEW ERRORS  ")

        for t,msg in [(self.t_timeline,"Add a CSV and press ⚡ Detect All Events"),
                      (self.t_events,  "Events appear here after detection"),
                      (self.t_periods, "Period breakdown appears after detection"),
                      (self.t_data,    "Load a CSV to preview data")]:
            tk.Label(t,text=msg,bg=BG_PANEL,fg=TEXT_DIM,font=("Courier New",10)).pack(expand=True)

        self._build_ts_tab()
        self._build_batt_tab()
        self._build_keymap_tab()
        self._build_multi_tab()
        self._build_newbugs_tab()

    # ── Match management ──────────────────────────────────────────────────────
    def _active(self):
        if self.active_idx is not None and self.active_idx < len(self.store.matches):
            return self.store.matches[self.active_idx]
        if self.store.matches:
            return self.store.matches[-1]
        return None

    def _on_match_select(self, e=None):
        sel = self.match_lb.curselection()
        if sel:
            self.active_idx = sel[0]
            m = self._active()
            if m:
                self._update_sidebar_stats(m)
                self._fill_listbox(m)
                self._refresh_keymap_tab(m)

    def _update_sidebar_stats(self, m):
        dur = m._dur()
        n_evts = len(m.events)
        crits  = sum(1 for e in m.events if e.severity=="CRITICAL")
        warns  = sum(1 for e in m.events if e.severity=="WARNING")
        self._stat["Rows"].config(text=f"{len(m.df):,}")
        self._stat["Columns"].config(text=str(len(m.df.columns)))
        self._stat["Duration"].config(text=f"{dur:.1f}s")
        # Detect period types present
        periods_present = set(m.periods) if m.periods else {"?"}
        self._stat["Period"].config(text=", ".join(sorted(periods_present)))
        self._stat["Events"].config(
            text=f"{n_evts} ({crits}❗{warns}⚠)",
            fg=AC_RED if crits else AC_WARN if warns else AC_GREEN)

    def load_csv(self):
        paths = filedialog.askopenfilenames(title="Open WPILog CSV(s)",
            filetypes=[("CSV Files","*.csv"),("All Files","*.*")])
        if not paths: return
        loaded = 0
        for path in paths:
            try:
                df    = pd.read_csv(path)
                label = os.path.splitext(os.path.basename(path))[0]
                az    = FRCAnalyzer(df, label=label, key_config=self.key_config)
                self.store.add(az)
                self.match_lb.insert("end", f"  {label}")
                loaded += 1
            except Exception as ex:
                messagebox.showerror("Load Error", f"{path}:\n{ex}")
        if loaded:
            self.active_idx = len(self.store.matches) - 1
            self.match_lb.selection_clear(0, "end")
            self.match_lb.selection_set(self.active_idx)
            m = self._active()
            self._update_sidebar_stats(m)
            self._fill_listbox(m)
            self._populate_data_tab()
            self._refresh_keymap_tab(m)
            self.status_var.set(f"Loaded {loaded} match(es). Press ⚡ Detect All Events.")

    def remove_match(self):
        sel = self.match_lb.curselection()
        if not sel: return
        idx = sel[0]
        self.store.remove(idx)
        self.match_lb.delete(idx)
        self.active_idx = None
        self._refresh_keymap_tab(self._active())

    def clear_matches(self):
        self.store.clear()
        self.match_lb.delete(0,"end")
        self.active_idx = None
        self._refresh_keymap_tab(None)
        self.status_var.set("All matches cleared.")

    # ── Event detection ───────────────────────────────────────────────────────

    def run_event_detection(self):
        m = self._active()
        if not m:
            messagebox.showwarning("No Match","Load a CSV first."); return
        if self._detection_running:
            self.status_var.set("Detection already running...")
            return

        self._set_detection_running(True, f"Detecting events for {m.label}...")
        thread = threading.Thread(target=self._detect_worker, args=(m, False), daemon=True)
        thread.start()

    def run_event_detection_all(self):
        if not self.store.matches:
            messagebox.showwarning("No Matches","Load CSV files first."); return
        if self._detection_running:
            self.status_var.set("Detection already running...")
            return

        self._set_detection_running(True, f"Detecting all matches (0/{len(self.store.matches)})...")
        matches_snapshot = list(self.store.matches)
        thread = threading.Thread(target=self._detect_worker, args=(matches_snapshot, True), daemon=True)
        thread.start()

    def _set_detection_running(self, running, status_text=None):
        """Toggle detection UI state on the main thread."""
        self._detection_running = running
        btn_state = "disabled" if running else "normal"
        self.detect_btn.config(state=btn_state)
        if hasattr(self, "detect_all_btn"):
            self.detect_all_btn.config(state=btn_state)
        if status_text:
            self.status_var.set(status_text)

    def _detect_worker(self, target=None, detect_all=False):
        """Run detection in background thread."""
        try:
            if detect_all:
                total_evts = 0
                matches = target if isinstance(target, list) else []
                total_matches = len(matches)
                for idx, m in enumerate(matches, start=1):
                    evts = m.detect_all_events()
                    total_evts += len(evts)
                    self.after(0, self.status_var.set, f"Detecting all matches ({idx}/{total_matches})...")
                self.after(0, self._on_detection_complete_all, total_evts)
            else:
                single_match = target
                evts = single_match.detect_all_events()
                self.after(0, self._on_detection_complete_single, single_match, evts)
        except Exception as ex:
            self.after(0, self._on_detection_error, ex)

    def _on_detection_error(self, ex):
        self._set_detection_running(False)
        messagebox.showerror("Detection Error", f"{type(ex).__name__}: {ex}")

    def _on_detection_complete_single(self, m, evts):
        """Callback after single match detection (runs on main thread)."""
        crits = sum(1 for e in evts if e.severity=="CRITICAL")
        warns = sum(1 for e in evts if e.severity=="WARNING")
        self._update_sidebar_stats(m)
        self.status_var.set(f"✓ {m.label}: {len(evts)} events — {crits} critical, {warns} warnings")
        self.after(1, self._refresh_single_match_views, m)
        self._set_detection_running(False)

    def _refresh_single_match_views(self, m):
        """Build heavy views after status updates so UI remains responsive."""
        self._build_timeline_tab(m)
        self._build_event_list_tab(m)
        self._build_periods_tab(m)
        self.nb.select(self.t_timeline)

    def _on_detection_complete_all(self, total_evts):
        """Callback after all matches detection (runs on main thread)."""
        self.status_var.set(f"✓ All matches detected — {total_evts} total events across {len(self.store.matches)} matches")
        self.after(1, self._refresh_all_match_views)
        self._set_detection_running(False)

    def _refresh_all_match_views(self):
        """Build multi-match and active-match views in one scheduled UI task."""
        self._build_multi_tab_content()
        self._build_newbugs_content()
        m = self._active()
        if m and m.events:
            self._build_timeline_tab(m)
            self._build_event_list_tab(m)
            self._build_periods_tab(m)
        self.nb.select(self.t_multi)

    # ── EVENT TIMELINE ────────────────────────────────────────────────────────
    def _build_timeline_tab(self, m):
        for w in self.t_timeline.winfo_children(): w.destroy()
        evts = m.events
        dur  = m._dur()
        if not evts:
            tk.Label(self.t_timeline,text="No events detected.",
                     bg=BG_PANEL,fg=TEXT_DIM,font=("Courier New",10)).pack(expand=True); return

        cats_ordered = sorted(set(e.category for e in evts))
        n_lanes      = len(cats_ordered)
        cat_y        = {c:i for i,c in enumerate(cats_ordered)}

        fig_h = max(5, n_lanes * 0.75 + 3)
        fig   = Figure(figsize=(14, fig_h), facecolor=BG_DARK)
        ax    = fig.add_subplot(111); ax.set_facecolor(BG_CARD)

        # Period shading
        t_arr = m._t().values
        periods = m.periods
        if periods and len(periods) > 0 and len(t_arr) > 0:
            period_changes = []
            cur_p = periods[0]
            cur_start = float(t_arr[0])
            for i in range(1, len(periods)):
                p = periods[i]
                if p != cur_p:
                    period_changes.append((cur_start, float(t_arr[i-1]), cur_p))
                    cur_p = p
                    cur_start = float(t_arr[i])
            period_changes.append((cur_start, float(t_arr[len(t_arr)-1]), cur_p))
            for ps,pe,pp in period_changes:
                pc = PERIOD_COLOR.get(pp,"#445566")
                ax.axvspan(ps, pe, alpha=0.08, color=pc)
                ax.text((ps+pe)/2, n_lanes-0.1, pp.upper(), color=pc,
                        fontsize=7, ha="center", va="top", fontfamily="monospace", alpha=0.8)

        ax.set_xlim(0, dur)
        ax.set_ylim(-0.5, n_lanes - 0.5)
        ax.set_xlabel("Match Time (seconds)", color=TEXT_SEC, fontsize=9, fontfamily="monospace")
        ax.set_yticks(range(n_lanes))
        ax.set_yticklabels(cats_ordered, color=TEXT_SEC, fontsize=8, fontfamily="monospace")
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        # 15s grid
        for x in range(0, int(dur)+1, 15):
            ax.axvline(x, color=BORDER, lw=0.5, alpha=0.5)
            if 0 < x < dur - 3:
                ax.text(x, -0.48, f"{x}s", color=TEXT_DIM, fontsize=6,
                        ha="center", fontfamily="monospace")

        # Lane bg
        for i in range(n_lanes):
            bg = BG_CARD if i%2==0 else BG_CARD2
            ax.axhspan(i-0.45, i+0.45, color=bg, alpha=0.15)

        # Events
        for e in evts:
            y   = cat_y[e.category]
            col = SEV_COLOR.get(e.severity, AC_CYAN)
            if e.end_time is not None and (e.end_time - e.time) > 0.2:
                ax.barh(y, e.end_time - e.time, left=e.time, height=0.5,
                        color=col, alpha=0.45, edgecolor=col, linewidth=0.8)
            ax.scatter(e.time, y, color=col, s=50, zorder=5,
                       edgecolors="white", linewidths=0.5)

        legend_patches = [mpatches.Patch(color=v,label=k) for k,v in SEV_COLOR.items()]
        legend_patches += [mpatches.Patch(color=v,label=k,alpha=0.4) for k,v in PERIOD_COLOR.items() if k != "unknown"]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
                  facecolor=BG_CARD2, edgecolor=BORDER, labelcolor=TEXT_PRIM, ncol=2)
        crits = sum(1 for e in evts if e.severity=="CRITICAL")
        ax.set_title(f"{m.label} — {len(evts)} events detected  ({crits} CRITICAL)",
                     color=AC_CYAN, fontsize=10, fontfamily="monospace", pad=8)
        fig.tight_layout()
        embed_fig(fig, self.t_timeline, toolbar=True)

    # ── EVENT LIST ────────────────────────────────────────────────────────────
    def _build_event_list_tab(self, m):
        for w in self.t_events.winfo_children(): w.destroy()
        evts = m.events

        ctrl = tk.Frame(self.t_events,bg=BG_PANEL); ctrl.pack(fill="x",padx=10,pady=6)
        mk_lbl(ctrl,"Severity:").pack(side="left",padx=4)
        self._evt_sev = tk.StringVar(value="ALL")
        for sev,col in [("ALL",TEXT_SEC),("CRITICAL",AC_RED),("WARNING",AC_WARN),("INFO",AC_CYAN),("GOOD",AC_GREEN)]:
            tk.Radiobutton(ctrl,text=sev,variable=self._evt_sev,value=sev,
                           bg=BG_PANEL,fg=col,selectcolor=BG_CARD2,
                           activebackground=BG_PANEL,activeforeground=col,
                           font=("Courier New",9,"bold"),cursor="hand2",
                           command=self._filter_event_list).pack(side="left",padx=4)

        mk_lbl(ctrl,"  Cat:").pack(side="left",padx=(8,2))
        self._evt_cat = tk.StringVar(value="ALL")
        all_cats = ["ALL"] + sorted(set(e.category for e in evts))
        cat_dd = ttk.Combobox(ctrl,textvariable=self._evt_cat,values=all_cats,
                               font=("Courier New",9),state="readonly",width=12)
        cat_dd.pack(side="left")
        cat_dd.bind("<<ComboboxSelected>>",lambda e: self._filter_event_list())

        mk_lbl(ctrl,"  Period:").pack(side="left",padx=(8,2))
        self._evt_period = tk.StringVar(value="ALL")
        all_periods = ["ALL"] + sorted(set(e.period for e in evts))
        per_dd = ttk.Combobox(ctrl,textvariable=self._evt_period,values=all_periods,
                               font=("Courier New",9),state="readonly",width=10)
        per_dd.pack(side="left")
        per_dd.bind("<<ComboboxSelected>>",lambda e: self._filter_event_list())

        mk_btn(ctrl,"💾 Export CSV",cmd=self._export_events,color=AC_GREEN).pack(side="right",padx=8)

        cols = ("time","end_time","duration","period","severity","category","key","description","detail")
        frame = tk.Frame(self.t_events,bg=BG_PANEL)
        frame.pack(fill="both",expand=True,padx=10,pady=4)
        xsb = ttk.Scrollbar(frame,orient="horizontal")
        ysb = ttk.Scrollbar(frame,orient="vertical")
        self.evt_tree = ttk.Treeview(frame,columns=cols,show="headings",
                                      xscrollcommand=xsb.set,yscrollcommand=ysb.set)
        xsb.config(command=self.evt_tree.xview); xsb.pack(side="bottom",fill="x")
        ysb.config(command=self.evt_tree.yview); ysb.pack(side="right",fill="y")
        self.evt_tree.pack(fill="both",expand=True)

        widths = {"time":80,"end_time":80,"duration":65,"period":70,"severity":80,
                  "category":80,"key":200,"description":360,"detail":500}
        for c in cols:
            self.evt_tree.heading(c,text=c.upper(),command=lambda _c=c: self._sort_events(_c))
            self.evt_tree.column(c,width=widths.get(c,100),minwidth=40)
        self.evt_tree.tag_configure("CRITICAL",foreground=AC_RED)
        self.evt_tree.tag_configure("WARNING",foreground=AC_WARN)
        self.evt_tree.tag_configure("INFO",foreground=AC_CYAN)
        self.evt_tree.tag_configure("GOOD",foreground=AC_GREEN)

        self._all_events_for_list = evts
        self._populate_event_tree(evts)

        by_sev = defaultdict(int)
        for e in evts: by_sev[e.severity] += 1
        summary = "  ".join(f"{s}:{n}" for s,n in sorted(by_sev.items()))
        tk.Label(self.t_events,text=f"Total: {len(evts)}  |  {summary}",
                 bg=BG_PANEL,fg=TEXT_SEC,font=("Courier New",8)).pack(pady=3)

    def _populate_event_tree(self, evts):
        self.evt_tree.delete(*self.evt_tree.get_children())
        for e in evts:
            self.evt_tree.insert("","end",values=(
                f"{e.time:.2f}s",
                f"{e.end_time:.2f}s" if e.end_time else "—",
                e.dur_str() or "—",
                e.period,
                e.severity,
                e.category,
                e.key,
                e.description,
                e.detail
            ), tags=(e.severity,))

    def _filter_event_list(self):
        sev = self._evt_sev.get(); cat = self._evt_cat.get()
        per = getattr(self,"_evt_period",None)
        per = per.get() if per else "ALL"
        filtered = [e for e in self._all_events_for_list
                    if (sev=="ALL" or e.severity==sev)
                    and (cat=="ALL" or e.category==cat)
                    and (per=="ALL" or e.period==per)]
        self._populate_event_tree(filtered)

    def _sort_events(self, col):
        items = [(self.evt_tree.set(k,col),k) for k in self.evt_tree.get_children("")]
        try: items.sort(key=lambda x: float(str(x[0]).rstrip("s—")))
        except: items.sort()
        for idx,(_,k) in enumerate(items): self.evt_tree.move(k,"",idx)

    def _export_events(self):
        evts = self._all_events_for_list
        if not evts: return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")],title="Export Events")
        if not path: return
        rows = [{"match":e.match_label,"time_s":e.time,"end_time_s":e.end_time,
                 "duration_s":e.dur(),"period":e.period,"severity":e.severity,
                 "category":e.category,"key":e.key,
                 "description":e.description,"detail":e.detail} for e in evts]
        pd.DataFrame(rows).to_csv(path,index=False)
        messagebox.showinfo("Exported",f"Saved {len(rows)} events to:\n{path}")

    # ── PERIODS TAB ───────────────────────────────────────────────────────────
    def _build_periods_tab(self, m):
        for w in self.t_periods.winfo_children(): w.destroy()
        evts     = m.events
        dur      = m._dur()
        summary  = m.period_summary()
        t_arr    = m._t().values

        # Top summary cards
        cards_f = tk.Frame(self.t_periods,bg=BG_PANEL); cards_f.pack(fill="x",padx=10,pady=8)
        period_order = ["auto","teleop","endgame","disabled","unknown"]
        for p in period_order:
            if p not in summary and p not in set(m.periods): continue
            info = summary.get(p, {"events":0,"critical":0,"warning":0,"info":0})
            rows_in_p = [i for i,pp in enumerate(m.periods) if pp==p]
            t_in_p    = len(rows_in_p) * (dur / max(len(m.periods),1))
            col = PERIOD_COLOR.get(p, TEXT_DIM)
            card = tk.Frame(cards_f,bg=BG_CARD,padx=10,pady=8)
            card.pack(side="left",expand=True,fill="both",padx=4)
            tk.Label(card,text=p.upper(),bg=BG_CARD,fg=col,
                     font=("Courier New",11,"bold")).pack()
            tk.Label(card,text=f"~{t_in_p:.0f}s",bg=BG_CARD,fg=TEXT_SEC,
                     font=("Courier New",8)).pack()
            tk.Label(card,text=f"{info['events']} events",bg=BG_CARD,fg=TEXT_PRIM,
                     font=("Courier New",9)).pack(pady=2)
            for sev,sc in [("critical",AC_RED),("warning",AC_WARN),("info",AC_CYAN)]:
                cnt = info.get(sev,0)
                if cnt:
                    tk.Label(card,text=f"{cnt} {sev}",bg=BG_CARD,fg=sc,
                             font=("Courier New",8)).pack()

        # Events table grouped by period
        tk.Frame(self.t_periods,bg=BORDER,height=1).pack(fill="x",padx=10,pady=4)
        mk_lbl(self.t_periods,"EVENTS BY PERIOD:",10,AC_CYAN,bold=True).pack(anchor="w",padx=12)

        frame = tk.Frame(self.t_periods,bg=BG_PANEL)
        frame.pack(fill="both",expand=True,padx=10,pady=4)
        xsb = ttk.Scrollbar(frame,orient="horizontal")
        ysb = ttk.Scrollbar(frame,orient="vertical")
        cols_p = ("period","time","dur","sev","cat","key","description")
        tree = ttk.Treeview(frame,columns=cols_p,show="headings",
                             xscrollcommand=xsb.set,yscrollcommand=ysb.set)
        xsb.config(command=tree.xview); xsb.pack(side="bottom",fill="x")
        ysb.config(command=tree.yview); ysb.pack(side="right",fill="y")
        tree.pack(fill="both",expand=True)
        for c,w in [("period",70),("time",90),("dur",65),("sev",80),("cat",80),("key",200),("description",450)]:
            tree.heading(c,text=c.upper()); tree.column(c,width=w,minwidth=40)
        tree.tag_configure("CRITICAL",foreground=AC_RED)
        tree.tag_configure("WARNING",foreground=AC_WARN)
        tree.tag_configure("INFO",foreground=AC_CYAN)

        for p in period_order:
            p_evts = [e for e in evts if e.period==p]
            for e in p_evts:
                tree.insert("","end",values=(
                    e.period, f"{e.time:.2f}s", e.dur_str() or "—",
                    e.severity, e.category, e.key, e.description
                ), tags=(e.severity,))

    # ── MULTI-MATCH TAB ───────────────────────────────────────────────────────
    def _build_multi_tab(self):
        for w in self.t_multi.winfo_children(): w.destroy()
        tk.Label(self.t_multi,text="Load multiple matches and press ⚡ Detect ALL Matches",
                 bg=BG_PANEL,fg=TEXT_DIM,font=("Courier New",10)).pack(expand=True)

    def _build_multi_tab_content(self):
        for w in self.t_multi.winfo_children(): w.destroy()
        matches = self.store.matches
        if not matches: return

        mk_lbl(self.t_multi,"PERFORMANCE ACROSS MATCHES",11,AC_CYAN,bold=True).pack(pady=(10,4))

        # Bar chart: critical + warning events per match
        counts = self.store.event_counts_by_match()
        labels = [r[0] for r in counts]
        crits  = [r[2] for r in counts]
        warns  = [r[3] for r in counts]
        infos  = [r[4] for r in counts]
        x      = np.arange(len(labels))

        fig = Figure(figsize=(13, 4), facecolor=BG_DARK)
        ax  = fig.add_subplot(111); ax.set_facecolor(BG_CARD)
        w   = 0.25
        ax.bar(x - w, crits, w, color=AC_RED,  alpha=0.85, label="Critical")
        ax.bar(x,     warns, w, color=AC_WARN,  alpha=0.85, label="Warning")
        ax.bar(x + w, infos, w, color=AC_CYAN,  alpha=0.85, label="Info")
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha="right",
                                              fontsize=8,color=TEXT_SEC,fontfamily="monospace")
        ax.set_ylabel("Event Count",color=TEXT_SEC,fontsize=8,fontfamily="monospace")
        ax.set_title("Events per Match",color=AC_CYAN,fontsize=10,fontfamily="monospace")
        ax.tick_params(colors=TEXT_DIM,labelsize=8)
        ax.legend(fontsize=8,facecolor=BG_CARD2,edgecolor=BORDER,labelcolor=TEXT_PRIM)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True,axis="y",color=BORDER,lw=0.4,alpha=0.5)
        fig.tight_layout()
        embed_fig(fig, self.t_multi)

        # Table summary
        tk.Frame(self.t_multi,bg=BORDER,height=1).pack(fill="x",padx=10,pady=4)
        frame = tk.Frame(self.t_multi,bg=BG_PANEL)
        frame.pack(fill="both",expand=True,padx=10,pady=4)
        ysb = ttk.Scrollbar(frame,orient="vertical")
        tree_cols = ("match","total","critical","warning","info","top_issue")
        tree = ttk.Treeview(frame,columns=tree_cols,show="headings",
                             yscrollcommand=ysb.set,height=8)
        ysb.config(command=tree.yview); ysb.pack(side="right",fill="y")
        tree.pack(fill="both",expand=True)
        for c,w2 in [("match",200),("total",70),("critical",80),("warning",80),("info",60),("top_issue",500)]:
            tree.heading(c,text=c.upper()); tree.column(c,width=w2,minwidth=40)
        tree.tag_configure("bad",foreground=AC_RED)
        tree.tag_configure("ok",foreground=AC_GREEN)

        for m in matches:
            top = next((e.description for e in m.events if e.severity=="CRITICAL"),
                       next((e.description for e in m.events if e.severity=="WARNING"),"—"))
            crit_c = sum(1 for e in m.events if e.severity=="CRITICAL")
            tree.insert("","end",values=(
                m.label,len(m.events),crit_c,
                sum(1 for e in m.events if e.severity=="WARNING"),
                sum(1 for e in m.events if e.severity=="INFO"),
                top
            ), tags=("bad" if crit_c>0 else "ok",))

    # ── NEW ERRORS TAB ────────────────────────────────────────────────────────
    def _build_newbugs_tab(self):
        for w in self.t_newbugs.winfo_children(): w.destroy()
        tk.Label(self.t_newbugs,text="Load multiple matches and press ⚡ Detect ALL Matches",
                 bg=BG_PANEL,fg=TEXT_DIM,font=("Courier New",10)).pack(expand=True)

    def _build_newbugs_content(self):
        for w in self.t_newbugs.winfo_children(): w.destroy()
        new_errors = self.store.new_errors_vs_previous()

        mk_lbl(self.t_newbugs,"NEW ERRORS VS PREVIOUS MATCH",11,AC_CYAN,bold=True).pack(pady=(10,4))
        if not new_errors:
            mk_lbl(self.t_newbugs,"✓ No new error types compared to previous matches.",
                   10,AC_GREEN).pack(pady=20); return

        frame = tk.Frame(self.t_newbugs,bg=BG_PANEL)
        frame.pack(fill="both",expand=True,padx=10,pady=4)
        ysb = ttk.Scrollbar(frame,orient="vertical")
        xsb = ttk.Scrollbar(frame,orient="horizontal")
        cols = ("match","period","severity","category","key","new_description","detail")
        tree = ttk.Treeview(frame,columns=cols,show="headings",
                             yscrollcommand=ysb.set,xscrollcommand=xsb.set)
        xsb.config(command=tree.xview); xsb.pack(side="bottom",fill="x")
        ysb.config(command=tree.yview); ysb.pack(side="right",fill="y")
        tree.pack(fill="both",expand=True)
        for c,w in [("match",150),("period",70),("severity",80),("category",80),
                     ("key",180),("new_description",360),("detail",500)]:
            tree.heading(c,text=c.upper()); tree.column(c,width=w,minwidth=40)
        tree.tag_configure("CRITICAL",foreground=AC_RED)
        tree.tag_configure("WARNING",foreground=AC_WARN)
        tree.tag_configure("INFO",foreground=AC_CYAN)

        for match_label, evts in new_errors.items():
            for e in evts:
                tree.insert("","end",values=(
                    match_label, e.period, e.severity, e.category,
                    e.key, e.description, e.detail
                ), tags=(e.severity,))

        total = sum(len(v) for v in new_errors.values())
        tk.Label(self.t_newbugs,text=f"Total new error types: {total}",
                 bg=BG_PANEL,fg=AC_WARN,font=("Courier New",8)).pack(pady=3)

    # ── DATA PREVIEW ──────────────────────────────────────────────────────────
    def _populate_data_tab(self):
        for w in self.t_data.winfo_children(): w.destroy()
        m = self._active()
        if not m: return
        df = m.df.head(400); cols = list(df.columns)
        f  = tk.Frame(self.t_data,bg=BG_PANEL); f.pack(fill="both",expand=True,padx=10,pady=10)
        xsb = ttk.Scrollbar(f,orient="horizontal"); ysb = ttk.Scrollbar(f,orient="vertical")
        tree= ttk.Treeview(f,columns=cols,show="headings",
                            xscrollcommand=xsb.set,yscrollcommand=ysb.set)
        xsb.config(command=tree.xview); xsb.pack(side="bottom",fill="x")
        ysb.config(command=tree.yview); ysb.pack(side="right",fill="y")
        tree.pack(fill="both",expand=True)
        for c in cols:
            tree.heading(c,text=c); tree.column(c,width=110,minwidth=50,anchor="center")
        for _,row in df.iterrows():
            tree.insert("","end",values=[str(v) if not pd.isna(v) else "—" for v in row])
        tk.Label(self.t_data,text=f"Showing first {len(df)} of {len(m.df)} rows | {m.label}",
                 bg=BG_PANEL,fg=TEXT_DIM,font=("Courier New",8)).pack(pady=2)

    # ── TIME SERIES ───────────────────────────────────────────────────────────
    def _build_ts_tab(self):
        ctrl=tk.Frame(self.t_ts,bg=BG_PANEL); ctrl.pack(fill="x",padx=10,pady=6)
        mk_lbl(ctrl,"Plot selected columns:").pack(side="left",padx=4)
        mk_btn(ctrl,"▶  Plot",cmd=self.plot_time_series,color=AC_CYAN).pack(side="left",padx=6)
        mk_btn(ctrl,"✕ Clear",
               cmd=lambda:[w.destroy() for w in self.ts_cf.winfo_children()],
               color=TEXT_SEC).pack(side="left")
        self.ts_event_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl,text="Overlay events",variable=self.ts_event_var,
                       bg=BG_PANEL,fg=TEXT_SEC,selectcolor=BG_CARD2,
                       activebackground=BG_PANEL,font=("Courier New",9)).pack(side="left",padx=8)
        self.ts_period_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl,text="Period shading",variable=self.ts_period_var,
                       bg=BG_PANEL,fg=TEXT_SEC,selectcolor=BG_CARD2,
                       activebackground=BG_PANEL,font=("Courier New",9)).pack(side="left",padx=4)
        self.ts_cf=tk.Frame(self.t_ts,bg=BG_DARK); self.ts_cf.pack(fill="both",expand=True)

    def plot_time_series(self):
        m = self._active()
        if not m: messagebox.showwarning("No Match","Load a CSV first."); return
        cols=[c for c in (self.selected_cols or m.numeric_cols[:4]) if c in m.numeric_cols]
        if not cols: messagebox.showwarning("No Columns","Select numeric columns."); return
        for w in self.ts_cf.winfo_children(): w.destroy()
        n=len(cols)
        fig=Figure(figsize=(14,max(3,n*2.5)),facecolor=BG_DARK)
        t=m._t(); evts=m.events if self.ts_event_var.get() else []
        t_arr = t.values

        for i,col in enumerate(cols):
            ax=fig.add_subplot(n,1,i+1); ax.set_facecolor(BG_CARD)
            cc=CHART_COLORS[i%len(CHART_COLORS)]
            s=pd.to_numeric(m.df[col],errors="coerce")
            ax.plot(t,s,color=cc,lw=1.0,alpha=0.9)
            ax.fill_between(t,s,alpha=0.1,color=cc)

            # Period shading
            if self.ts_period_var.get() and m.periods:
                prev_p = m.periods[0]; ps = float(t_arr[0])
                for ri,p in enumerate(m.periods):
                    if p != prev_p or ri == len(m.periods)-1:
                        pe = float(t_arr[min(ri,len(t_arr)-1)])
                        pc = PERIOD_COLOR.get(prev_p,"#445566")
                        ax.axvspan(ps, pe, alpha=0.07, color=pc)
                        prev_p = p; ps = float(t_arr[ri])

            # Event markers
            for e in evts:
                ec=SEV_COLOR.get(e.severity,AC_CYAN)
                ax.axvline(e.time,color=ec,lw=0.7,alpha=0.5,linestyle="--")
                if e.end_time: ax.axvspan(e.time,e.end_time,color=ec,alpha=0.07)

            ax.set_ylabel(col,color=TEXT_SEC,fontsize=7,fontfamily="monospace")
            ax.tick_params(colors=TEXT_DIM,labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True,color=BORDER,lw=0.4,alpha=0.5)
            if i<n-1: ax.set_xticklabels([])

        fig.tight_layout(rect=[0,0.02,1,1])
        embed_fig(fig,self.ts_cf,toolbar=True)

    # ── CORRELATIONS ──────────────────────────────────────────────────────────
    def _build_corr_tab(self):
        ctrl=tk.Frame(self.t_corr,bg=BG_PANEL); ctrl.pack(fill="x",padx=10,pady=6)
        mk_lbl(ctrl,"Correlation matrix (selected columns):").pack(side="left",padx=4)
        mk_btn(ctrl,"▶  Compute",cmd=self.plot_correlations,color=AC_CYAN).pack(side="left",padx=6)
        self.corr_cf=tk.Frame(self.t_corr,bg=BG_DARK); self.corr_cf.pack(fill="both",expand=True)
        self.corr_list=tk.Frame(self.t_corr,bg=BG_PANEL,height=110)
        self.corr_list.pack(fill="x",side="bottom",padx=10,pady=4)
        self.corr_list.pack_propagate(False)

    def plot_correlations(self):
        m = self._active()
        if not m: messagebox.showwarning("No Match","Load a CSV first."); return
        cols=[c for c in (self.selected_cols or m.numeric_cols[:14]) if c in m.numeric_cols]
        if len(cols)<2: messagebox.showwarning("Too Few","Select >= 2 numeric columns."); return
        for w in self.corr_cf.winfo_children(): w.destroy()
        for w in self.corr_list.winfo_children(): w.destroy()

        df_num = m.df[cols].copy()
        for c in cols: df_num[c] = pd.to_numeric(df_num[c],errors="coerce")
        corr = df_num.dropna().corr()

        cmap=mcolors.LinearSegmentedColormap.from_list("frc",[AC_RED,BG_CARD2,AC_CYAN])
        fig=Figure(figsize=(7,6),facecolor=BG_DARK)
        ax=fig.add_subplot(111); ax.set_facecolor(BG_CARD)
        im=ax.imshow(corr.values,cmap=cmap,vmin=-1,vmax=1,aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols,rotation=45,ha="right",fontsize=7,color=TEXT_SEC,fontfamily="monospace")
        ax.set_yticklabels(cols,fontsize=7,color=TEXT_SEC,fontfamily="monospace")
        ax.tick_params(colors=TEXT_DIM)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v=corr.values[i,j]
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,
                        color=BG_DARK if abs(v)>0.6 else TEXT_PRIM,fontfamily="monospace")
        cb=fig.colorbar(im,ax=ax,fraction=0.046)
        cb.ax.tick_params(labelsize=7,colors=TEXT_SEC); cb.outline.set_edgecolor(BORDER)
        fig.tight_layout(); embed_fig(fig,self.corr_cf)

        mk_lbl(self.corr_list,"TOP CORRELATIONS:",9,AC_CYAN,bold=True).pack(anchor="w",pady=(4,2))
        top=m.top_correlations(cols)
        rf=tk.Frame(self.corr_list,bg=BG_PANEL); rf.pack(fill="x")
        for idx,(c1,c2,v) in enumerate(top[:8]):
            cl=AC_GREEN if v>0.7 else AC_WARN if v>0.4 else AC_RED if v<-0.4 else TEXT_SEC
            sym="↑↑" if v>0.7 else "↑" if v>0.4 else "↓↓" if v<-0.7 else "↓" if v<-0.4 else "~"
            tk.Label(rf,text=f"  {sym}  {c1}  ↔  {c2}  →  r={v:.3f}",
                     bg=BG_PANEL,fg=cl,font=("Courier New",8)).grid(row=idx//4,column=idx%4,sticky="w",padx=8)

    # ── BATTERY / CAN ─────────────────────────────────────────────────────────
    def _build_batt_tab(self):
        ctrl=tk.Frame(self.t_batt,bg=BG_PANEL); ctrl.pack(fill="x",padx=10,pady=6)
        mk_btn(ctrl,"▶  Plot Battery & CAN",cmd=self.plot_battery_can,color=AC_CYAN).pack(side="left",padx=8)
        self.batt_cf=tk.Frame(self.t_batt,bg=BG_DARK); self.batt_cf.pack(fill="both",expand=True)
        self.batt_sum=scrolledtext.ScrolledText(self.t_batt,bg=BG_CARD,fg=TEXT_PRIM,
            font=("Courier New",8),relief="flat",height=7,insertbackground=AC_CYAN)
        self.batt_sum.pack(fill="x",side="bottom",padx=10,pady=6)

    def plot_battery_can(self):
        m = self._active()
        if not m: messagebox.showwarning("No Match","Load a CSV first."); return
        for w in self.batt_cf.winfo_children(): w.destroy()
        self.batt_sum.delete("1.0","end")
        cats=m.categorize_columns()
        groups=[(g,lb) for g,lb in [
            (cats.get("battery",[]),"Voltage (V)"),
            (cats.get("current",[]),"Current (A)"),
            (cats.get("can",[]),"CAN Util (%)"),
            (cats.get("temperature",[]),"Temp (°C)"),
        ] if g]
        if not groups:
            self.batt_sum.insert("end","No battery/current/CAN/temp columns detected.\n"
                f"Columns: {', '.join(m.numeric_cols[:30])}"); return
        ng=len(groups); fig=Figure(figsize=(14,ng*3),facecolor=BG_DARK)
        t=m._t()
        evts=m.events
        for i,(group,ylabel) in enumerate(groups):
            ax=fig.add_subplot(ng,1,i+1); ax.set_facecolor(BG_CARD)
            for j,col in enumerate(group[:6]):
                s=pd.to_numeric(m.df[col],errors="coerce")
                ax.plot(t,s,color=CHART_COLORS[j%len(CHART_COLORS)],lw=1.1,label=col,alpha=0.9)
            if "V)" in ylabel:
                ax.axhline(6.8,color=AC_RED,ls="--",lw=0.9,alpha=0.8,label="Brownout 6.8V")
                ax.axhline(11.5,color=AC_WARN,ls="--",lw=0.9,alpha=0.8,label="Low 11.5V")
                ax.axhspan(0,6.8,alpha=0.07,color=AC_RED)
            elif "CAN" in ylabel:
                ax.axhline(80,color=AC_WARN,ls="--",lw=0.9,alpha=0.8,label="Warn 80%")
                ax.axhline(95,color=AC_RED,ls="--",lw=0.9,alpha=0.8,label="Critical 95%")
            elif "°C" in ylabel:
                ax.axhline(60,color=AC_WARN,ls="--",lw=0.9,alpha=0.8,label="Warn 60°C")
                ax.axhline(80,color=AC_RED,ls="--",lw=0.9,alpha=0.8,label="Critical 80°C")
            for e in evts:
                ec=SEV_COLOR.get(e.severity,AC_CYAN)
                ax.axvline(e.time,color=ec,lw=0.5,alpha=0.35,linestyle=":")
            # Period shading
            if m.periods:
                t_arr=t.values; prev_p=m.periods[0]; ps=float(t_arr[0])
                for ri,p in enumerate(m.periods):
                    if p != prev_p or ri == len(m.periods)-1:
                        pe=float(t_arr[min(ri,len(t_arr)-1)])
                        pc=PERIOD_COLOR.get(prev_p,"#445566")
                        ax.axvspan(ps,pe,alpha=0.06,color=pc)
                        prev_p=p; ps=float(t_arr[ri])
            ax.set_ylabel(ylabel,color=TEXT_SEC,fontsize=8,fontfamily="monospace")
            ax.legend(fontsize=7,facecolor=BG_CARD2,edgecolor=BORDER,labelcolor=TEXT_PRIM,loc="upper right")
            ax.tick_params(colors=TEXT_DIM,labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True,color=BORDER,lw=0.4,alpha=0.5)
        fig.tight_layout(); embed_fig(fig,self.batt_cf,toolbar=True)
        summary=["BATTERY / POWER / CAN EVENTS","─"*55]
        for e in evts:
            if e.category in ("battery","current","can"):
                summary.append(f"  [{e.severity}][{e.period}] {e.time_str()}  Key:{e.key}  {e.description}")
        self.batt_sum.insert("end","\n".join(summary))

    # ── UTILS ─────────────────────────────────────────────────────────────────
    def _find_icon_path(self):
        """Find first .ico file in project root."""
        try:
            base = os.path.dirname(__file__)
            for name in os.listdir(base):
                if name.lower().endswith(".ico"):
                    return os.path.join(base, name)
        except Exception:
            pass
        return None

    def _set_app_icon(self):
        """Set Windows app icon if an .ico exists."""
        if not self._icon_path:
            return
        try:
            self.iconbitmap(self._icon_path)
        except Exception:
            # Keep app running even if icon loading fails.
            pass

    def _show_loading_screen(self):
        """Simple splash while UI initializes."""
        self._loading_win = tk.Toplevel(self)
        self._loading_win.overrideredirect(True)
        self._loading_win.configure(bg=BG_PANEL)
        self._loading_win.attributes("-topmost", True)
        if self._icon_path:
            try:
                self._loading_win.iconbitmap(self._icon_path)
            except Exception:
                pass

        w, h = 500, 220
        self.update_idletasks()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self._loading_win.geometry(f"{w}x{h}+{x}+{y}")

        card = tk.Frame(self._loading_win, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=10, pady=10)
        tk.Label(card, text="⬡ PITBOSS", bg=BG_CARD, fg=AC_CYAN,
                 font=("Courier New", 22, "bold")).pack(pady=(34, 8))
        tk.Label(card, text="FRC Match Analyzer", bg=BG_CARD, fg=TEXT_SEC,
                 font=("Courier New", 10)).pack()
        tk.Label(card, textvariable=self._loading_var, bg=BG_CARD, fg=AC_GREEN,
                 font=("Courier New", 9, "bold")).pack(pady=(20, 0))
        self._loading_win.update_idletasks()

    def _set_loading_message(self, text):
        if self._loading_win and self._loading_win.winfo_exists():
            self._loading_var.set(text)
            self._loading_win.update_idletasks()

    def _close_loading_screen(self):
        if self._loading_win and self._loading_win.winfo_exists():
            self._loading_win.destroy()
        self._loading_win = None

    def _load_key_config(self):
        default_cfg = {"signal_map": {}, "column_category": {}}
        try:
            if not os.path.exists(self.key_config_path):
                return default_cfg
            with open(self.key_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return default_cfg
            data.setdefault("signal_map", {})
            data.setdefault("column_category", {})
            return data
        except Exception:
            return default_cfg

    def _save_key_config(self):
        try:
            with open(self.key_config_path, "w", encoding="utf-8") as f:
                json.dump(self.key_config, f, indent=2)
            self.status_var.set(f"Saved key mapping to {os.path.basename(self.key_config_path)}")
        except Exception as ex:
            messagebox.showerror("Save Error", f"Could not save key config:\n{ex}")

    def _apply_key_config_to_all_matches(self):
        for m in self.store.matches:
            m.update_key_config(self.key_config)

    def _build_keymap_tab(self):
        for w in self.t_keymap.winfo_children():
            w.destroy()

        container = tk.Frame(self.t_keymap, bg=BG_PANEL)
        container.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(container, bg=BG_CARD)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(container, bg=BG_CARD2)
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        mk_lbl(left, "KEYS IN ACTIVE MATCH", 10, AC_CYAN, bold=True, bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        self.keymap_list = tk.Listbox(
            left, bg=BG_CARD, fg=TEXT_PRIM, selectbackground=AC_CYAN, selectforeground=BG_DARK,
            font=("Courier New", 8), relief="flat", bd=0, activestyle="none"
        )
        self.keymap_list.pack(fill="both", expand=True, padx=8, pady=6)
        self.keymap_list.bind("<<ListboxSelect>>", self._on_keymap_key_selected)

        ctrl = tk.Frame(left, bg=BG_CARD)
        ctrl.pack(fill="x", padx=8, pady=(0, 8))
        mk_lbl(ctrl, "Category:", 9, TEXT_SEC, bg=BG_CARD).pack(side="left")
        self.key_category_combo = ttk.Combobox(
            ctrl, values=self.CATEGORY_FIELDS, state="readonly", width=16, textvariable=self._selected_key_category_var
        )
        self.key_category_combo.pack(side="left", padx=6)
        mk_btn(ctrl, "Assign Category", cmd=self._assign_selected_key_category, color=AC_GREEN).pack(side="left", padx=4)
        mk_btn(ctrl, "Clear Category", cmd=self._clear_selected_key_category, color=AC_WARN).pack(side="left", padx=4)

        mk_lbl(right, "SIGNAL-TO-KEY MAPPING", 10, AC_CYAN, bold=True, bg=BG_CARD2).pack(anchor="w", padx=8, pady=(8, 4))
        self._signal_vars.clear()
        self._signal_combos.clear()
        for signal_name, label in self.SIGNAL_FIELDS:
            if signal_name == "vision_has_target":
                continue
            row = tk.Frame(right, bg=BG_CARD2)
            row.pack(fill="x", padx=8, pady=3)
            mk_lbl(row, label, 8, TEXT_SEC, bg=BG_CARD2).pack(side="left")
            var = tk.StringVar(value="")
            combo = ttk.Combobox(row, textvariable=var, values=[""], state="readonly", width=38)
            combo.pack(side="right")
            self._signal_vars[signal_name] = var
            self._signal_combos[signal_name] = combo

        vis_hdr = tk.Frame(right, bg=BG_CARD2)
        vis_hdr.pack(fill="x", padx=8, pady=(8, 2))
        mk_lbl(vis_hdr, "Vision has-target signals (multiple cameras):", 8, TEXT_SEC, bg=BG_CARD2).pack(side="left")

        vis_add = tk.Frame(right, bg=BG_CARD2)
        vis_add.pack(fill="x", padx=8, pady=(0, 4))
        self.vision_add_combo = ttk.Combobox(vis_add, textvariable=self._vision_add_var, values=[""], state="readonly", width=38)
        self.vision_add_combo.pack(side="left", padx=(0, 6))
        mk_btn(vis_add, "Add Camera", cmd=self._add_vision_camera_key, color=AC_GREEN).pack(side="left")
        mk_btn(vis_add, "Remove Selected", cmd=self._remove_vision_camera_key, color=AC_WARN).pack(side="left", padx=6)

        self.vision_key_list = tk.Listbox(
            right, selectmode="extended", bg=BG_CARD, fg=TEXT_PRIM,
            selectbackground=AC_CYAN, selectforeground=BG_DARK,
            font=("Courier New", 8), relief="flat", bd=0, height=6
        )
        self.vision_key_list.pack(fill="x", padx=8, pady=(0, 6))

        btn_row = tk.Frame(right, bg=BG_CARD2)
        btn_row.pack(fill="x", padx=8, pady=8)
        mk_btn(btn_row, "Apply Mappings", cmd=self._apply_signal_mappings, color=AC_GREEN).pack(side="left", padx=3)
        mk_btn(btn_row, "Auto-Guess", cmd=self._autoguess_signal_mappings, color=AC_BLUE).pack(side="left", padx=3)
        mk_btn(btn_row, "Save Config", cmd=self._save_key_config, color=AC_CYAN).pack(side="left", padx=3)

        self.keymap_summary = scrolledtext.ScrolledText(
            right, bg=BG_CARD, fg=TEXT_PRIM, font=("Courier New", 8), relief="flat", height=10, insertbackground=AC_CYAN
        )
        self.keymap_summary.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._refresh_keymap_tab(self._active())

    def _refresh_keymap_tab(self, m):
        if not hasattr(self, "keymap_list"):
            return
        cols = list(m.df.columns) if m is not None else []
        self.keymap_list.delete(0, "end")
        for c in cols:
            cat = self.key_config.get("column_category", {}).get(c, "auto")
            self.keymap_list.insert("end", f"[{cat}] {c}")

        combo_values = [""] + cols
        signal_map = self.key_config.get("signal_map", {})
        for signal_name, combo in self._signal_combos.items():
            combo.config(values=combo_values)
            current = signal_map.get(signal_name, "")
            self._signal_vars[signal_name].set(current if current in cols else "")

        self.vision_add_combo.config(values=combo_values)
        self._vision_add_var.set("")
        self.vision_key_list.delete(0, "end")
        vision_keys = signal_map.get("vision_has_target", [])
        if isinstance(vision_keys, str):
            vision_keys = [vision_keys]
        for vk in vision_keys:
            if vk in cols:
                self.vision_key_list.insert("end", vk)

        self._refresh_keymap_summary()

    def _refresh_keymap_summary(self):
        if not hasattr(self, "keymap_summary"):
            return
        self.keymap_summary.delete("1.0", "end")
        self.keymap_summary.insert("end", "Signal map:\n")
        for signal_name, label in self.SIGNAL_FIELDS:
            v = self.key_config.get("signal_map", {}).get(signal_name, "—")
            if isinstance(v, list):
                v = ", ".join(v) if v else "—"
            self.keymap_summary.insert("end", f"  {label}: {v}\n")
        self.keymap_summary.insert("end", "\nCategory overrides:\n")
        cat_map = self.key_config.get("column_category", {})
        if not cat_map:
            self.keymap_summary.insert("end", "  (none)\n")
            return
        for k, v in sorted(cat_map.items()):
            self.keymap_summary.insert("end", f"  {k} -> {v}\n")

    def _on_keymap_key_selected(self, e=None):
        m = self._active()
        if not m:
            return
        sel = self.keymap_list.curselection()
        if not sel:
            return
        raw = self.keymap_list.get(sel[0])
        col = raw.split("] ", 1)[1] if "] " in raw else raw
        self._selected_key_var.set(col)
        cat = self.key_config.get("column_category", {}).get(col, "drive")
        self._selected_key_category_var.set(cat if cat in self.CATEGORY_FIELDS else "drive")

    def _assign_selected_key_category(self):
        col = self._selected_key_var.get()
        if not col or col == "No key selected":
            messagebox.showinfo("Select Key", "Select a key from the list first.")
            return
        cat = self._selected_key_category_var.get()
        self.key_config.setdefault("column_category", {})[col] = cat
        self._apply_key_config_to_all_matches()
        self._refresh_keymap_tab(self._active())

    def _clear_selected_key_category(self):
        col = self._selected_key_var.get()
        if not col or col == "No key selected":
            return
        self.key_config.setdefault("column_category", {}).pop(col, None)
        self._apply_key_config_to_all_matches()
        self._refresh_keymap_tab(self._active())

    def _apply_signal_mappings(self):
        m = self._active()
        valid_cols = set(m.df.columns) if m is not None else set()
        sm = self.key_config.setdefault("signal_map", {})
        for signal_name, var in self._signal_vars.items():
            val = var.get().strip()
            if val and val in valid_cols:
                sm[signal_name] = val
            else:
                sm.pop(signal_name, None)
        vision_vals = [self.vision_key_list.get(i) for i in range(self.vision_key_list.size())]
        vision_vals = [v for v in vision_vals if v in valid_cols]
        if vision_vals:
            sm["vision_has_target"] = vision_vals
        else:
            sm.pop("vision_has_target", None)
        self._apply_key_config_to_all_matches()
        self._refresh_keymap_summary()
        self.status_var.set("Applied signal mappings. Re-run event detection.")

    def _autoguess_signal_mappings(self):
        m = self._active()
        if not m:
            return
        cols = list(m.df.columns)
        guess_rules = {
            "robot_pose": ["pose2d", "pose", "odometry"],
            "robot_velocity": ["chassis", "velocity", "speed", "mps"],
            "vision_has_target": ["hastarget", "tv", "detected", "validtarget"],
            "enabled_state": ["isenabled", "enabled", "robotenabled"],
            "battery_voltage": ["battery", "voltage", "vbus"],
            "can_utilization": ["canutilization", "canbus", "can_util"],
            "loop_time": ["looptime", "loopruntime", "cycletime"],
            "pneumatic_pressure": ["pressure", "airpressure"],
            "compressor_state": ["compressor"],
            "gyro_heading": ["heading", "yaw", "angle"],
        }
        sm = self.key_config.setdefault("signal_map", {})
        for signal_name, words in guess_rules.items():
            pick = next((c for c in cols if (not self._is_bad_guess_name(c)) and any(w in c.lower() for w in words)), None)
            if not pick:
                pick = next((c for c in cols if any(w in c.lower() for w in words)), None)
            if pick:
                sm[signal_name] = pick
        vision_candidates = [c for c in cols if any(w in c.lower() for w in guess_rules["vision_has_target"]) and not self._is_bad_guess_name(c)]
        if vision_candidates:
            sm["vision_has_target"] = vision_candidates[:6]
        self._refresh_keymap_tab(m)
        self._apply_key_config_to_all_matches()
        self.status_var.set("Auto-guessed mappings. Review before running detection.")

    def _add_vision_camera_key(self):
        val = self._vision_add_var.get().strip()
        if not val:
            return
        existing = [self.vision_key_list.get(i) for i in range(self.vision_key_list.size())]
        if val not in existing:
            self.vision_key_list.insert("end", val)
        self._vision_add_var.set("")

    def _remove_vision_camera_key(self):
        sel = list(self.vision_key_list.curselection())
        for idx in reversed(sel):
            self.vision_key_list.delete(idx)

    def _is_bad_guess_name(self, col):
        cl = col.lower()
        return any(k in cl for k in ["joystick", "controller", "pov", "button", "trigger"])

    def _fill_listbox(self, m):
        self.col_lb.delete(0,"end"); self._lb_cols=[]
        for cat,cols in sorted(m.categorize_columns().items()):
            for col in cols:
                self.col_lb.insert("end",f"[{cat[:4].upper()}] {col}")
                self._lb_cols.append(col)

    def _on_col_filter_focus_in(self, e=None):
        if self.col_filter.get() == "Filter columns...":
            self.col_filter.delete(0,"end")
            self.col_filter.config(fg=TEXT_PRIM)

    def _on_col_filter_focus_out(self, e=None):
        if self.col_filter.get() == "":
            self.col_filter.insert(0,"Filter columns...")
            self.col_filter.config(fg=TEXT_SEC)

    def _filter_cols(self, e=None):
        m = self._active()
        if not m: return
        q=self.col_filter.get().lower()
        if q == "filter columns...": q = ""
        self.col_lb.delete(0,"end"); self._lb_cols=[]
        for cat,cols in sorted(m.categorize_columns().items()):
            for col in cols:
                if q in col.lower():
                    self.col_lb.insert("end",f"[{cat[:4].upper()}] {col}")
                    self._lb_cols.append(col)

    def _apply_selection(self):
        idxs=self.col_lb.curselection()
        self.selected_cols=[self._lb_cols[i] for i in idxs if i<len(self._lb_cols)]
        self.status_var.set(f"Selected {len(self.selected_cols)} columns")


if __name__ == "__main__":
    app = FRCApp()
    app.mainloop()

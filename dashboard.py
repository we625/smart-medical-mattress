"""
dashboard.py – لوحة تحكم المفرش الطبي الذكي (Streamlit)

تصميم نهاري (Light Mode) بأسلوب طبي نظيف يضم:
  • بطاقة الملف الصحي للمريض الافتراضي والنمط الطبيعي المتعلّم بالذكاء الاصطناعي
  • شريط التحكم بالسيناريوهات السبعة + وحدة التحكم بالتهوية والتبريد والتدفئة الآمنة
  • صف بطاقات لملخص القراءة الأخيرة (الحرارة، الرطوبة، أقصى ضغط، حالة التنبيه)
  • تبويبان منظمان: عرض 3D تفاعلي للطبقات الخمس والمريض ووحدة التحكم / خريطة الضغط 2D
  • منحنيات الحرارة والرطوبة وسجل التنبيهات وتوصيات التمريض
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Literal

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from mattress_3d import build_mattress_3d_html
from model import SmartMattressModel
from simulator import (
    COLS,
    ROWS,
    SCENARIO_CATALOG,
    SCENARIO_ORDER,
    ScenarioType,
    generate_reading,
    get_cell_region,
)

# ──────────────────── ثوابت التصميم ────────────────────

PRIMARY = "#0284C7"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
GRID = "#F1F5F9"
FONT_FAMILY = "Cairo, Tajawal, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

PRESSURE_COLORSCALE = [
    [0.00, "#F0F9FF"],
    [0.16, "#E0F2FE"],
    [0.34, "#7DD3FC"],
    [0.48, "#FEF08A"],
    [0.62, "#FACC15"],
    [0.76, "#F97316"],
    [0.90, "#DC2626"],
    [1.00, "#991B1B"],
]
PRESSURE_MAX = 100

TEMP_NORMAL_BAND = (33.2, 34.8)
HUM_NORMAL_BAND = (49.0, 61.0)

HvacControlMode = Literal["auto", "cool", "heat", "off"]
HVAC_LABELS: dict[HvacControlMode, str] = {
    "auto": "تلقائي",
    "cool": "تهوية وتبريد آمن",
    "heat": "تدفئة آمنة",
    "off": "إيقاف",
}
HVAC_ORDER: list[HvacControlMode] = ["auto", "cool", "heat", "off"]

ROW_LABELS = [
    "رأس",             # صف 1
    "رأس",             # صف 2
    "كتفان",           # صف 3
    "كتفان · ذراع",    # صف 4
    "ظهر · ذراع",      # صف 5
    "ظهر · ذراع",      # صف 6
    "حوض · يد",        # صف 7
    "حوض · يد",        # صف 8
    "حوض · يد",        # صف 9
    "حوض",             # صف 10
    "فخذان",           # صف 11
    "فخذان",           # صف 12
    "فخذان",           # صف 13
    "ساقان",           # صف 14
    "كعبان وقدمان",    # صف 15
    "كعبان وقدمان",    # صف 16
]
COL_LABELS = ["١", "٢", "٣", "٤", "٥", "٦", "٧", "٨"]


def _num(value) -> str:
    """يعزل الرقم باتجاه LTR كي تظهر إشارة السالب والأرقام العشرية في مكانها الصحيح."""
    return f'<span class="ltr-num">{value}</span>'


# ──────────────────── إعداد الصفحة ────────────────────

st.set_page_config(
    page_title="المفرش الطبي الذكي",
    page_icon=":material/monitor_heart:",
    layout="wide",
)


def _html(markup: str) -> None:
    """يعرض HTML مضغوطًا (بدون مسافات بادئة كي لا يفسّره Markdown ككتلة شيفرة)."""
    compact = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700&display=swap');

:root {
  --primary: #0284C7;
  --primary-dark: #0369A1;
  --primary-soft: #F0F9FF;
  --text: #0F172A;
  --muted: #64748B;
  --border: #E2E8F0;
  --card: #FFFFFF;
  --bg: #F8FAFC;
  --radius: 16px;
  --shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 20px rgba(15, 23, 42, 0.05);
}

html, body, .stApp { background: var(--bg); }
.stApp { direction: rtl; text-align: right; color: var(--text); }

.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-baseweb="select"], [data-baseweb="popover"] li,
.stApp [data-baseweb="tab"] p,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stSliderThumbValue"],
.stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] {
  font-family: 'Cairo', 'Tajawal', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

[data-baseweb="popover"] ul { direction: rtl; text-align: right; }

header[data-testid="stHeader"] { background: transparent; }
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1360px;
}

/* البطاقات: كل حاوية مفتاحها يبدأ بـ card_ */
[class*="st-key-card_"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 22px;
}

/* القائمة المنسدلة */
[data-testid="stSelectbox"] label p {
  color: var(--muted); font-size: 13px; font-weight: 600;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  min-height: 44px !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
  border-color: var(--primary) !important;
}

/* الأزرار */
.stButton > button { border-radius: 12px; min-height: 42px; font-weight: 600; transition: all .15s ease; padding: 4px 10px; }
.stButton > button p { font-weight: 600; font-size: 13.5px; }
[data-testid="stBaseButton-primary"] {
  background: var(--primary) !important; border: 1px solid var(--primary) !important; color: #FFFFFF !important;
  box-shadow: 0 4px 12px rgba(2, 132, 199, 0.22);
}
[data-testid="stBaseButton-primary"]:hover { background: var(--primary-dark) !important; border-color: var(--primary-dark) !important; }
[data-testid="stBaseButton-secondary"] {
  background: #FFFFFF !important; border: 1px solid var(--border) !important; color: var(--text) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--primary) !important; color: var(--primary) !important; background: var(--primary-soft) !important;
}

/* شريط التمرير */
[data-testid="stSlider"] [data-baseweb="slider"] { direction: ltr; }
[data-testid="stSlider"] label p { color: var(--muted); font-size: 13px; font-weight: 600; }

/* التبويبات */
[data-baseweb="tab-list"] {
  gap: 8px; background: #F1F5F9; padding: 5px; border-radius: 12px; border-bottom: none !important;
}
[data-baseweb="tab"] {
  border-radius: 9px !important; padding: 7px 18px !important; font-weight: 700 !important;
  color: var(--muted) !important; border: none !important; background: transparent !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  background: #FFFFFF !important; color: var(--primary) !important;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-panel"] { padding-top: 12px !important; }

/* الترويسة */
.app-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.brand { display: flex; align-items: center; gap: 14px; }
.brand-icon {
  width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #0EA5E9, #0284C7);
  display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(2, 132, 199, 0.28);
}
.app-title { font-size: 28px; font-weight: 800; color: var(--text); line-height: 1.25; letter-spacing: -0.2px; }
.app-subtitle { font-size: 14px; color: var(--muted); margin-top: 2px; }
.header-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px;
  font-size: 12.5px; font-weight: 600; border: 1px solid var(--border); background: #FFFFFF; color: var(--muted);
}
.badge-tone-normal { color: #047857; background: #ECFDF5; border-color: #A7F3D0; }
.badge-tone-warn { color: #B45309; background: #FFFBEB; border-color: #FDE68A; }
.badge-tone-cold { color: #0369A1; background: #F0F9FF; border-color: #BAE6FD; }
.badge-tone-danger { color: #B91C1C; background: #FEF2F2; border-color: #FECACA; }
.badge-tone-critical { color: #991B1B; background: #FEE2E2; border-color: #FCA5A5; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }

/* بطاقة الملف الصحي للمريض الافتراضي */
.patient-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}
.patient-id { display: flex; align-items: center; gap: 12px; }
.patient-avatar {
  width: 40px; height: 40px; border-radius: 12px; background: #E0F2FE; color: #0284C7;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.patient-title { font-size: 14.5px; font-weight: 700; color: var(--text); }
.patient-sub { font-size: 12.5px; color: var(--muted); }
.baseline-chips { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.b-chip {
  padding: 5px 11px; border-radius: 10px; background: #F8FAFC; border: 1px solid var(--border);
  font-size: 12px; color: var(--muted);
}
.b-chip b { color: var(--text); font-weight: 700; }
.patient-note {
  margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  font-size: 12px; color: #0369A1;
}

.scn-desc-bar {
  margin-top: 8px; padding: 8px 12px; border-radius: 10px;
  background: #F8FAFC; border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  font-size: 12px; color: var(--muted);
}
.scn-desc-bar b { color: var(--text); font-weight: 600; }

.hvac-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  margin-bottom: 8px;
}
.hvac-title { font-size: 14.5px; font-weight: 700; color: var(--text); }

/* بطاقات الملخص */
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 4px 0 6px; }
@media (max-width: 960px) { .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.metric-card {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 18px 20px; min-height: 138px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.metric-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.metric-label { display: flex; align-items: center; gap: 8px; font-size: 13.5px; font-weight: 600; color: var(--muted); }
.metric-accent { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.metric-value { font-size: 31px; font-weight: 700; color: var(--text); margin-top: 6px; line-height: 1.2; }
.metric-unit { font-size: 14px; font-weight: 600; color: var(--muted); margin-inline-start: 6px; }
.metric-sub { font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height: 1.5; }
.delta { font-weight: 700; }
.delta.up { color: #B45309; }
.delta.down { color: #0369A1; }

.metric-card.status-ok { border-color: #A7F3D0; background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 70%); }
.metric-card.status-warn { border-color: #FDE68A; background: linear-gradient(180deg, #FFFBEB 0%, #FFFFFF 70%); }
.metric-card.status-danger { border-color: #FECACA; background: linear-gradient(180deg, #FEF2F2 0%, #FFFFFF 70%); }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.pill-ok { color: #047857; background: #D1FAE5; }
.pill-warn { color: #B45309; background: #FEF3C7; }
.pill-cool { color: #0369A1; background: #E0F2FE; }
.pill-heat { color: #C2410C; background: #FFEDD5; }
.pill-danger { color: #B91C1C; background: #FEE2E2; }
.pill-critical { color: #FFFFFF; background: #DC2626; }
.pill-region { color: #0369A1; background: #E0F2FE; font-weight: 600; }
.pill-neutral { color: var(--muted); background: #F1F5F9; }

/* عناوين الأقسام */
.section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
.section-title { font-size: 17px; font-weight: 700; color: var(--text); }
.section-sub { font-size: 12.5px; color: var(--muted); }
.zones-legend {
  margin-top: 6px; padding-top: 8px; border-top: 1px dashed var(--border);
  font-size: 11.5px; color: var(--muted); line-height: 1.7;
}

.ltr-num { direction: ltr; unicode-bidi: isolate; display: inline-block; }
.empty-state { text-align: center; color: var(--muted); padding: 48px 12px; font-size: 14px; }

/* سجل التنبيهات */
.alert-list { display: flex; flex-direction: column; gap: 12px; max-height: 440px; overflow-y: auto; padding-inline-end: 4px; margin-top: 10px; }
.alert-item {
  border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; background: #FFFFFF;
  border-inline-start: 4px solid var(--border);
  display: flex; flex-direction: column; gap: 8px;
}
.alert-item.sev-critical { border-inline-start-color: #991B1B; background: linear-gradient(90deg, #FFFFFF 0%, #FEF2F2 100%); }
.alert-item.sev-danger { border-inline-start-color: #DC2626; }
.alert-item.sev-warn { border-inline-start-color: #F59E0B; }
.alert-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.alert-badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.alert-cause { font-size: 14px; font-weight: 700; color: var(--text); line-height: 1.5; }
.alert-rec {
  padding: 8px 12px; border-radius: 9px; background: #F0F9FF; border: 1px solid #BAE6FD;
  color: #0C4A6E; font-size: 13px; line-height: 1.6;
}
.alert-rec b { color: #075985; font-weight: 700; }
.alert-meta { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.alert-meta b { color: var(--text); font-weight: 600; }

.disclaimer {
  margin-top: 18px; padding: 14px 18px; border-radius: var(--radius); border: 1px solid #BAE6FD;
  background: var(--primary-soft); color: #075985; font-size: 13px; line-height: 1.8;
}
</style>
"""
)

# ──────────────────── منطق التكييف السطحي وحالة الجلسة ────────────────────


def _resolve_effective_climate(
    hvac_mode: HvacControlMode,
    latest_reading,
    latest_result,
    latched_auto: str = "standby",
) -> Literal["standby", "cool", "heat", "off"]:
    """يحدّد حالة التكييف الفعلية للمفرش (سواء يدوية أو تلقائية مع ثبات حراري دون تذبذب)."""
    if hvac_mode == "off":
        return "off"
    if hvac_mode == "cool":
        return "cool"
    if hvac_mode == "heat":
        return "heat"

    # الوضع التلقائي (auto): يفحص القراءة الحالية أو يحافظ على الاستقرار النشط بعد التفعيل
    if latched_auto in ("cool", "heat"):
        return latched_auto  # يستمر بالتكييف لمنع التذبذب الحراري حتى يتغير السيناريو أو يُعاد الضبط
    if latest_reading is None:
        return "standby"
    if latest_reading.temperature >= 35.8 or latest_reading.humidity >= 66.0:
        return "cool"
    if latest_reading.temperature <= 32.4:
        return "heat"
    if latest_result and latest_result.get("alert_triggered"):
        risk = latest_result["analysis"]["risk_type"]
        if risk in ("حرارة ورطوبة", "خطر مركّب"):
            return "cool"
        if risk == "برودة سطحية":
            return "heat"
    return "standby"


def _append_readings(count: int) -> None:
    """يولّد قراءات جديدة مع احتساب الأثر التدريجي الواقعي لوحدة التهوية والتكييف."""
    model_obj: SmartMattressModel = st.session_state.model
    current_scenario: ScenarioType = st.session_state.scenario
    scenario_label = SCENARIO_CATALOG[current_scenario]["label"]
    hvac_mode: HvacControlMode = st.session_state.hvac_mode
    target_temp: float = st.session_state.target_temp

    for _ in range(count):
        # في الوضع اليدوي يعمل التكييف فوراً؛ وفي التلقائي يثبت التفعيل فور انطلاق التنبيه الحراري/الرطوبي
        if hvac_mode in ("cool", "heat"):
            sim_climate: Literal["off", "cool", "heat"] = hvac_mode
            st.session_state.climate_active_steps += 1
        elif hvac_mode == "auto" and st.session_state.auto_latched_mode in ("cool", "heat"):
            sim_climate = st.session_state.auto_latched_mode
            st.session_state.climate_active_steps += 1
        else:
            sim_climate = "off"
            st.session_state.climate_active_steps = 0

        reading = generate_reading(
            current_scenario,
            step=st.session_state.step,
            climate_mode=sim_climate,
            climate_steps=st.session_state.climate_active_steps,
            target_temp=target_temp,
        )
        result = model_obj.predict(reading)

        # تثبيت التكييف التلقائي بعد تحقق عتبة التنبيه المتتالي لمنع التذبذب
        if hvac_mode == "auto" and result["alert_triggered"] and st.session_state.auto_latched_mode == "standby":
            risk = result["analysis"]["risk_type"]
            if risk in ("حرارة ورطوبة", "خطر مركّب") or reading.temperature >= 35.8 or reading.humidity >= 66.0:
                st.session_state.auto_latched_mode = "cool"
            elif risk == "برودة سطحية" or reading.temperature <= 32.4:
                st.session_state.auto_latched_mode = "heat"

        st.session_state.step += 1
        st.session_state.total_readings += 1
        st.session_state.history_index.append(st.session_state.total_readings)
        st.session_state.history_temp.append(reading.temperature)
        st.session_state.history_hum.append(reading.humidity)
        st.session_state.history_time.append(datetime.now().strftime("%H:%M:%S"))
        st.session_state.history_scenario.append(scenario_label)

        if result["alert_triggered"] and result["alert"]:
            st.session_state.alerts_log.append(result["alert"])

        st.session_state.latest_reading = reading
        st.session_state.latest_result = result


def _init_session_state() -> None:
    """تهيئة النموذج وحالة الجلسة ووحدة التكييف عند التشغيل لأول مرة."""
    if "model" not in st.session_state:
        m = SmartMattressModel(consecutive_threshold=3)
        m.train()
        st.session_state.model = m

    if "history_temp" not in st.session_state:
        st.session_state.history_temp = []
        st.session_state.history_hum = []
        st.session_state.history_time = []
        st.session_state.history_index = []
        st.session_state.history_scenario = []
        st.session_state.step = 0
        st.session_state.total_readings = 0
        st.session_state.scenario = "طبيعي"
        st.session_state.batch_size = 5
        st.session_state.hvac_mode = "auto"
        st.session_state.auto_latched_mode = "standby"
        st.session_state.target_temp = 34.0
        st.session_state.climate_active_steps = 0
        st.session_state.alerts_log = []
        st.session_state.latest_reading = None
        st.session_state.latest_result = None

        _append_readings(3)


_init_session_state()
model: SmartMattressModel = st.session_state.model
current_scn: ScenarioType = st.session_state.scenario
scn_meta = SCENARIO_CATALOG[current_scn]

# ──────────────────── الترويسة ────────────────────

tone_cls = f"badge-tone-{scn_meta['badge_tone']}"
scenario_badge = (
    f'<span class="badge {tone_cls}"><span class="dot"></span>'
    f'السيناريو: {escape(scn_meta["label"])}</span>'
)

_html(
    f"""
<div class="app-header">
  <div class="brand">
    <div class="brand-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2"
           stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-5 4 10 2-5h6"/></svg>
    </div>
    <div>
      <div class="app-title">المفرش الطبي الذكي</div>
      <div class="app-subtitle">مراقبة تشريحية وتكييف سطحي آمن · نظام محاكاة داعم للقرار</div>
    </div>
  </div>
  <div class="header-badges">
    {scenario_badge}
    <span class="badge">عتبة التنبيه: {model.consecutive_threshold} قراءات متتالية</span>
    <span class="badge">بيانات محاكاة</span>
  </div>
</div>
"""
)

# ──────────────────── بطاقة الملف الصحي للمريض الافتراضي ────────────────────

with st.container(key="card_patient"):
    _html(
        f"""
<div class="patient-bar">
  <div class="patient-id">
    <div class="patient-avatar">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
    <div>
      <div class="patient-title">ملف المريض الافتراضي · الفئة: رعاية طويلة الأمد / مريض طريح فراش</div>
      <div class="patient-sub">خط الأساس الشخصي المتعلّم بالذكاء الاصطناعي (نموذج Isolation Forest متدرّب على النمط الطبيعي للمريض)</div>
    </div>
  </div>
  <div class="baseline-chips">
    <span class="b-chip">نطاق الحرارة الطبيعي: <b>{_num("33.2 – 34.8")} °م</b></span>
    <span class="b-chip">نطاق الرطوبة الطبيعي: <b>{_num("49 – 61")} %</b></span>
    <span class="b-chip">نطاق الضغط المتوازن: <b>{_num("8 – 35")} ملم زئبق</b> (عتبة التركّز الموضعي <b>فوق {_num("45")}</b>)</span>
  </div>
</div>
<div class="patient-note">
  <span>إخلاء مسؤولية طبية: جميع البيانات المعروضة محاكاة توليدية لمريض افتراضي، والنظام أداة داعمة للقرار التمريضي ولا يُستخدم للتشخيص الطبي.</span>
  <span>حجم عيّنة التعلّم الأساسي: <b>{_num("600")} قراءة طبيعية</b></span>
</div>
"""
    )

# ──────────────────── صف التحكم: السيناريوهات + وحدة التهوية والتكييف ────────────────────

col_ctrl_sim, col_ctrl_hvac = st.columns([1.22, 1], gap="medium")

with col_ctrl_sim:
    with st.container(key="card_controls"):
        c_scn, c_slider = st.columns([1.45, 1], gap="small")
        with c_slider:
            n_steps = st.slider(
                "عدد القراءات في كل دفعة",
                min_value=1,
                max_value=30,
                value=st.session_state.batch_size,
            )
            st.session_state.batch_size = n_steps

        with c_scn:
            selected_scn: ScenarioType = st.selectbox(
                "اختر سيناريو المحاكاة",
                options=SCENARIO_ORDER,
                index=SCENARIO_ORDER.index(current_scn),
                format_func=lambda key: SCENARIO_CATALOG[key]["label"],
            )

        b_gen, b_reset = st.columns([1.45, 1], gap="small")
        with b_gen:
            generate_btn = st.button("ولّد قراءات جديدة", type="primary", use_container_width=True)
        with b_reset:
            reset_btn = st.button("إعادة تعيين الجلسة", use_container_width=True)

        if selected_scn != current_scn:
            st.session_state.scenario = selected_scn
            st.session_state.step = 0
            st.session_state.climate_active_steps = 0
            st.session_state.auto_latched_mode = "standby"
            model.reset_streak()
            _append_readings(n_steps)
            st.rerun()

        _html(
            f'<div class="scn-desc-bar">'
            f'<span><b>وصف السيناريو:</b> {escape(scn_meta["description"])}</span>'
            f'<span>القراءات: <b>{_num(st.session_state.total_readings)}</b></span>'
            f"</div>"
        )

with col_ctrl_hvac:
    with st.container(key="card_hvac"):
        effective_preview = _resolve_effective_climate(
            st.session_state.hvac_mode,
            st.session_state.latest_reading,
            st.session_state.latest_result,
            st.session_state.auto_latched_mode,
        )
        if effective_preview == "cool":
            hvac_pill = '<span class="pill pill-cool"><span class="dot"></span>تهوية وتبريد نشط</span>'
        elif effective_preview == "heat":
            hvac_pill = '<span class="pill pill-heat"><span class="dot"></span>تدفئة آمنة نشطة</span>'
        elif effective_preview == "off":
            hvac_pill = '<span class="pill pill-neutral">متوقف يدويًا</span>'
        else:
            hvac_pill = '<span class="pill pill-ok"><span class="dot"></span>مراقبة تلقائية مستقرة</span>'

        _html(
            f'<div class="hvac-head">'
            f'<span class="hvac-title">وحدة التهوية والتبريد والتدفئة الآمنة</span>'
            f"{hvac_pill}</div>"
        )

        hb_cols = st.columns(4, gap="small")
        cur_hvac: HvacControlMode = st.session_state.hvac_mode
        short_hvac_labels: dict[HvacControlMode, str] = {
            "auto": "تلقائي",
            "cool": "تهوية وتبريد",
            "heat": "تدفئة آمنة",
            "off": "إيقاف",
        }
        for idx, mode_key in enumerate(HVAC_ORDER):
            with hb_cols[idx]:
                if st.button(
                    short_hvac_labels[mode_key],
                    key=f"btn_hvac_{mode_key}",
                    type="primary" if cur_hvac == mode_key else "secondary",
                    use_container_width=True,
                ):
                    if cur_hvac != mode_key:
                        st.session_state.hvac_mode = mode_key
                        st.session_state.climate_active_steps = 0
                        st.session_state.auto_latched_mode = "standby"
                        st.rerun()

        target_temp_val = st.slider(
            "درجة الحرارة المستهدفة (النطاق الطبي الآمن ٣٢.٥ – ٣٦.٠ °م)",
            min_value=32.5,
            max_value=36.0,
            value=float(st.session_state.target_temp),
            step=0.5,
        )
        if target_temp_val != st.session_state.target_temp:
            st.session_state.target_temp = target_temp_val

        _html(
            f'<div class="scn-desc-bar">'
            f'<span>الوضع المختار: <b>{HVAC_LABELS[cur_hvac]}</b></span>'
            f'<span>الهدف الآمن: <b>{_num(f"{st.session_state.target_temp:.1f}")} °م</b></span>'
            f"</div>"
        )

if reset_btn:
    st.session_state.history_temp.clear()
    st.session_state.history_hum.clear()
    st.session_state.history_time.clear()
    st.session_state.history_index.clear()
    st.session_state.history_scenario.clear()
    st.session_state.step = 0
    st.session_state.total_readings = 0
    st.session_state.climate_active_steps = 0
    st.session_state.auto_latched_mode = "standby"
    st.session_state.alerts_log.clear()
    st.session_state.latest_reading = None
    st.session_state.latest_result = None
    model.reset()
    st.rerun()

if generate_btn:
    _append_readings(n_steps)
    st.rerun()

latest_reading = st.session_state.latest_reading
latest_result = st.session_state.latest_result
effective_climate = _resolve_effective_climate(
    st.session_state.hvac_mode,
    latest_reading,
    latest_result,
    st.session_state.auto_latched_mode,
)

# ──────────────────── صف بطاقات الملخص ────────────────────


def _delta_html(history: list[float], unit: str) -> str:
    if len(history) < 2:
        return "أول قراءة في الجلسة"
    diff = history[-1] - history[-2]
    if abs(diff) < 0.05:
        return "مستقر مقارنة بالقراءة السابقة"
    arrow, cls = ("▲", "up") if diff > 0 else ("▼", "down")
    return (
        f'<span class="delta {cls}">{_num(f"{arrow} {abs(diff):.1f}{unit}")}</span>'
        " عن القراءة السابقة"
    )


def _metric_card(label: str, accent: str, value: str, unit: str, sub: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-top"><span class="metric-label">'
        f'<span class="metric-accent" style="background:{accent}"></span>{label}</span></div>'
        f'<div><div class="metric-value"><bdi>{value}</bdi><span class="metric-unit">{unit}</span></div>'
        f'<div class="metric-sub">{sub}</div></div>'
        "</div>"
    )


if latest_reading is not None and latest_result is not None:
    analysis = latest_result["analysis"]

    temp_val = latest_reading.temperature
    if temp_val >= 36.0:
        temp_note = "مرتفعة · " + _delta_html(st.session_state.history_temp, "°")
    elif temp_val <= 32.2:
        temp_note = "منخفضة (برودة) · " + _delta_html(st.session_state.history_temp, "°")
    else:
        temp_note = _delta_html(st.session_state.history_temp, "°")

    hum_val = latest_reading.humidity
    if hum_val >= 68.0:
        hum_note = "رطوبة/تعرق مرتفع · " + _delta_html(st.session_state.history_hum, "%")
    else:
        hum_note = _delta_html(st.session_state.history_hum, "%")

    temp_card = _metric_card("الحرارة", "#F43F5E", f"{temp_val:.1f}", "°م", temp_note)
    hum_card = _metric_card("الرطوبة", PRIMARY, f"{hum_val:.1f}", "%", hum_note)

    top_mean_num = _num(f"{analysis['top_zone_mean']:.1f}")
    if analysis["affected_regions"]:
        zone_sub = f"تركّز ضغط: <b>{escape(analysis['region'])}</b> ({top_mean_num} ملم زئبق)"
    else:
        zone_sub = f"توزيع متوازن · أعلى منطقة: {escape(analysis['top_zone_name'])} ({top_mean_num})"

    pressure_card = _metric_card(
        "أقصى ضغط",
        "#F59E0B",
        f"{latest_reading.pressure_matrix.max():.1f}",
        "ملم زئبق",
        zone_sub,
    )

    score_html = "درجة النموذج: " + _num(f"{latest_result['score']:.4f}")
    if latest_result["alert_triggered"]:
        status_cls = "status-danger"
        if analysis["is_compound"]:
            pill = '<span class="pill pill-critical">خطر حرج</span>'
            status_value = "خطر مركّب"
        else:
            pill = '<span class="pill pill-danger"><span class="dot"></span>تنبيه نشط</span>'
            status_value = (
                analysis["region"] if analysis["affected_regions"] else analysis["risk_type"]
            )
        status_sub = f"مستمر منذ {latest_result['consecutive']} قراءات · {score_html}"
    elif latest_result["is_anomaly"]:
        status_cls = "status-warn"
        pill = '<span class="pill pill-warn"><span class="dot"></span>قيد المراقبة</span>'
        status_value = "قراءة شاذة"
        status_sub = (
            f"{latest_result['consecutive']} من {model.consecutive_threshold} قبل إطلاق التنبيه · {score_html}"
        )
    else:
        status_cls = "status-ok"
        pill = '<span class="pill pill-ok"><span class="dot"></span>مستقر</span>'
        status_value = "طبيعي"
        status_sub = f"القراءة ضمن النمط الطبيعي · {score_html}"
else:
    placeholder = "اضغط «ولّد قراءات جديدة» للبدء"
    temp_card = _metric_card("الحرارة", "#F43F5E", "—", "", placeholder)
    hum_card = _metric_card("الرطوبة", PRIMARY, "—", "", placeholder)
    pressure_card = _metric_card("أقصى ضغط", "#F59E0B", "—", "", placeholder)
    status_cls, pill = "", '<span class="pill pill-neutral">بانتظار البيانات</span>'
    status_value, status_sub = "—", placeholder

status_font = "22px" if len(status_value) > 14 else "26px"
status_card = (
    f'<div class="metric-card {status_cls}">'
    f'<div class="metric-top"><span class="metric-label">حالة التنبيه</span>{pill}</div>'
    f'<div><div class="metric-value" style="font-size:{status_font}">{escape(status_value)}</div>'
    f'<div class="metric-sub">{status_sub}</div></div>'
    "</div>"
)

_html(f'<div class="metrics-grid">{temp_card}{hum_card}{pressure_card}{status_card}</div>')

# ──────────────────── الرسوم البيانية وعرض 3D التفاعلي ────────────────────


def _base_layout(height: int) -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color=TEXT, size=12),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(family=FONT_FAMILY, color=TEXT)),
        showlegend=False,
    )


def _pressure_heatmap(matrix, highlight_shapes: list[dict]) -> go.Figure:
    customdata = [
        [
            f"{get_cell_region(r, c)} (صف {r + 1} · عمود {c + 1})"
            for c in range(COLS)
        ]
        for r in range(ROWS)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(range(COLS)),
            y=list(range(ROWS)),
            zmin=0,
            zmax=PRESSURE_MAX,
            colorscale=PRESSURE_COLORSCALE,
            xgap=3,
            ygap=3,
            customdata=customdata,
            texttemplate="%{z:.0f}",
            textfont=dict(size=10, family=FONT_FAMILY),
            hovertemplate="%{customdata}<br>الضغط: %{z:.1f} ملم زئبق<extra></extra>",
            colorbar=dict(
                title=dict(text="ملم زئبق", side="top", font=dict(size=11, color=MUTED)),
                thickness=10,
                len=0.85,
                outlinewidth=0,
                x=-0.03,
                xanchor="right",
                tickfont=dict(size=10, color=MUTED),
            ),
        )
    )

    for shp in highlight_shapes:
        if shp["type"] == "rect":
            fig.add_shape(
                type="rect",
                x0=shp["x0"], y0=shp["y0"], x1=shp["x1"], y1=shp["y1"],
                line=dict(color="#0F172A", width=2, dash="dot"),
            )
        elif shp["type"] == "path":
            fig.add_shape(
                type="path",
                path=shp["path"],
                line=dict(color="#0F172A", width=2, dash="dot"),
            )

    fig.update_layout(_base_layout(500))
    fig.update_layout(margin=dict(l=70, r=8, t=8, b=8))
    fig.update_yaxes(
        autorange="reversed", side="right", tickmode="array",
        tickvals=list(range(ROWS)), ticktext=ROW_LABELS,
        tickfont=dict(size=11, color=MUTED), showgrid=False, zeroline=False, ticks="",
    )
    fig.update_xaxes(
        autorange="reversed", tickmode="array", tickvals=list(range(COLS)), ticktext=COL_LABELS,
        tickfont=dict(size=11, color=MUTED), showgrid=False, zeroline=False, ticks="",
    )
    return fig


def _trend_chart(x, y, scenarios: list[str], color: str, unit: str, band: tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor="rgba(2,132,199,0.08)", line_width=0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            customdata=scenarios,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.55),
            marker=dict(size=6, color="#FFFFFF", line=dict(color=color, width=2)),
            hovertemplate=f"القراءة %{{x}} (%{{customdata}})<br>%{{y:.1f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(_base_layout(248))
    fig.update_xaxes(
        title=dict(text="رقم القراءة", font=dict(size=11, color=MUTED)),
        gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(size=10.5, color=MUTED),
    )
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(size=10.5, color=MUTED))
    return fig


PLOT_CONFIG = {"displayModeBar": False}

col_visual, col_trends = st.columns([1.35, 1], gap="medium")

with col_visual:
    with st.container(key="card_visuals"):
        tab_3d, tab_2d = st.tabs(["عرض 3D تفاعلي للطبقات", "خريطة الضغط 2D"])

        with tab_3d:
            _html(
                '<div class="section-head">'
                '<div class="section-title">التوأم الرقمي للمفرش الطبي الذكي (٥ طبقات هندسية)</div>'
                '<div class="section-sub">تدوير وتقريب بالماوس · تفكيك تفاعلي للطبقات وإظهار/إخفاء المريض</div>'
                "</div>"
            )
            html_3d = build_mattress_3d_html(
                pressure_matrix=latest_reading.pressure_matrix if latest_reading is not None else None,
                climate_visual=effective_climate,
                alert_triggered=bool(latest_result["alert_triggered"]) if latest_result is not None else False,
                affected_region=latest_result["analysis"]["region"] if latest_result is not None else "توزيع متوازن",
                temperature=latest_reading.temperature if latest_reading is not None else 34.0,
                humidity=latest_reading.humidity if latest_reading is not None else 55.0,
            )
            components.html(html_3d, height=548, scrolling=False)

        with tab_2d:
            if latest_result is not None and latest_result["analysis"]["affected_regions"]:
                map_sub = f"مصفوفة ١٦ × ٨ · الإطار المتقطع يحدد منطقة الضغط: {escape(latest_result['analysis']['region'])}"
            else:
                map_sub = "مصفوفة ١٦ × ٨ حسّاسًا · توزيع تشريحي كامل للجسم"

            _html(
                '<div class="section-head"><div class="section-title">خريطة توزيع الضغط الثنائية الأبعاد</div>'
                f'<div class="section-sub">{map_sub}</div></div>'
            )
            if latest_reading is not None and latest_result is not None:
                st.plotly_chart(
                    _pressure_heatmap(
                        latest_reading.pressure_matrix,
                        latest_result["analysis"]["highlight_shapes"],
                    ),
                    use_container_width=True,
                    config=PLOT_CONFIG,
                )
                _html(
                    '<div class="zones-legend">'
                    "تقسيم المناطق: الرأس (صف ١–٢) · الكتفان (صف ٣–٤) · الظهر (صف ٥–٦) · "
                    "الذراعان واليدان (الأعمدة الطرفية ١–٢ و٧–٨ في الصفوف ٤–٩) · "
                    "الحوض (صف ٧–١٠) · الفخذان (صف ١١–١٣) · الساقان والكعبان (صف ١٤–١٦)"
                    "</div>"
                )
            else:
                _html('<div class="empty-state">لا توجد قراءات بعد. اضغط «ولّد قراءات جديدة» لعرض خريطة الضغط.</div>')

with col_trends:
    with st.container(key="card_temp"):
        _html(
            '<div class="section-head"><div class="section-title">منحنى الحرارة</div>'
            '<div class="section-sub">درجة مئوية · المنطقة المظللة = النمط الطبيعي</div></div>'
        )
        if st.session_state.history_temp:
            st.plotly_chart(
                _trend_chart(
                    st.session_state.history_index,
                    st.session_state.history_temp,
                    st.session_state.history_scenario,
                    "#F43F5E",
                    "°م",
                    TEMP_NORMAL_BAND,
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )
        else:
            _html('<div class="empty-state">لا توجد بيانات بعد</div>')

    with st.container(key="card_hum"):
        _html(
            '<div class="section-head"><div class="section-title">منحنى الرطوبة</div>'
            '<div class="section-sub">رطوبة نسبية % · المنطقة المظللة = النمط الطبيعي</div></div>'
        )
        if st.session_state.history_hum:
            st.plotly_chart(
                _trend_chart(
                    st.session_state.history_index,
                    st.session_state.history_hum,
                    st.session_state.history_scenario,
                    PRIMARY,
                    "%",
                    HUM_NORMAL_BAND,
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )
        else:
            _html('<div class="empty-state">لا توجد بيانات بعد</div>')

# ──────────────────── سجل التنبيهات وتوصيات التمريض ────────────────────

alerts = st.session_state.alerts_log
with st.container(key="card_alerts"):
    count_pill = (
        f'<span class="pill pill-danger">{len(alerts)} تنبيه</span>'
        if alerts else '<span class="pill pill-ok">لا تنبيهات</span>'
    )
    _html(
        '<div class="section-head">'
        '<div><span class="section-title">سجل التنبيهات وتوصيات التمريض</span></div>'
        f"{count_pill}</div>"
    )

    if alerts:
        items = []
        for alert in reversed(alerts):
            if alert.severity == "خطر حرج":
                sev_cls, pill_cls = "sev-critical", "pill-critical"
            elif alert.severity == "خطر":
                sev_cls, pill_cls = "sev-danger", "pill-danger"
            else:
                sev_cls, pill_cls = "sev-warn", "pill-warn"

            d = alert.details
            consec = d.get("قراءات_متتالية", "—")
            cause_text = alert.cause or alert.message
            rec_text = alert.recommendation or "تفقّد وضعية المريض ومراجعة المؤشرات."
            region_text = alert.region or d.get("المنطقة", "—")

            items.append(
                f'<div class="alert-item {sev_cls}">'
                f'<div class="alert-top">'
                f'<div class="alert-badges">'
                f'<span class="pill {pill_cls}">{escape(alert.severity)}</span>'
                f'<span class="pill pill-region">المنطقة: {escape(region_text)}</span>'
                f'<span class="pill pill-neutral">شذوذ متتالي: {_num(consec)} قراءات</span>'
                f"</div>"
                f'<span class="alert-meta">الوقت <b>{_num(escape(alert.timestamp))}</b></span>'
                f"</div>"
                f'<div class="alert-cause">{escape(cause_text)}</div>'
                f'<div class="alert-rec"><b>الإجراء الموصى به للممرض:</b> {escape(rec_text)}</div>'
                '<div class="alert-meta">'
                f'<span>أقصى ضغط <b>{_num(d.get("أقصى_ضغط", "—"))}</b> ملم زئبق</span>'
                f'<span>الحرارة <b>{_num(d.get("الحرارة", "—"))}</b> °م</span>'
                f'<span>الرطوبة <b>{_num(d.get("الرطوبة", "—"))}</b> %</span>'
                f'<span>درجة الشذوذ <b>{_num(d.get("درجة_الشذوذ", "—"))}</b></span>'
                "</div></div>"
            )
        _html(f'<div class="alert-list">{"".join(items)}</div>')
    else:
        _html(
            '<div class="empty-state" style="padding:28px 12px">لا توجد تنبيهات حتى الآن. '
            "اختر أيًا من سيناريوهات الخطر من القائمة العلوية لمشاهدة رصد المنطقة المتأثرة وتوصية الإجراء التمريضي.</div>"
        )

# ──────────────────── تذييل ────────────────────

_html(
    '<div class="disclaimer"><b>تنبيه مهم:</b> يعمل هذا النظام ببيانات محاكاة توليدية، '
    "وهو أداة داعمة للقرار فقط. لا يُغني عن التقييم السريري المتخصص ولا يُستخدم للتشخيص الطبي.</div>"
)

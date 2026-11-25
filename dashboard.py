"""
dashboard.py – لوحة تحكم المفرش الطبي الذكي (ثنائية اللغة: العربية RTL | الإنجليزية LTR)

تصميم نهاري (Light Mode) بأسلوب طبي نظيف يضم:
  • زر تبديل فوري بين العربية (افتراضي RTL) والإنجليزية (LTR) دون المساس بحالة الجلسة أو النموذج
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
from model import SmartMattressModel, localize_alert, localize_analysis
from simulator import (
    COLS,
    ROWS,
    SCENARIO_CATALOG,
    SCENARIO_ORDER,
    LangType,
    ScenarioKey,
    generate_reading,
    get_cell_region,
    get_scenario_description,
    get_scenario_label,
    normalize_scenario_key,
)

# ──────────────────── ثوابت التصميم ────────────────────

PRIMARY = "#0284C7"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
GRID = "#F1F5F9"
FONT_FAMILY = "Cairo, Inter, Tajawal, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

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
HVAC_ORDER: list[HvacControlMode] = ["auto", "cool", "heat", "off"]

# ──────────────────── قاموس الترجمة المنظّم (ar / en) ────────────────────

UI_I18N: dict[LangType, dict] = {
    "ar": {
        "dir": "rtl",
        "align": "right",
        "crit_grad": "90deg",
        "app_title": "المفرش الطبي الذكي",
        "app_subtitle": "مراقبة تشريحية وتكييف سطحي آمن · نظام محاكاة داعم للقرار",
        "badge_scenario": "السيناريو",
        "badge_threshold": "عتبة التنبيه: {n} قراءات متتالية",
        "badge_sim": "بيانات محاكاة",
        "patient_title": "ملف المريض الافتراضي · الفئة: رعاية طويلة الأمد / مريض طريح فراش",
        "patient_sub": "خط الأساس الشخصي المتعلّم بالذكاء الاصطناعي (نموذج Isolation Forest متدرّب على النمط الطبيعي للمريض)",
        "chip_temp": "نطاق الحرارة الطبيعي",
        "chip_hum": "نطاق الرطوبة الطبيعي",
        "chip_press": "نطاق الضغط المتوازن",
        "chip_hotspot": "عتبة التركّز الموضعي فوق",
        "patient_disclaimer": "إخلاء مسؤولية طبية: جميع البيانات المعروضة محاكاة توليدية لمريض افتراضي، والنظام أداة داعمة للقرار التمريضي ولا يُستخدم للتشخيص الطبي.",
        "sample_size": "حجم عيّنة التعلّم الأساسي: <b>{n} قراءة طبيعية</b>",
        "select_scenario": "اختر سيناريو المحاكاة",
        "batch_slider": "عدد القراءات في كل دفعة",
        "btn_generate": "ولّد قراءات جديدة",
        "btn_reset": "إعادة تعيين الجلسة",
        "scn_desc_prefix": "وصف السيناريو:",
        "readings_count": "القراءات:",
        "hvac_title": "وحدة التهوية والتبريد والتدفئة الآمنة",
        "hvac_pill_cool": "تهوية وتبريد نشط",
        "hvac_pill_heat": "تدفئة آمنة نشطة",
        "hvac_pill_off": "متوقف يدويًا",
        "hvac_pill_standby": "مراقبة تلقائية مستقرة",
        "hvac_short": {
            "auto": "تلقائي",
            "cool": "تهوية وتبريد",
            "heat": "تدفئة آمنة",
            "off": "إيقاف",
        },
        "hvac_full": {
            "auto": "تلقائي",
            "cool": "تهوية وتبريد آمن",
            "heat": "تدفئة آمنة",
            "off": "إيقاف",
        },
        "target_slider": "درجة الحرارة المستهدفة (النطاق الطبي الآمن 32.5 – 36.0 °م)",
        "selected_mode": "الوضع المختار:",
        "safe_target": "الهدف الآمن:",
        "unit_temp": "°م",
        "unit_press": "ملم زئبق",
        "delta_first": "أول قراءة في الجلسة",
        "delta_steady": "مستقر مقارنة بالقراءة السابقة",
        "delta_prev": " عن القراءة السابقة",
        "card_temp": "الحرارة",
        "card_hum": "الرطوبة",
        "card_press": "أقصى ضغط",
        "card_status": "حالة التنبيه",
        "note_high_temp": "مرتفعة · ",
        "note_low_temp": "منخفضة (برودة) · ",
        "note_high_hum": "رطوبة/تعرق مرتفع · ",
        "zone_hotspot": "تركّز ضغط: <b>{region}</b> ({val} ملم زئبق)",
        "zone_balanced": "توزيع متوازن · أعلى منطقة: {zone} ({val})",
        "model_score": "درجة النموذج: ",
        "pill_critical": "خطر حرج",
        "val_compound": "خطر مركّب",
        "pill_active_alert": "تنبيه نشط",
        "sub_consecutive": "مستمر منذ {n} قراءات · {score}",
        "pill_monitoring": "قيد المراقبة",
        "val_anomaly": "قراءة شاذة",
        "sub_before_alert": "{c} من {t} قبل إطلاق التنبيه · {score}",
        "pill_stable": "مستقر",
        "val_normal": "طبيعي",
        "sub_normal": "القراءة ضمن النمط الطبيعي · {score}",
        "placeholder_start": "اضغط «ولّد قراءات جديدة» للبدء",
        "pill_waiting": "بانتظار البيانات",
        "tab_3d": "عرض 3D تفاعلي للطبقات",
        "tab_2d": "خريطة الضغط 2D",
        "twin_title": "التوأم الرقمي للمفرش الطبي الذكي (٥ طبقات هندسية)",
        "twin_sub": "تدوير وتقريب بالماوس · تفكيك تفاعلي للطبقات وإظهار/إخفاء المريض",
        "map2d_title": "خريطة توزيع الضغط الثنائية الأبعاد",
        "map2d_sub_hotspot": "مصفوفة ١٦ × ٨ · الإطار المتقطع يحدد منطقة الضغط: {region}",
        "map2d_sub_normal": "مصفوفة ١٦ × ٨ حسّاسًا · توزيع تشريحي كامل للجسم",
        "zones_legend": (
            "تقسيم المناطق: الرأس (صف ١–٢) · الكتفان (صف ٣–٤) · الظهر (صف ٥–٦) · "
            "الذراعان واليدان (الأعمدة الطرفية ١–٢ و٧–٨ في الصفوف ٤–٩) · "
            "الحوض (صف ٧–١٠) · الفخذان (صف ١١–١٣) · الساقان والكعبان (صف ١٤–١٦)"
        ),
        "empty_map": "لا توجد قراءات بعد. اضغط «ولّد قراءات جديدة» لعرض خريطة الضغط.",
        "empty_data": "لا توجد بيانات بعد",
        "hover_row_col": "صف {r} · عمود {c}",
        "hover_pressure": "الضغط",
        "trend_temp_title": "منحنى الحرارة",
        "trend_temp_sub": "درجة مئوية · المنطقة المظللة = النمط الطبيعي",
        "trend_hum_title": "منحنى الرطوبة",
        "trend_hum_sub": "رطوبة نسبية % · المنطقة المظللة = النمط الطبيعي",
        "axis_reading": "رقم القراءة",
        "hover_reading": "القراءة",
        "alerts_title": "سجل التنبيهات وتوصيات التمريض",
        "alerts_count": "{n} تنبيه",
        "alerts_none": "لا تنبيهات",
        "lbl_region": "المنطقة:",
        "lbl_consec": "شذوذ متتالي: {n} قراءات",
        "lbl_time": "الوقت",
        "lbl_nurse_rec": "الإجراء الموصى به للممرض:",
        "lbl_peak_p": "أقصى ضغط",
        "lbl_temp": "الحرارة",
        "lbl_hum": "الرطوبة",
        "lbl_anom_score": "درجة الشذوذ",
        "alerts_empty": (
            "لا توجد تنبيهات حتى الآن. اختر أيًا من سيناريوهات الخطر من القائمة العلوية "
            "لمشاهدة رصد المنطقة المتأثرة وتوصية الإجراء التمريضي."
        ),
        "footer_disclaimer": (
            "<b>تنبيه مهم:</b> يعمل هذا النظام ببيانات محاكاة توليدية، "
            "وهو أداة داعمة للقرار فقط. لا يُغني عن التقييم السريري المتخصص ولا يُستخدم للتشخيص الطبي."
        ),
        "row_labels": [
            "رأس", "رأس", "كتفان", "كتفان · ذراع",
            "ظهر · ذراع", "ظهر · ذراع", "حوض · يد", "حوض · يد",
            "حوض · يد", "حوض", "فخذان", "فخذان",
            "فخذان", "ساقان", "كعبان وقدمان", "كعبان وقدمان",
        ],
        "col_labels": ["١", "٢", "٣", "٤", "٥", "٦", "٧", "٨"],
    },
    "en": {
        "dir": "ltr",
        "align": "left",
        "crit_grad": "270deg",
        "app_title": "Smart Medical Mattress",
        "app_subtitle": "Full-Body Pressure Monitoring & Safe Microclimate · Decision-Support System",
        "badge_scenario": "Scenario",
        "badge_threshold": "Alert Threshold: {n} Consecutive Readings",
        "badge_sim": "Simulated Data",
        "patient_title": "Virtual Patient Profile · Category: Long-Term Care / Bedridden Patient",
        "patient_sub": "AI-Learned Personal Baseline (Isolation Forest Model Trained Exclusively on Normal Posture Data)",
        "chip_temp": "Normal Temp Range",
        "chip_hum": "Normal Humidity Range",
        "chip_press": "Balanced Pressure",
        "chip_hotspot": "Hotspot Threshold above",
        "patient_disclaimer": "Medical Disclaimer: All displayed metrics represent synthetic simulated data for a virtual patient. This system is a nursing decision-support tool and does not provide medical diagnosis.",
        "sample_size": "Baseline Training Sample: <b>{n} Normal Readings</b>",
        "select_scenario": "Select Simulation Scenario",
        "batch_slider": "Readings per Batch",
        "btn_generate": "Generate New Readings",
        "btn_reset": "Reset Session",
        "scn_desc_prefix": "Scenario Description:",
        "readings_count": "Readings:",
        "hvac_title": "Safe Ventilation, Cooling & Warming Unit",
        "hvac_pill_cool": "Active Cooling & Vent",
        "hvac_pill_heat": "Active Safe Warming",
        "hvac_pill_off": "Manually Off",
        "hvac_pill_standby": "Auto Standby",
        "hvac_short": {
            "auto": "Auto",
            "cool": "Cool & Vent",
            "heat": "Safe Warm",
            "off": "Off",
        },
        "hvac_full": {
            "auto": "Automatic",
            "cool": "Safe Cooling & Ventilation",
            "heat": "Safe Warming",
            "off": "Off",
        },
        "target_slider": "Target Surface Temperature (Safe Medical Range 32.5 – 36.0 °C)",
        "selected_mode": "Selected Mode:",
        "safe_target": "Safe Target:",
        "unit_temp": "°C",
        "unit_press": "mmHg",
        "delta_first": "First reading in session",
        "delta_steady": "Steady vs previous reading",
        "delta_prev": " vs previous reading",
        "card_temp": "Temperature",
        "card_hum": "Humidity",
        "card_press": "Peak Pressure",
        "card_status": "Alert Status",
        "note_high_temp": "Elevated · ",
        "note_low_temp": "Low (Cold) · ",
        "note_high_hum": "High Moisture/Sweat · ",
        "zone_hotspot": "Hotspot: <b>{region}</b> ({val} mmHg)",
        "zone_balanced": "Balanced · Peak Zone: {zone} ({val})",
        "model_score": "Model Score: ",
        "pill_critical": "Critical",
        "val_compound": "Compound Risk",
        "pill_active_alert": "Active Alert",
        "sub_consecutive": "Sustained for {n} readings · {score}",
        "pill_monitoring": "Monitoring",
        "val_anomaly": "Anomalous Reading",
        "sub_before_alert": "{c} of {t} before alert trigger · {score}",
        "pill_stable": "Stable",
        "val_normal": "Normal",
        "sub_normal": "Reading within normal baseline · {score}",
        "placeholder_start": "Click “Generate New Readings” to start",
        "pill_waiting": "Awaiting Data",
        "tab_3d": "Interactive 3D Layers",
        "tab_2d": "2D Pressure Map",
        "twin_title": "Smart Medical Mattress Digital Twin (5 Engineered Layers)",
        "twin_sub": "Drag to rotate & scroll to zoom · Interactive layer explosion & patient visibility toggle",
        "map2d_title": "2D Anatomical Pressure Distribution",
        "map2d_sub_hotspot": "16 × 8 Matrix · Dotted frame marks elevated pressure: {region}",
        "map2d_sub_normal": "16 × 8 Sensor Array · Full-Body Anatomical Mapping",
        "zones_legend": (
            "Anatomical Map: Head (Rows 1–2) · Shoulders (Rows 3–4) · Back (Rows 5–6) · "
            "Arms & Hands (Outer Columns 1–2 & 7–8 in Rows 4–9) · "
            "Pelvis (Rows 7–10) · Thighs (Rows 11–13) · Lower Legs & Heels (Rows 14–16)"
        ),
        "empty_map": "No readings yet. Click “Generate New Readings” to display the pressure map.",
        "empty_data": "No data available yet",
        "hover_row_col": "Row {r} · Col {c}",
        "hover_pressure": "Pressure",
        "trend_temp_title": "Temperature Trend",
        "trend_temp_sub": "°C · Shaded band = Learned Normal Baseline",
        "trend_hum_title": "Humidity Trend",
        "trend_hum_sub": "Relative Humidity % · Shaded band = Learned Normal Baseline",
        "axis_reading": "Reading #",
        "hover_reading": "Reading",
        "alerts_title": "Alert Log & Nursing Recommendations",
        "alerts_count": "{n} Alerts",
        "alerts_none": "No Alerts",
        "lbl_region": "Region:",
        "lbl_consec": "Consecutive: {n} readings",
        "lbl_time": "Time",
        "lbl_nurse_rec": "Recommended Nursing Action:",
        "lbl_peak_p": "Peak Pressure",
        "lbl_temp": "Temp",
        "lbl_hum": "Humidity",
        "lbl_anom_score": "Anomaly Score",
        "alerts_empty": (
            "No alerts recorded yet. Select any risk scenario from the top menu to observe "
            "automatic region detection and nursing action recommendations."
        ),
        "footer_disclaimer": (
            "<b>Important Notice:</b> This system operates on synthetic simulated data and serves strictly "
            "as a clinical decision-support tool. It does not replace professional clinical evaluation or medical diagnosis."
        ),
        "row_labels": [
            "Head", "Head", "Shoulders", "Shoulders · Arm",
            "Back · Arm", "Back · Arm", "Pelvis · Hand", "Pelvis · Hand",
            "Pelvis · Hand", "Pelvis", "Thighs", "Thighs",
            "Thighs", "Lower Legs", "Heels & Feet", "Heels & Feet",
        ],
        "col_labels": ["1", "2", "3", "4", "5", "6", "7", "8"],
    },
}


def _num(value) -> str:
    """يعزل الرقم باتجاه LTR كي تظهر إشارة السالب والأرقام العشرية في مكانها الصحيح."""
    return f'<span class="ltr-num">{value}</span>'


# ──────────────────── إعداد الصفحة ────────────────────

st.set_page_config(
    page_title="المفرش الطبي الذكي | Smart Medical Mattress",
    page_icon=":material/monitor_heart:",
    layout="wide",
)


def _html(markup: str) -> None:
    compact = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


# ──────────────────── منطق التكييف السطحي وحالة الجلسة ────────────────────


def _resolve_effective_climate(
    hvac_mode: HvacControlMode,
    latest_reading,
    latest_result,
    latched_auto: str = "standby",
) -> Literal["standby", "cool", "heat", "off"]:
    if hvac_mode == "off":
        return "off"
    if hvac_mode == "cool":
        return "cool"
    if hvac_mode == "heat":
        return "heat"

    if latched_auto in ("cool", "heat"):
        return latched_auto  # type: ignore[return-value]
    if latest_reading is None:
        return "standby"
    if latest_reading.temperature >= 35.8 or latest_reading.humidity >= 66.0:
        return "cool"
    if latest_reading.temperature <= 32.4:
        return "heat"
    if latest_result and latest_result.get("alert_triggered"):
        rk = latest_result["analysis"].get("risk_key")
        if rk in ("heat_humidity", "compound"):
            return "cool"
        if rk == "cold_surface":
            return "heat"
    return "standby"


def _append_readings(count: int) -> None:
    """يولّد قراءات جديدة باستخدام المفاتيح الداخلية الثابتة دون التأثر بتغيير اللغة."""
    model_obj: SmartMattressModel = st.session_state.model
    current_scenario: ScenarioKey = normalize_scenario_key(st.session_state.scenario)
    st.session_state.scenario = current_scenario
    hvac_mode: HvacControlMode = st.session_state.hvac_mode
    target_temp: float = st.session_state.target_temp

    for _ in range(count):
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

        if hvac_mode == "auto" and result["alert_triggered"] and st.session_state.auto_latched_mode == "standby":
            rk = result["analysis"].get("risk_key")
            if rk in ("heat_humidity", "compound") or reading.temperature >= 35.8 or reading.humidity >= 66.0:
                st.session_state.auto_latched_mode = "cool"
            elif rk == "cold_surface" or reading.temperature <= 32.4:
                st.session_state.auto_latched_mode = "heat"

        st.session_state.step += 1
        st.session_state.total_readings += 1
        st.session_state.history_index.append(st.session_state.total_readings)
        st.session_state.history_temp.append(reading.temperature)
        st.session_state.history_hum.append(reading.humidity)
        st.session_state.history_time.append(datetime.now().strftime("%H:%M:%S"))
        st.session_state.history_scenario.append(current_scenario)

        if result["alert_triggered"] and result["alert"]:
            st.session_state.alerts_log.append(result["alert"])

        st.session_state.latest_reading = reading
        st.session_state.latest_result = result


def _init_session_state() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "ar"

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
        st.session_state.scenario = "normal"
        st.session_state.batch_size = 5
        st.session_state.hvac_mode = "auto"
        st.session_state.auto_latched_mode = "standby"
        st.session_state.target_temp = 34.0
        st.session_state.climate_active_steps = 0
        st.session_state.alerts_log = []
        st.session_state.latest_reading = None
        st.session_state.latest_result = None

        _append_readings(3)
    else:
        st.session_state.scenario = normalize_scenario_key(st.session_state.scenario)


_init_session_state()
lang: LangType = st.session_state.lang
tr = UI_I18N[lang]

# ──────────────────── حقن تنسيق الثيم النهاري واتجاه اللغة (RTL / LTR) ────────────────────

_html(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700&display=swap');

:root {{
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
}}

html, body, .stApp {{ background: var(--bg); }}
.stApp {{ direction: {tr['dir']}; text-align: {tr['align']}; color: var(--text); }}

.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-baseweb="select"], [data-baseweb="popover"] li,
.stApp [data-baseweb="tab"] p,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stSliderThumbValue"],
.stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] {{
  font-family: 'Cairo', 'Inter', 'Tajawal', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

[data-baseweb="popover"] ul {{ direction: {tr['dir']}; text-align: {tr['align']}; }}

header[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1360px;
}}

/* البطاقات */
[class*="st-key-card_"] {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 22px;
}}

/* زر تبديل اللغة العلوي الأنيق */
.st-key-lang_switcher {{
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
}}
.st-key-lang_switcher .stButton > button {{
  border-radius: 999px !important;
  min-height: 34px !important;
  padding: 2px 12px !important;
  box-shadow: none !important;
}}
.st-key-lang_switcher [data-testid="stBaseButton-secondary"] {{
  border-color: transparent !important;
  background: transparent !important;
  color: var(--muted) !important;
}}
.st-key-lang_switcher [data-testid="stBaseButton-primary"] {{
  box-shadow: 0 2px 8px rgba(2, 132, 199, 0.24) !important;
}}

/* القائمة المنسدلة */
[data-testid="stSelectbox"] label p {{
  color: var(--muted); font-size: 13px; font-weight: 600;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  min-height: 44px !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
  border-color: var(--primary) !important;
}}

/* الأزرار */
.stButton > button {{ border-radius: 12px; min-height: 42px; font-weight: 600; transition: all .15s ease; padding: 4px 10px; }}
.stButton > button p {{ font-weight: 600; font-size: 13.5px; }}
[data-testid="stBaseButton-primary"] {{
  background: var(--primary) !important; border: 1px solid var(--primary) !important; color: #FFFFFF !important;
  box-shadow: 0 4px 12px rgba(2, 132, 199, 0.22);
}}
[data-testid="stBaseButton-primary"]:hover {{ background: var(--primary-dark) !important; border-color: var(--primary-dark) !important; }}
[data-testid="stBaseButton-secondary"] {{
  background: #FFFFFF !important; border: 1px solid var(--border) !important; color: var(--text) !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
  border-color: var(--primary) !important; color: var(--primary) !important; background: var(--primary-soft) !important;
}}

/* شريط التمرير */
[data-testid="stSlider"] [data-baseweb="slider"] {{ direction: ltr; }}
[data-testid="stSlider"] label p {{ color: var(--muted); font-size: 13px; font-weight: 600; }}

/* التبويبات */
[data-baseweb="tab-list"] {{
  gap: 8px; background: #F1F5F9; padding: 5px; border-radius: 12px; border-bottom: none !important;
}}
[data-baseweb="tab"] {{
  border-radius: 9px !important; padding: 7px 18px !important; font-weight: 700 !important;
  color: var(--muted) !important; border: none !important; background: transparent !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  background: #FFFFFF !important; color: var(--primary) !important;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
}}
[data-baseweb="tab-highlight"] {{ display: none !important; }}
[data-baseweb="tab-panel"] {{ padding-top: 12px !important; }}

/* الترويسة */
.app-header {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; }}
.brand {{ display: flex; align-items: center; gap: 14px; }}
.brand-icon {{
  width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #0EA5E9, #0284C7);
  display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(2, 132, 199, 0.28);
  flex-shrink: 0;
}}
.app-title {{ font-size: 27px; font-weight: 800; color: var(--text); line-height: 1.25; letter-spacing: -0.2px; }}
.app-subtitle {{ font-size: 13.5px; color: var(--muted); margin-top: 2px; }}
.header-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px;
  font-size: 12px; font-weight: 600; border: 1px solid var(--border); background: #FFFFFF; color: var(--muted);
}}
.badge-tone-normal {{ color: #047857; background: #ECFDF5; border-color: #A7F3D0; }}
.badge-tone-warn {{ color: #B45309; background: #FFFBEB; border-color: #FDE68A; }}
.badge-tone-cold {{ color: #0369A1; background: #F0F9FF; border-color: #BAE6FD; }}
.badge-tone-danger {{ color: #B91C1C; background: #FEF2F2; border-color: #FECACA; }}
.badge-tone-critical {{ color: #991B1B; background: #FEE2E2; border-color: #FCA5A5; }}
.dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }}

/* بطاقة الملف الصحي للمريض الافتراضي */
.patient-bar {{
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}}
.patient-id {{ display: flex; align-items: center; gap: 12px; }}
.patient-avatar {{
  width: 40px; height: 40px; border-radius: 12px; background: #E0F2FE; color: #0284C7;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}}
.patient-title {{ font-size: 14.5px; font-weight: 700; color: var(--text); }}
.patient-sub {{ font-size: 12.5px; color: var(--muted); }}
.baseline-chips {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.b-chip {{
  padding: 5px 11px; border-radius: 10px; background: #F8FAFC; border: 1px solid var(--border);
  font-size: 12px; color: var(--muted);
}}
.b-chip b {{ color: var(--text); font-weight: 700; }}
.patient-note {{
  margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  font-size: 12px; color: #0369A1;
}}

.scn-desc-bar {{
  margin-top: 8px; padding: 8px 12px; border-radius: 10px;
  background: #F8FAFC; border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  font-size: 12px; color: var(--muted);
}}
.scn-desc-bar b {{ color: var(--text); font-weight: 600; }}

.hvac-head {{
  display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  margin-bottom: 8px;
}}
.hvac-title {{ font-size: 14.5px; font-weight: 700; color: var(--text); }}

/* بطاقات الملخص */
.metrics-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 4px 0 6px; }}
@media (max-width: 960px) {{ .metrics-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.metric-card {{
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 18px 20px; min-height: 138px;
  display: flex; flex-direction: column; justify-content: space-between;
}}
.metric-top {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
.metric-label {{ display: flex; align-items: center; gap: 8px; font-size: 13.5px; font-weight: 600; color: var(--muted); }}
.metric-accent {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.metric-value {{ font-size: 31px; font-weight: 700; color: var(--text); margin-top: 6px; line-height: 1.2; }}
.metric-unit {{ font-size: 14px; font-weight: 600; color: var(--muted); margin-inline-start: 6px; }}
.metric-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height: 1.5; }}
.delta {{ font-weight: 700; }}
.delta.up {{ color: #B45309; }}
.delta.down {{ color: #0369A1; }}

.metric-card.status-ok {{ border-color: #A7F3D0; background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 70%); }}
.metric-card.status-warn {{ border-color: #FDE68A; background: linear-gradient(180deg, #FFFBEB 0%, #FFFFFF 70%); }}
.metric-card.status-danger {{ border-color: #FECACA; background: linear-gradient(180deg, #FEF2F2 0%, #FFFFFF 70%); }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
.pill-ok {{ color: #047857; background: #D1FAE5; }}
.pill-warn {{ color: #B45309; background: #FEF3C7; }}
.pill-cool {{ color: #0369A1; background: #E0F2FE; }}
.pill-heat {{ color: #C2410C; background: #FFEDD5; }}
.pill-danger {{ color: #B91C1C; background: #FEE2E2; }}
.pill-critical {{ color: #FFFFFF; background: #DC2626; }}
.pill-region {{ color: #0369A1; background: #E0F2FE; font-weight: 600; }}
.pill-neutral {{ color: var(--muted); background: #F1F5F9; }}

/* عناوين الأقسام */
.section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }}
.section-title {{ font-size: 17px; font-weight: 700; color: var(--text); }}
.section-sub {{ font-size: 12.5px; color: var(--muted); }}
.zones-legend {{
  margin-top: 6px; padding-top: 8px; border-top: 1px dashed var(--border);
  font-size: 11.5px; color: var(--muted); line-height: 1.7;
}}

.ltr-num {{ direction: ltr; unicode-bidi: isolate; display: inline-block; }}
.empty-state {{ text-align: center; color: var(--muted); padding: 48px 12px; font-size: 14px; }}

/* سجل التنبيهات */
.alert-list {{ display: flex; flex-direction: column; gap: 12px; max-height: 440px; overflow-y: auto; padding-inline-end: 4px; margin-top: 10px; }}
.alert-item {{
  border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; background: #FFFFFF;
  border-inline-start: 4px solid var(--border);
  display: flex; flex-direction: column; gap: 8px;
}}
.alert-item.sev-critical {{ border-inline-start-color: #991B1B; background: linear-gradient({tr['crit_grad']}, #FFFFFF 0%, #FEF2F2 100%); }}
.alert-item.sev-danger {{ border-inline-start-color: #DC2626; }}
.alert-item.sev-warn {{ border-inline-start-color: #F59E0B; }}
.alert-top {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }}
.alert-badges {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.alert-cause {{ font-size: 14px; font-weight: 700; color: var(--text); line-height: 1.5; }}
.alert-rec {{
  padding: 8px 12px; border-radius: 9px; background: #F0F9FF; border: 1px solid #BAE6FD;
  color: #0C4A6E; font-size: 13px; line-height: 1.6;
}}
.alert-rec b {{ color: #075985; font-weight: 700; }}
.alert-meta {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }}
.alert-meta b {{ color: var(--text); font-weight: 600; }}

.disclaimer {{
  margin-top: 18px; padding: 14px 18px; border-radius: var(--radius); border: 1px solid #BAE6FD;
  background: var(--primary-soft); color: #075985; font-size: 13px; line-height: 1.8;
}}
</style>
"""
)

model: SmartMattressModel = st.session_state.model
current_scn: ScenarioKey = normalize_scenario_key(st.session_state.scenario)
scn_meta = SCENARIO_CATALOG[current_scn]
scn_label = get_scenario_label(current_scn, lang)
scn_desc = get_scenario_description(current_scn, lang)

# ──────────────────── الترويسة + زر تبديل اللغة (العربية | English) ────────────────────

tone_cls = f"badge-tone-{scn_meta['badge_tone']}"
scenario_badge = (
    f'<span class="badge {tone_cls}"><span class="dot"></span>'
    f'{escape(tr["badge_scenario"])}: {escape(scn_label)}</span>'
)
threshold_badge_txt = tr["badge_threshold"].format(n=model.consecutive_threshold)

hdr_col_main, hdr_col_lang = st.columns([4.8, 1.15], gap="medium", vertical_alignment="center")

with hdr_col_main:
    _html(
        f"""
<div class="app-header">
  <div class="brand">
    <div class="brand-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2"
           stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-5 4 10 2-5h6"/></svg>
    </div>
    <div>
      <div class="app-title">{escape(tr['app_title'])}</div>
      <div class="app-subtitle">{escape(tr['app_subtitle'])}</div>
    </div>
  </div>
  <div class="header-badges">
    {scenario_badge}
    <span class="badge">{escape(threshold_badge_txt)}</span>
    <span class="badge">{escape(tr['badge_sim'])}</span>
  </div>
</div>
"""
    )

with hdr_col_lang:
    with st.container(key="lang_switcher"):
        lc1, lc2 = st.columns(2, gap="small")
        with lc1:
            if st.button(
                "العربية",
                key="btn_lang_ar",
                type="primary" if lang == "ar" else "secondary",
                use_container_width=True,
            ):
                if lang != "ar":
                    st.session_state.lang = "ar"
                    st.rerun()
        with lc2:
            if st.button(
                "English",
                key="btn_lang_en",
                type="primary" if lang == "en" else "secondary",
                use_container_width=True,
            ):
                if lang != "en":
                    st.session_state.lang = "en"
                    st.rerun()

# ──────────────────── بطاقة الملف الصحي للمريض الافتراضي ────────────────────

with st.container(key="card_patient"):
    sample_html = tr["sample_size"].format(n=_num("600"))
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
      <div class="patient-title">{escape(tr['patient_title'])}</div>
      <div class="patient-sub">{escape(tr['patient_sub'])}</div>
    </div>
  </div>
  <div class="baseline-chips">
    <span class="b-chip">{escape(tr['chip_temp'])}: <b>{_num("33.2 – 34.8")} {escape(tr['unit_temp'])}</b></span>
    <span class="b-chip">{escape(tr['chip_hum'])}: <b>{_num("49 – 61")} %</b></span>
    <span class="b-chip">{escape(tr['chip_press'])}: <b>{_num("8 – 35")} {escape(tr['unit_press'])}</b> ({escape(tr['chip_hotspot'])} <b>{_num("45")}</b>)</span>
  </div>
</div>
<div class="patient-note">
  <span>{escape(tr['patient_disclaimer'])}</span>
  <span>{sample_html}</span>
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
                tr["batch_slider"],
                min_value=1,
                max_value=30,
                value=st.session_state.batch_size,
            )
            st.session_state.batch_size = n_steps

        with c_scn:
            selected_scn: ScenarioKey = st.selectbox(
                tr["select_scenario"],
                options=SCENARIO_ORDER,
                index=SCENARIO_ORDER.index(current_scn),
                format_func=lambda key: get_scenario_label(key, lang),
            )

        b_gen, b_reset = st.columns([1.45, 1], gap="small")
        with b_gen:
            generate_btn = st.button(tr["btn_generate"], type="primary", use_container_width=True)
        with b_reset:
            reset_btn = st.button(tr["btn_reset"], use_container_width=True)

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
            f'<span><b>{escape(tr["scn_desc_prefix"])}</b> {escape(scn_desc)}</span>'
            f'<span>{escape(tr["readings_count"])} <b>{_num(st.session_state.total_readings)}</b></span>'
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
            hvac_pill = f'<span class="pill pill-cool"><span class="dot"></span>{escape(tr["hvac_pill_cool"])}</span>'
        elif effective_preview == "heat":
            hvac_pill = f'<span class="pill pill-heat"><span class="dot"></span>{escape(tr["hvac_pill_heat"])}</span>'
        elif effective_preview == "off":
            hvac_pill = f'<span class="pill pill-neutral">{escape(tr["hvac_pill_off"])}</span>'
        else:
            hvac_pill = f'<span class="pill pill-ok"><span class="dot"></span>{escape(tr["hvac_pill_standby"])}</span>'

        _html(
            f'<div class="hvac-head">'
            f'<span class="hvac-title">{escape(tr["hvac_title"])}</span>'
            f"{hvac_pill}</div>"
        )

        hb_cols = st.columns(4, gap="small")
        cur_hvac: HvacControlMode = st.session_state.hvac_mode
        for idx, mode_key in enumerate(HVAC_ORDER):
            with hb_cols[idx]:
                if st.button(
                    tr["hvac_short"][mode_key],
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
            tr["target_slider"],
            min_value=32.5,
            max_value=36.0,
            value=float(st.session_state.target_temp),
            step=0.5,
        )
        if target_temp_val != st.session_state.target_temp:
            st.session_state.target_temp = target_temp_val

        _html(
            f'<div class="scn-desc-bar">'
            f'<span>{escape(tr["selected_mode"])} <b>{escape(tr["hvac_full"][cur_hvac])}</b></span>'
            f'<span>{escape(tr["safe_target"])} <b>{_num(f"{st.session_state.target_temp:.1f}")} {escape(tr["unit_temp"])}</b></span>'
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
loc_analysis = (
    localize_analysis(latest_result["analysis"], lang)
    if latest_result is not None
    else None
)
effective_climate = _resolve_effective_climate(
    st.session_state.hvac_mode,
    latest_reading,
    latest_result,
    st.session_state.auto_latched_mode,
)

# ──────────────────── صف بطاقات الملخص ────────────────────


def _delta_html(history: list[float], unit: str) -> str:
    if len(history) < 2:
        return escape(tr["delta_first"])
    diff = history[-1] - history[-2]
    if abs(diff) < 0.05:
        return escape(tr["delta_steady"])
    arrow, cls = ("▲", "up") if diff > 0 else ("▼", "down")
    return (
        f'<span class="delta {cls}">{_num(f"{arrow} {abs(diff):.1f}{unit}")}</span>'
        f'{escape(tr["delta_prev"])}'
    )


def _metric_card(label: str, accent: str, value: str, unit: str, sub: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-top"><span class="metric-label">'
        f'<span class="metric-accent" style="background:{accent}"></span>{escape(label)}</span></div>'
        f'<div><div class="metric-value"><bdi>{value}</bdi><span class="metric-unit">{escape(unit)}</span></div>'
        f'<div class="metric-sub">{sub}</div></div>'
        "</div>"
    )


if latest_reading is not None and latest_result is not None and loc_analysis is not None:
    temp_val = latest_reading.temperature
    if temp_val >= 36.0:
        temp_note = escape(tr["note_high_temp"]) + _delta_html(st.session_state.history_temp, "°")
    elif temp_val <= 32.2:
        temp_note = escape(tr["note_low_temp"]) + _delta_html(st.session_state.history_temp, "°")
    else:
        temp_note = _delta_html(st.session_state.history_temp, "°")

    hum_val = latest_reading.humidity
    if hum_val >= 68.0:
        hum_note = escape(tr["note_high_hum"]) + _delta_html(st.session_state.history_hum, "%")
    else:
        hum_note = _delta_html(st.session_state.history_hum, "%")

    temp_card = _metric_card(tr["card_temp"], "#F43F5E", f"{temp_val:.1f}", tr["unit_temp"], temp_note)
    hum_card = _metric_card(tr["card_hum"], PRIMARY, f"{hum_val:.1f}", "%", hum_note)

    top_mean_num = _num(f"{loc_analysis['top_zone_mean']:.1f}")
    if loc_analysis["affected_regions"]:
        zone_sub = tr["zone_hotspot"].format(region=escape(loc_analysis["region"]), val=top_mean_num)
    else:
        zone_sub = tr["zone_balanced"].format(zone=escape(loc_analysis["top_zone_name"]), val=top_mean_num)

    pressure_card = _metric_card(
        tr["card_press"],
        "#F59E0B",
        f"{latest_reading.pressure_matrix.max():.1f}",
        tr["unit_press"],
        zone_sub,
    )

    score_html = escape(tr["model_score"]) + _num(f"{latest_result['score']:.4f}")
    if latest_result["alert_triggered"]:
        status_cls = "status-danger"
        if loc_analysis["is_compound"]:
            pill = f'<span class="pill pill-critical">{escape(tr["pill_critical"])}</span>'
            status_value = tr["val_compound"]
        else:
            pill = f'<span class="pill pill-danger"><span class="dot"></span>{escape(tr["pill_active_alert"])}</span>'
            status_value = (
                loc_analysis["region"] if loc_analysis["affected_regions"] else loc_analysis["risk_type"]
            )
        status_sub = tr["sub_consecutive"].format(n=_num(latest_result["consecutive"]), score=score_html)
    elif latest_result["is_anomaly"]:
        status_cls = "status-warn"
        pill = f'<span class="pill pill-warn"><span class="dot"></span>{escape(tr["pill_monitoring"])}</span>'
        status_value = tr["val_anomaly"]
        status_sub = tr["sub_before_alert"].format(
            c=_num(latest_result["consecutive"]),
            t=_num(model.consecutive_threshold),
            score=score_html,
        )
    else:
        status_cls = "status-ok"
        pill = f'<span class="pill pill-ok"><span class="dot"></span>{escape(tr["pill_stable"])}</span>'
        status_value = tr["val_normal"]
        status_sub = tr["sub_normal"].format(score=score_html)
else:
    placeholder = escape(tr["placeholder_start"])
    temp_card = _metric_card(tr["card_temp"], "#F43F5E", "—", "", placeholder)
    hum_card = _metric_card(tr["card_hum"], PRIMARY, "—", "", placeholder)
    pressure_card = _metric_card(tr["card_press"], "#F59E0B", "—", "", placeholder)
    status_cls = ""
    pill = f'<span class="pill pill-neutral">{escape(tr["pill_waiting"])}</span>'
    status_value, status_sub = "—", placeholder

status_font = "21px" if len(status_value) > 16 else "25px"
status_card = (
    f'<div class="metric-card {status_cls}">'
    f'<div class="metric-top"><span class="metric-label">{escape(tr["card_status"])}</span>{pill}</div>'
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
            f"{get_cell_region(r, c, lang)} ({tr['hover_row_col'].format(r=r + 1, c=c + 1)})"
            for c in range(COLS)
        ]
        for r in range(ROWS)
    ]
    is_ar = lang == "ar"
    cbar_cfg = dict(
        title=dict(text=tr["unit_press"], side="top", font=dict(size=11, color=MUTED)),
        thickness=10,
        len=0.85,
        outlinewidth=0,
        x=-0.03 if is_ar else 1.03,
        xanchor="right" if is_ar else "left",
        tickfont=dict(size=10, color=MUTED),
    )

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
            hovertemplate=f"%{{customdata}}<br>{tr['hover_pressure']}: %{{z:.1f}} {tr['unit_press']}<extra></extra>",
            colorbar=cbar_cfg,
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
    fig.update_layout(margin=dict(l=70, r=8, t=8, b=8) if is_ar else dict(l=8, r=70, t=8, b=8))
    fig.update_yaxes(
        autorange="reversed",
        side="right" if is_ar else "left",
        tickmode="array",
        tickvals=list(range(ROWS)),
        ticktext=tr["row_labels"],
        tickfont=dict(size=11, color=MUTED),
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    fig.update_xaxes(
        autorange="reversed" if is_ar else True,
        tickmode="array",
        tickvals=list(range(COLS)),
        ticktext=tr["col_labels"],
        tickfont=dict(size=11, color=MUTED),
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    return fig


def _trend_chart(x, y, scenario_keys: list[str], color: str, unit: str, band: tuple[float, float]) -> go.Figure:
    localized_scenarios = [get_scenario_label(k, lang) for k in scenario_keys]
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor="rgba(2,132,199,0.08)", line_width=0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            customdata=localized_scenarios,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.55),
            marker=dict(size=6, color="#FFFFFF", line=dict(color=color, width=2)),
            hovertemplate=f"{tr['hover_reading']} %{{x}} (%{{customdata}})<br>%{{y:.1f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(_base_layout(248))
    fig.update_xaxes(
        title=dict(text=tr["axis_reading"], font=dict(size=11, color=MUTED)),
        gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(size=10.5, color=MUTED),
    )
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(size=10.5, color=MUTED))
    return fig


PLOT_CONFIG = {"displayModeBar": False}

col_visual, col_trends = st.columns([1.35, 1], gap="medium")

with col_visual:
    with st.container(key="card_visuals"):
        tab_3d, tab_2d = st.tabs([tr["tab_3d"], tr["tab_2d"]])

        with tab_3d:
            _html(
                '<div class="section-head">'
                f'<div class="section-title">{escape(tr["twin_title"])}</div>'
                f'<div class="section-sub">{escape(tr["twin_sub"])}</div>'
                "</div>"
            )
            html_3d = build_mattress_3d_html(
                pressure_matrix=latest_reading.pressure_matrix if latest_reading is not None else None,
                climate_visual=effective_climate,
                alert_triggered=bool(latest_result["alert_triggered"]) if latest_result is not None else False,
                affected_region=loc_analysis["region"] if loc_analysis is not None else tr["val_normal"],
                temperature=latest_reading.temperature if latest_reading is not None else 34.0,
                humidity=latest_reading.humidity if latest_reading is not None else 55.0,
                lang=lang,
            )
            components.html(html_3d, height=548, scrolling=False)

        with tab_2d:
            if loc_analysis is not None and loc_analysis["affected_regions"]:
                map_sub = tr["map2d_sub_hotspot"].format(region=escape(loc_analysis["region"]))
            else:
                map_sub = escape(tr["map2d_sub_normal"])

            _html(
                f'<div class="section-head"><div class="section-title">{escape(tr["map2d_title"])}</div>'
                f'<div class="section-sub">{map_sub}</div></div>'
            )
            if latest_reading is not None and loc_analysis is not None:
                st.plotly_chart(
                    _pressure_heatmap(
                        latest_reading.pressure_matrix,
                        loc_analysis["highlight_shapes"],
                    ),
                    use_container_width=True,
                    config=PLOT_CONFIG,
                )
                _html(f'<div class="zones-legend">{escape(tr["zones_legend"])}</div>')
            else:
                _html(f'<div class="empty-state">{escape(tr["empty_map"])}</div>')

with col_trends:
    with st.container(key="card_temp"):
        _html(
            f'<div class="section-head"><div class="section-title">{escape(tr["trend_temp_title"])}</div>'
            f'<div class="section-sub">{escape(tr["trend_temp_sub"])}</div></div>'
        )
        if st.session_state.history_temp:
            st.plotly_chart(
                _trend_chart(
                    st.session_state.history_index,
                    st.session_state.history_temp,
                    st.session_state.history_scenario,
                    "#F43F5E",
                    tr["unit_temp"],
                    TEMP_NORMAL_BAND,
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )
        else:
            _html(f'<div class="empty-state">{escape(tr["empty_data"])}</div>')

    with st.container(key="card_hum"):
        _html(
            f'<div class="section-head"><div class="section-title">{escape(tr["trend_hum_title"])}</div>'
            f'<div class="section-sub">{escape(tr["trend_hum_sub"])}</div></div>'
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
            _html(f'<div class="empty-state">{escape(tr["empty_data"])}</div>')

# ──────────────────── سجل التنبيهات وتوصيات التمريض ────────────────────

alerts = st.session_state.alerts_log
with st.container(key="card_alerts"):
    count_pill = (
        f'<span class="pill pill-danger">{escape(tr["alerts_count"].format(n=len(alerts)))}</span>'
        if alerts else f'<span class="pill pill-ok">{escape(tr["alerts_none"])}</span>'
    )
    _html(
        '<div class="section-head">'
        f'<div><span class="section-title">{escape(tr["alerts_title"])}</span></div>'
        f"{count_pill}</div>"
    )

    if alerts:
        items = []
        for alert in reversed(alerts):
            la = localize_alert(alert, lang)
            if la["severity_key"] == "critical":
                sev_cls, pill_cls = "sev-critical", "pill-critical"
            elif la["severity_key"] == "danger":
                sev_cls, pill_cls = "sev-danger", "pill-danger"
            else:
                sev_cls, pill_cls = "sev-warn", "pill-warn"

            consec_txt = tr["lbl_consec"].format(n=_num(la["consecutive"]))
            items.append(
                f'<div class="alert-item {sev_cls}">'
                f'<div class="alert-top">'
                f'<div class="alert-badges">'
                f'<span class="pill {pill_cls}">{escape(la["severity"])}</span>'
                f'<span class="pill pill-region">{escape(tr["lbl_region"])} {escape(la["region"])}</span>'
                f'<span class="pill pill-neutral">{consec_txt}</span>'
                f"</div>"
                f'<span class="alert-meta">{escape(tr["lbl_time"])} <b>{_num(escape(la["timestamp"]))}</b></span>'
                f"</div>"
                f'<div class="alert-cause">{escape(la["cause"])}</div>'
                f'<div class="alert-rec"><b>{escape(tr["lbl_nurse_rec"])}</b> {escape(la["recommendation"])}</div>'
                '<div class="alert-meta">'
                f'<span>{escape(tr["lbl_peak_p"])} <b>{_num(la["peak_pressure"])}</b> {escape(tr["unit_press"])}</span>'
                f'<span>{escape(tr["lbl_temp"])} <b>{_num(la["temperature"])}</b> {escape(tr["unit_temp"])}</span>'
                f'<span>{escape(tr["lbl_hum"])} <b>{_num(la["humidity"])}</b> %</span>'
                f'<span>{escape(tr["lbl_anom_score"])} <b>{_num(la["score"])}</b></span>'
                "</div></div>"
            )
        _html(f'<div class="alert-list">{"".join(items)}</div>')
    else:
        _html(f'<div class="empty-state" style="padding:28px 12px">{escape(tr["alerts_empty"])}</div>')

# ──────────────────── تذييل ────────────────────

_html(f'<div class="disclaimer">{tr["footer_disclaimer"]}</div>')

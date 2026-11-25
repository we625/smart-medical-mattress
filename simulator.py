"""
simulator.py – مولّد بيانات المحاكاة للمفرش الطبي الذكي (ثنائي اللغة: العربية / الإنجليزية)

يولّد:
  • مصفوفة ضغط 16×8 موزعة تشريحيًا على مناطق الجسم الكاملة
  • درجة حرارة سطح المفرش (°م / °C)
  • الرطوبة النسبية (%)

يعتمد مفاتيح داخلية ثابتة (Internal Keys) للسيناريوهات السبعة مع قاموس ترجمة منظم (ar / en).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

ROWS, COLS = 16, 8

LangType = Literal["ar", "en"]

ScenarioKey = Literal[
    "normal",
    "pelvis_pressure",
    "heels_pressure",
    "lateral_pressure",
    "heat_sweat",
    "cold_surface",
    "compound_risk",
]

# توافق خلفي في حال تمرير مفاتيح عربية قديمة
ScenarioType = ScenarioKey
_LEGACY_SCENARIO_MAP: dict[str, ScenarioKey] = {
    "طبيعي": "normal",
    "ضغط_حوض": "pelvis_pressure",
    "ضغط_كعبين": "heels_pressure",
    "ضغط_جانبي": "lateral_pressure",
    "تعرق_وحرارة": "heat_sweat",
    "برودة": "cold_surface",
    "خطر_مركب": "compound_risk",
}

SCENARIO_CATALOG: dict[ScenarioKey, dict] = {
    "normal": {
        "badge_tone": "normal",
        "label": {
            "ar": "السيناريو الطبيعي",
            "en": "Normal Baseline Scenario",
        },
        "description": {
            "ar": "توزيع ضغط متوازن على مناطق الجسم وحرارة ورطوبة ضمن النمط الطبيعي للمريض.",
            "en": "Balanced full-body pressure distribution with surface temperature and humidity within the patient's normal baseline.",
        },
    },
    "pelvis_pressure": {
        "badge_tone": "danger",
        "label": {
            "ar": "ضغط مستمر على الحوض",
            "en": "Sustained Pelvic Pressure",
        },
        "description": {
            "ar": "ارتفاع مستمر ومتصاعد للضغط في منطقة الحوض والعجز (الصفوف ٧–١٠).",
            "en": "Sustained, progressively rising pressure concentrated over the pelvis and sacral area (rows 7–10).",
        },
    },
    "heels_pressure": {
        "badge_tone": "danger",
        "label": {
            "ar": "ضغط على الكعبين والساقين",
            "en": "Heels & Lower Legs Pressure",
        },
        "description": {
            "ar": "تركّز الضغط في الصفوف السفلية لمنطقة الساقين والكعبين والقدمين (الصفوف ١٤–١٦).",
            "en": "Localized pressure buildup across the lower legs, heels, and feet (rows 14–16).",
        },
    },
    "lateral_pressure": {
        "badge_tone": "danger",
        "label": {
            "ar": "ضغط جانبي على الكتف والذراع واليد",
            "en": "Lateral Shoulder, Arm & Hand Pressure",
        },
        "description": {
            "ar": "يحاكي استلقاء المريض طويلاً على جانب واحد مع ارتفاع الضغط على الكتف والذراع واليد.",
            "en": "Simulates prolonged side-lying posture with elevated pressure along the shoulder, arm, and hand.",
        },
    },
    "heat_sweat": {
        "badge_tone": "warn",
        "label": {
            "ar": "تعرق ورطوبة وحرارة مرتفعة",
            "en": "High Moisture, Sweating & Heat",
        },
        "description": {
            "ar": "ارتفاع ملحوظ في رطوبة السطح وحرارته يستدعي التهوية والتبريد الآمن.",
            "en": "Elevated surface moisture and temperature requiring active ventilation and safe cooling.",
        },
    },
    "cold_surface": {
        "badge_tone": "cold",
        "label": {
            "ar": "برودة وانخفاض حرارة السطح",
            "en": "Cold Surface / Low Temperature",
        },
        "description": {
            "ar": "انخفاض غير معتاد في حرارة سطح المفرش يستدعي التدفئة الآمنة.",
            "en": "Unusual drop in mattress surface temperature requiring safe medical warming.",
        },
    },
    "compound_risk": {
        "badge_tone": "critical",
        "label": {
            "ar": "خطر مركّب (ضغط + رطوبة + حرارة)",
            "en": "Compound Risk (Pressure + Moisture + Heat)",
        },
        "description": {
            "ar": "سيناريو طارئ يجمع بين الضغط المستمر والتعرق وارتفاع الحرارة (أعلى خطر لتقرحات الجلد).",
            "en": "Emergency scenario combining sustained pelvic pressure with sweating and elevated heat (highest ulcer risk).",
        },
    },
}

SCENARIO_ORDER: list[ScenarioKey] = list(SCENARIO_CATALOG.keys())

CELL_REGION_LABELS: dict[str, dict[LangType, str]] = {
    "right_arm_hand": {"ar": "الذراع واليد اليمنى", "en": "Right Arm & Hand"},
    "right_shoulder_arm": {"ar": "الكتف والذراع الأيمن", "en": "Right Shoulder & Arm"},
    "left_arm_hand": {"ar": "الذراع واليد اليسرى", "en": "Left Arm & Hand"},
    "left_shoulder_arm": {"ar": "الكتف والذراع الأيسر", "en": "Left Shoulder & Arm"},
    "head": {"ar": "الرأس", "en": "Head"},
    "pillow": {"ar": "وسادة الرأس", "en": "Head Pillow"},
    "shoulders": {"ar": "الكتفان", "en": "Shoulders"},
    "right_shoulder": {"ar": "الكتف الأيمن", "en": "Right Shoulder"},
    "left_shoulder": {"ar": "الكتف الأيسر", "en": "Left Shoulder"},
    "back": {"ar": "الظهر", "en": "Back"},
    "pelvis": {"ar": "الحوض", "en": "Pelvis"},
    "pelvis_side": {"ar": "جانب الحوض", "en": "Lateral Hip"},
    "thighs": {"ar": "الفخذان", "en": "Thighs"},
    "thigh_side": {"ar": "جانب الفخذ", "en": "Outer Thigh"},
    "calves": {"ar": "الساقان", "en": "Lower Legs"},
    "calf_side": {"ar": "جانب الساق", "en": "Outer Leg"},
    "heels_feet": {"ar": "الكعبان والقدمان", "en": "Heels & Feet"},
    "foot_end": {"ar": "أسفل المفرش", "en": "Foot Margin"},
}


def normalize_scenario_key(scenario: str) -> ScenarioKey:
    """يحوّل أي مفتاح سيناريو (داخلي أو عربي قديم) إلى المفتاح الداخلي القياسي."""
    if scenario in SCENARIO_CATALOG:
        return scenario  # type: ignore[return-value]
    return _LEGACY_SCENARIO_MAP.get(scenario, "normal")


def get_scenario_label(scenario: str, lang: LangType = "ar") -> str:
    key = normalize_scenario_key(scenario)
    return SCENARIO_CATALOG[key]["label"][lang]


def get_scenario_description(scenario: str, lang: LangType = "ar") -> str:
    key = normalize_scenario_key(scenario)
    return SCENARIO_CATALOG[key]["description"][lang]


@dataclass
class SensorReading:
    """قراءة واحدة من حساسات المفرش الطبي الذكي."""
    pressure_matrix: np.ndarray          # shape (16, 8)
    temperature: float                   # °C
    humidity: float                      # %
    scenario: ScenarioKey


def get_cell_region_key(row: int, col: int) -> str:
    """يُحدّد المفتاح الداخلي للمنطقة التشريحية للخلية (row, col)."""
    if 3 <= row <= 8 and col in (0, 1):
        return "right_arm_hand" if row >= 4 else "right_shoulder_arm"
    if 3 <= row <= 8 and col in (6, 7):
        return "left_arm_hand" if row >= 4 else "left_shoulder_arm"

    if 0 <= row <= 1:
        return "head" if 2 <= col <= 5 else "pillow"
    if 2 <= row <= 3:
        if 1 <= col <= 6:
            return "shoulders"
        return "right_shoulder" if col == 0 else "left_shoulder"
    if 4 <= row <= 5:
        return "back"
    if 6 <= row <= 9:
        return "pelvis" if 2 <= col <= 5 else "pelvis_side"
    if 10 <= row <= 12:
        return "thighs" if 2 <= col <= 5 else "thigh_side"
    if row == 13:
        return "calves" if 2 <= col <= 5 else "calf_side"
    return "heels_feet" if 2 <= col <= 5 else "foot_end"


def get_cell_region(row: int, col: int, lang: LangType = "ar") -> str:
    """يُرجع اسم المنطقة التشريحية للخلية باللغة المطلوبة."""
    key = get_cell_region_key(row, col)
    return CELL_REGION_LABELS[key][lang]


def _base_pressure() -> np.ndarray:
    """
    يولّد توزيع ضغط أساسي واقعي لمريض مستلقٍ على الظهر بشكل متوازن
    على مصفوفة 16×8 مع تمثيل كافة مناطق الجسم.
    """
    mat = np.random.normal(loc=8.5, scale=1.4, size=(ROWS, COLS))

    # 1. الرأس (صفوف 1–2 -> 0:2، أعمدة وسطى 2:6)
    mat[0:2, 2:6] = np.random.normal(loc=21.0, scale=2.0, size=(2, 4))

    # 2. الكتفان (صفوف 3–4 -> 2:4، أعمدة 1:7)
    mat[2:4, 1:7] = np.random.normal(loc=27.0, scale=2.3, size=(2, 6))

    # 3. الظهر (صفوف 5–6 -> 4:6، أعمدة وسطى 2:6)
    mat[4:6, 2:6] = np.random.normal(loc=25.0, scale=2.2, size=(2, 4))

    # 4. الذراعان واليدان (صفوف 4–9 -> 3:9، أعمدة طرفية يمنى 0:2 ويسرى 6:8)
    mat[3:9, 0:2] = np.random.normal(loc=18.5, scale=2.0, size=(6, 2))
    mat[3:9, 6:8] = np.random.normal(loc=18.5, scale=2.0, size=(6, 2))
    mat[3, 1] = np.random.normal(loc=23.5, scale=2.0)
    mat[3, 6] = np.random.normal(loc=23.5, scale=2.0)

    # 5. الحوض (صفوف 7–10 -> 6:10، أعمدة وسطى 2:6)
    mat[6:10, 2:6] = np.random.normal(loc=29.5, scale=2.4, size=(4, 4))

    # 6. الفخذان (صفوف 11–13 -> 10:13، أعمدة وسطى 2:6)
    mat[10:13, 2:6] = np.random.normal(loc=23.5, scale=2.1, size=(3, 4))

    # 7. الساقان والكعبان/القدمان (صفوف 14–16 -> 13:16، أعمدة وسطى 2:6)
    mat[13:14, 2:6] = np.random.normal(loc=19.5, scale=1.9, size=(1, 4))
    mat[14:16, 2:6] = np.random.normal(loc=23.0, scale=2.2, size=(2, 4))

    return np.clip(mat, 4.0, 80.0)


def generate_reading(
    scenario: str = "normal",
    step: int = 0,
    climate_mode: Literal["off", "cool", "heat"] = "off",
    climate_steps: int = 0,
    target_temp: float = 34.0,
) -> SensorReading:
    """
    تولّد قراءة واحدة حسب مفتاح السيناريو الداخلي ورقم الخطوة الزمنية،
    مع محاكاة الأثر التدريجي الواقعي لوحدة التهوية والتبريد أو التدفئة الآمنة.
    """
    scn = normalize_scenario_key(scenario)
    pressure = _base_pressure()
    temp = np.random.normal(loc=34.0, scale=0.35)
    humidity = np.random.normal(loc=55.0, scale=2.8)

    if scn == "pelvis_pressure":
        intensity = min(35.0 + step * 2.0, 65.0)
        pressure[6:10, 2:6] = np.random.normal(
            loc=29.0 + intensity, scale=2.2, size=(4, 4)
        )
        temp = np.random.normal(loc=34.3, scale=0.35)
        humidity = np.random.normal(loc=56.5, scale=2.6)

    elif scn == "heels_pressure":
        calf_boost = min(26.0 + step * 1.5, 48.0)
        heel_boost = min(40.0 + step * 2.0, 69.0)
        pressure[13:14, 2:6] = np.random.normal(
            loc=20.0 + calf_boost, scale=2.0, size=(1, 4)
        )
        pressure[14:16, 2:6] = np.random.normal(
            loc=23.5 + heel_boost, scale=2.2, size=(2, 4)
        )
        temp = np.random.normal(loc=34.1, scale=0.35)
        humidity = np.random.normal(loc=55.5, scale=2.6)

    elif scn == "lateral_pressure":
        pressure[:, 5:8] = np.random.normal(loc=8.5, scale=1.3, size=(ROWS, 3))
        pressure[4:13, 2:5] *= 0.82
        lateral_boost = min(38.0 + step * 2.0, 66.0)
        pressure[2:4, 0:3] = np.random.normal(
            loc=27.5 + lateral_boost, scale=2.3, size=(2, 3)
        )
        pressure[3:9, 0:2] = np.random.normal(
            loc=24.0 + lateral_boost * 0.92, scale=2.2, size=(6, 2)
        )
        temp = np.random.normal(loc=34.4, scale=0.35)
        humidity = np.random.normal(loc=57.0, scale=2.6)

    elif scn == "heat_sweat":
        base_temp = min(36.8 + step * 0.24, 38.8)
        base_hum = min(74.0 + step * 1.7, 90.0)
        temp = np.random.normal(loc=base_temp, scale=0.25)
        humidity = np.random.normal(loc=base_hum, scale=2.0)

    elif scn == "cold_surface":
        base_temp = max(30.2 - step * 0.25, 28.0)
        temp = np.random.normal(loc=base_temp, scale=0.22)
        humidity = np.random.normal(loc=47.5, scale=2.2)

    elif scn == "compound_risk":
        intensity = min(40.0 + step * 2.2, 68.0)
        pressure[6:10, 2:6] = np.random.normal(
            loc=30.0 + intensity, scale=2.3, size=(4, 4)
        )
        pressure[5:6, 2:6] = np.random.normal(loc=41.0, scale=2.0, size=(1, 4))
        base_temp = min(37.4 + step * 0.25, 39.2)
        base_hum = min(78.0 + step * 1.8, 93.0)
        temp = np.random.normal(loc=base_temp, scale=0.25)
        humidity = np.random.normal(loc=base_hum, scale=2.0)

    safe_target = float(np.clip(target_temp, 32.5, 36.0))
    if climate_mode == "cool" and climate_steps > 0:
        alpha = min(climate_steps * 0.10, 0.80)
        cooled_goal = min(safe_target, temp - 0.3) if temp > safe_target else safe_target
        temp = temp * (1.0 - alpha) + cooled_goal * alpha
        if humidity > 53.0:
            hum_alpha = min(climate_steps * 0.12, 0.82)
            humidity = humidity * (1.0 - hum_alpha) + 53.0 * hum_alpha
    elif climate_mode == "heat" and climate_steps > 0:
        alpha = min(climate_steps * 0.10, 0.80)
        warm_goal = max(safe_target, temp + 0.3) if temp < safe_target else safe_target
        temp = temp * (1.0 - alpha) + warm_goal * alpha

    pressure = np.clip(pressure, 0.0, 120.0)
    temp = float(np.clip(temp, 20.0, 45.0))
    humidity = float(np.clip(humidity, 10.0, 100.0))

    return SensorReading(
        pressure_matrix=np.round(pressure, 1),
        temperature=round(temp, 1),
        humidity=round(humidity, 1),
        scenario=scn,
    )


def generate_batch(scenario: str = "normal",
                   n: int = 100) -> list[SensorReading]:
    """تولّد دفعة من القراءات المتتالية."""
    return [generate_reading(scenario, step=i) for i in range(n)]

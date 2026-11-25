"""
simulator.py – مولّد بيانات المحاكاة للمفرش الطبي الذكي

يولّد:
  • مصفوفة ضغط 16×8  (وحدة: ملم زئبق تقريبية) موزعة تشريحيًا على مناطق الجسم:
      - الرأس: الصفوف 1–2 (الأعمدة الوسطى 3–6)
      - الكتفان: الصفوف 3–4 (الأعمدة 2–7)
      - الظهر: الصفوف 5–6 (الأعمدة الوسطى 3–6)
      - الذراعان واليدان: الصفوف 4–9 (الأعمدة الطرفية اليمنى 1–2 واليسرى 7–8)
      - الحوض: الصفوف 7–10 (الأعمدة الوسطى 3–6)
      - الفخذان: الصفوف 11–13 (الأعمدة الوسطى 3–6)
      - الساقان والكعبان/القدمان: الصفوف 14–16 (الأعمدة الوسطى 3–6)
  • درجة حرارة سطح المفرش (°م)
  • الرطوبة النسبية (%)

يدعم 7 سيناريوهات محاكاة متكاملة لتقييم مخاطر تقرحات الفراش والظروف الحرارية.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

ROWS, COLS = 16, 8

ScenarioType = Literal[
    "طبيعي",
    "ضغط_حوض",
    "ضغط_كعبين",
    "ضغط_جانبي",
    "تعرق_وحرارة",
    "برودة",
    "خطر_مركب",
]

SCENARIO_CATALOG: dict[ScenarioType, dict[str, str]] = {
    "طبيعي": {
        "label": "السيناريو الطبيعي",
        "description": "توزيع ضغط متوازن على مناطق الجسم وحرارة ورطوبة ضمن النمط الطبيعي للمريض.",
        "badge_tone": "normal",
    },
    "ضغط_حوض": {
        "label": "ضغط مستمر على الحوض",
        "description": "ارتفاع مستمر ومتصاعد للضغط في منطقة الحوض والعجز (الصفوف ٧–١٠).",
        "badge_tone": "danger",
    },
    "ضغط_كعبين": {
        "label": "ضغط على الكعبين والساقين",
        "description": "تركّز الضغط في الصفوف السفلية لمنطقة الساقين والكعبين والقدمين (الصفوف ١٤–١٦).",
        "badge_tone": "danger",
    },
    "ضغط_جانبي": {
        "label": "ضغط جانبي على الكتف والذراع واليد",
        "description": "يحاكي استلقاء المريض طويلاً على جانب واحد مع ارتفاع الضغط على الكتف والذراع واليد.",
        "badge_tone": "danger",
    },
    "تعرق_وحرارة": {
        "label": "تعرق ورطوبة وحرارة مرتفعة",
        "description": "ارتفاع ملحوظ في رطوبة السطح وحرارته يستدعي التهوية والتبريد الآمن.",
        "badge_tone": "warn",
    },
    "برودة": {
        "label": "برودة وانخفاض حرارة السطح",
        "description": "انخفاض غير معتاد في حرارة سطح المفرش يستدعي التدفئة الآمنة.",
        "badge_tone": "cold",
    },
    "خطر_مركب": {
        "label": "خطر مركّب (ضغط + رطوبة + حرارة)",
        "description": "سيناريو طارئ يجمع بين الضغط المستمر والتعرق وارتفاع الحرارة (أعلى خطر لتقرحات الجلد).",
        "badge_tone": "critical",
    },
}

SCENARIO_ORDER: list[ScenarioType] = list(SCENARIO_CATALOG.keys())


@dataclass
class SensorReading:
    """قراءة واحدة من حساسات المفرش الطبي الذكي."""
    pressure_matrix: np.ndarray          # shape (16, 8)
    temperature: float                   # °م
    humidity: float                      # %
    scenario: ScenarioType


def get_cell_region(row: int, col: int) -> str:
    """
    يُحدّد اسم المنطقة التشريحية للخلية (row, col) بنظام فهرسة يبدأ من 0:
      - الرأس: صف 0–1 (صف 1–2)
      - الكتفان: صف 2–3 (صف 3–4)
      - الذراعان واليدان: الأعمدة الطرفية (0–1 يمين، 6–7 يسار) في الصفوف 3–8 (صف 4–9)
      - الظهر: صف 4–5 (صف 5–6)
      - الحوض: صف 6–9 (صف 7–10)
      - الفخذان: صف 10–12 (صف 11–13)
      - الساقان والكعبان/القدمان: صف 13–15 (صف 14–16)
    """
    # الذراعان واليدان في الأعمدة الطرفية للصفوف 4–9 (indices 3..8)
    if 3 <= row <= 8 and col in (0, 1):
        return "الذراع واليد اليمنى" if row >= 4 else "الكتف والذراع الأيمن"
    if 3 <= row <= 8 and col in (6, 7):
        return "الذراع واليد اليسرى" if row >= 4 else "الكتف والذراع الأيسر"

    if 0 <= row <= 1:
        return "الرأس" if 2 <= col <= 5 else "وسادة الرأس"
    if 2 <= row <= 3:
        if 1 <= col <= 6:
            return "الكتفان"
        return "الكتف الأيمن" if col == 0 else "الكتف الأيسر"
    if 4 <= row <= 5:
        return "الظهر"
    if 6 <= row <= 9:
        return "الحوض" if 2 <= col <= 5 else "جانب الحوض"
    if 10 <= row <= 12:
        return "الفخذان" if 2 <= col <= 5 else "جانب الفخذ"
    if row == 13:
        return "الساقان" if 2 <= col <= 5 else "جانب الساق"
    return "الكعبان والقدمان" if 2 <= col <= 5 else "أسفل المفرش"


def _base_pressure() -> np.ndarray:
    """
    يولّد توزيع ضغط أساسي واقعي لمريض مستلقٍ على الظهر بشكل متوازن
    على مصفوفة 16×8 مع تمثيل كافة مناطق الجسم.
    """
    # خلفية المفرش غير المحمّلة
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
    # نقطة التقاء الكتف مع أعلى الذراع (صف 4 -> index 3، عمود 1 و 6)
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
    scenario: ScenarioType = "طبيعي",
    step: int = 0,
    climate_mode: Literal["off", "cool", "heat"] = "off",
    climate_steps: int = 0,
    target_temp: float = 34.0,
) -> SensorReading:
    """
    تولّد قراءة واحدة حسب السيناريو المختار ورقم الخطوة الزمنية،
    مع محاكاة الأثر التدريجي الواقعي لوحدة التهوية والتبريد أو التدفئة الآمنة عند تفعيلها.
    """
    pressure = _base_pressure()
    temp = np.random.normal(loc=34.0, scale=0.35)
    humidity = np.random.normal(loc=55.0, scale=2.8)

    if scenario == "ضغط_حوض":
        # ارتفاع مستمر في منطقة الحوض (صفوف 7–10 -> 6:10، أعمدة 2:6)
        intensity = min(35.0 + step * 2.0, 65.0)
        pressure[6:10, 2:6] = np.random.normal(
            loc=29.0 + intensity, scale=2.2, size=(4, 4)
        )
        temp = np.random.normal(loc=34.3, scale=0.35)
        humidity = np.random.normal(loc=56.5, scale=2.6)

    elif scenario == "ضغط_كعبين":
        # ارتفاع الضغط على الساقين والكعبين/القدمين (صفوف 14–16 -> 13:16، أعمدة 2:6)
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

    elif scenario == "ضغط_جانبي":
        # استلقاء جانبي طويل (على الجانب الأيمن): تخفيف الجانب المقابل وتركز الحمل على الكتف والذراع واليد
        pressure[:, 5:8] = np.random.normal(loc=8.5, scale=1.3, size=(ROWS, 3))
        pressure[4:13, 2:5] *= 0.82
        lateral_boost = min(38.0 + step * 2.0, 66.0)
        # الكتف الأيمن (صفوف 3–4 -> 2:4، أعمدة 0:3)
        pressure[2:4, 0:3] = np.random.normal(
            loc=27.5 + lateral_boost, scale=2.3, size=(2, 3)
        )
        # الذراع واليد اليمنى (صفوف 4–9 -> 3:9، أعمدة 0:2)
        pressure[3:9, 0:2] = np.random.normal(
            loc=24.0 + lateral_boost * 0.92, scale=2.2, size=(6, 2)
        )
        temp = np.random.normal(loc=34.4, scale=0.35)
        humidity = np.random.normal(loc=57.0, scale=2.6)

    elif scenario == "تعرق_وحرارة":
        # ضغط متوازن طبيعي مع ارتفاع ملحوظ في الحرارة والرطوبة السطحية
        base_temp = min(36.8 + step * 0.24, 38.8)
        base_hum = min(74.0 + step * 1.7, 90.0)
        temp = np.random.normal(loc=base_temp, scale=0.25)
        humidity = np.random.normal(loc=base_hum, scale=2.0)

    elif scenario == "برودة":
        # ضغط متوازن طبيعي مع انخفاض غير معتاد في حرارة سطح المفرش
        base_temp = max(30.2 - step * 0.25, 28.0)
        temp = np.random.normal(loc=base_temp, scale=0.22)
        humidity = np.random.normal(loc=47.5, scale=2.2)

    elif scenario == "خطر_مركب":
        # سيناريو طارئ: ضغط مرتفع على الحوض + تعرق ورطوبة عالية + حرارة مرتفعة
        intensity = min(40.0 + step * 2.2, 68.0)
        pressure[6:10, 2:6] = np.random.normal(
            loc=30.0 + intensity, scale=2.3, size=(4, 4)
        )
        # انتقال جزئي للحمل نحو أسفل الظهر (صف 6 -> index 5)
        pressure[5:6, 2:6] = np.random.normal(loc=41.0, scale=2.0, size=(1, 4))
        base_temp = min(37.4 + step * 0.25, 39.2)
        base_hum = min(78.0 + step * 1.8, 93.0)
        temp = np.random.normal(loc=base_temp, scale=0.25)
        humidity = np.random.normal(loc=base_hum, scale=2.0)

    # الأثر التدريجي لوحدة التهوية والتبريد أو التدفئة الآمنة على الحرارة والرطوبة السطحية
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
        scenario=scenario,
    )


def generate_batch(scenario: ScenarioType = "طبيعي",
                   n: int = 100) -> list[SensorReading]:
    """تولّد دفعة من القراءات المتتالية."""
    return [generate_reading(scenario, step=i) for i in range(n)]

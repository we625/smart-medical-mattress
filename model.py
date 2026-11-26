"""
model.py – نموذج كشف الشذوذ والتتبع الزمني (60 دقيقة / ساعة متواصلة) وفلترة الإنذارات الكاذبة

يستخدم Isolation Forest من scikit-learn:
  • يتدرب على 600 قراءة من السيناريو الطبيعي فقط.
  • يكشف الشذوذ في توزيع الضغط (في أي منطقة من الجسم) أو الحرارة أو الرطوبة.
  • يطبّق فلترة ذكية لمنع الإنذارات الكاذبة: إذا تحرّك المريض أو تقلّب بشكل عابر لثوانٍ أو دقائق،
    يُعدّ ذلك حركة طبيعية لا تزيد عدّاد الوقت المستمر ولا تُطلق إنذاراً.
  • يتتبع استمرار الضغط أو الشذوذ على نفس المنطقة بفترات مدة كل منها 15 دقيقة في المحاكاة المُسرّعة،
    ويُطلق التنبيه التمريضي فور بلوغ 4 فترات متواصلة (ساعة متواصلة = 60 دقيقة).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

from simulator import STEP_MINUTES, LangType, SensorReading, generate_batch

# ───────── عتبات تفسير المؤشرات السطحية والتتبع الزمني ─────────

DEFAULT_SUSTAINED_PERIODS = 4                           # 4 فترات متتالية
DEFAULT_SUSTAINED_MINUTES = DEFAULT_SUSTAINED_PERIODS * STEP_MINUTES  # 60 دقيقة (ساعة متواصلة)
ZONE_PRESSURE_THRESHOLD = 45.0                          # mmHg
HIGH_TEMP_THRESHOLD = 36.0                              # °C
LOW_TEMP_THRESHOLD = 32.2                               # °C
HIGH_HUMIDITY_THRESHOLD = 68.0                          # %

ZONE_NAMES: dict[str, dict[LangType, str]] = {
    "pelvis": {"ar": "الحوض", "en": "Pelvis"},
    "heels_legs": {"ar": "الكعبين والساقين", "en": "Heels & Lower Legs"},
    "right_lateral": {"ar": "الكتف والذراع واليد (الأيمن)", "en": "Right Shoulder, Arm & Hand"},
    "left_lateral": {"ar": "الكتف والذراع واليد (الأيسر)", "en": "Left Shoulder, Arm & Hand"},
    "shoulders": {"ar": "الكتفين", "en": "Shoulders"},
    "back": {"ar": "الظهر", "en": "Back"},
    "thighs": {"ar": "الفخذين", "en": "Thighs"},
    "head": {"ar": "الرأس", "en": "Head"},
    "full_surface": {"ar": "سطح المفرش العام", "en": "Overall Mattress Surface"},
    "balanced": {"ar": "توزيع متوازن", "en": "Balanced Distribution"},
}

RISK_LABELS: dict[str, dict[LangType, str]] = {
    "compound": {"ar": "خطر مركّب", "en": "Compound Risk"},
    "localized_pressure": {"ar": "ضغط موضعي", "en": "Localized Pressure"},
    "heat_humidity": {"ar": "حرارة ورطوبة", "en": "Heat & Moisture"},
    "cold_surface": {"ar": "برودة سطحية", "en": "Cold Surface"},
    "transient_motion": {"ar": "حركة عابرة", "en": "Transient Motion"},
    "normal": {"ar": "طبيعي", "en": "Normal"},
    "general_anomaly": {"ar": "انحراف عام", "en": "Pattern Deviation"},
}

SEVERITY_LABELS: dict[str, dict[LangType, str]] = {
    "critical": {"ar": "خطر حرج", "en": "Critical"},
    "danger": {"ar": "خطر", "en": "High Risk"},
    "warn": {"ar": "تحذير", "en": "Warning"},
}

PRESSURE_RECOMMENDATIONS: dict[str, dict[LangType, str]] = {
    "pelvis": {
        "ar": "تغيير وضعية المريض الآن (إمالة جانبية بزاوية ٣٠°) لتخفيف الضغط المستمر عن منطقة الحوض والعجز.",
        "en": "Reposition the patient now (30° lateral tilt) to relieve sustained pressure off the pelvis and sacrum.",
    },
    "heels_legs": {
        "ar": "تغيير وضعية المريض الآن ورفع الكعبين عن سطح المفرش بوضع وسادة داعمة أسفل الساقين.",
        "en": "Reposition now and offload the heels by placing a supportive pillow beneath the calves.",
    },
    "right_lateral": {
        "ar": "تغيير وضعية المريض الآن من الاستلقاء الجانبي إلى وضعية متوازنة وتحرير الكتف والذراع واليد من الضغط.",
        "en": "Reposition the patient now from prolonged side-lying to a balanced posture and offload the right shoulder, arm, and hand.",
    },
    "left_lateral": {
        "ar": "تغيير وضعية المريض الآن من الاستلقاء الجانبي إلى وضعية متوازنة وتحرير الكتف والذراع واليد من الضغط.",
        "en": "Reposition the patient now from prolonged side-lying to a balanced posture and offload the left shoulder, arm, and hand.",
    },
    "shoulders": {
        "ar": "تغيير وضعية المريض وتعديل زاوية الجزء العلوي للجسم لإعادة توزيع الارتكاز عن منطقة الكتفين.",
        "en": "Reposition the patient and adjust upper-body elevation angle to redistribute weight off the shoulders.",
    },
    "back": {
        "ar": "إمالة المريض جزئيًا بوسائد ساندة الآن لتخفيف الضغط المتواصل عن منطقة الظهر.",
        "en": "Apply lateral support pillows now to relieve continuous pressure along the back.",
    },
}


def format_sustained_duration(minutes: int, lang: LangType = "ar") -> str:
    """يصوغ مدة الاستمرار بالعربية والإنجليزية (60 دقيقة -> ساعة متواصلة (60 دقيقة) / 1 continuous hour (60 min))."""
    mins = max(int(minutes), 0)
    if lang == "en":
        if mins == 60:
            return "1 continuous hour (60 min)"
        if mins == 120:
            return "2 continuous hours (120 min)"
        if mins > 120 and mins % 60 == 0:
            return f"{mins // 60} continuous hours ({mins} min)"
        return f"{mins} continuous min"

    if mins == 60:
        return "ساعة متواصلة (60 دقيقة)"
    if mins == 120:
        return "ساعتين متواصلتين (120 دقيقة)"
    if mins > 120 and mins % 60 == 0:
        hrs = mins // 60
        hr_word = "ساعات متواصلة" if 3 <= hrs <= 10 else "ساعة متواصلة"
        return f"{hrs} {hr_word} ({mins} دقيقة)"
    return f"{mins} دقيقة متواصلة"


def _zone_slices(mat: np.ndarray) -> list[dict]:
    """يُرجع قائمة المناطق التشريحية مع قيم الضغط وإحداثيات التظليل."""
    right_lateral = np.concatenate([mat[2:4, 0:3].ravel(), mat[4:9, 0:2].ravel()])
    left_lateral = np.concatenate([mat[2:4, 5:8].ravel(), mat[4:9, 6:8].ravel()])
    heels_legs = mat[13:16, 2:6]
    heels_only = mat[14:16, 2:6]

    return [
        {
            "key": "pelvis",
            "name": ZONE_NAMES["pelvis"]["ar"],
            "mean": float(mat[6:10, 2:6].mean()),
            "max": float(mat[6:10, 2:6].max()),
            "shape": {"type": "rect", "x0": 1.55, "y0": 5.55, "x1": 5.45, "y1": 9.45},
        },
        {
            "key": "heels_legs",
            "name": ZONE_NAMES["heels_legs"]["ar"],
            "mean": float(max(heels_legs.mean(), heels_only.mean() * 0.95)),
            "max": float(heels_legs.max()),
            "shape": {"type": "rect", "x0": 1.55, "y0": 12.55, "x1": 5.45, "y1": 15.45},
        },
        {
            "key": "right_lateral",
            "name": ZONE_NAMES["right_lateral"]["ar"],
            "mean": float(right_lateral.mean()),
            "max": float(right_lateral.max()),
            "shape": {
                "type": "path",
                "path": "M -0.45,1.55 L 2.45,1.55 L 2.45,3.45 L 1.45,3.45 L 1.45,8.45 L -0.45,8.45 Z",
            },
        },
        {
            "key": "left_lateral",
            "name": ZONE_NAMES["left_lateral"]["ar"],
            "mean": float(left_lateral.mean()),
            "max": float(left_lateral.max()),
            "shape": {
                "type": "path",
                "path": "M 5.55,1.55 L 7.45,1.55 L 7.45,8.45 L 5.55,8.45 L 5.55,3.45 L 4.55,3.45 L 4.55,1.55 Z",
            },
        },
        {
            "key": "shoulders",
            "name": ZONE_NAMES["shoulders"]["ar"],
            "mean": float(mat[2:4, 1:7].mean()),
            "max": float(mat[2:4, 1:7].max()),
            "shape": {"type": "rect", "x0": 0.55, "y0": 1.55, "x1": 6.45, "y1": 3.45},
        },
        {
            "key": "back",
            "name": ZONE_NAMES["back"]["ar"],
            "mean": float(mat[4:6, 2:6].mean()),
            "max": float(mat[4:6, 2:6].max()),
            "shape": {"type": "rect", "x0": 1.55, "y0": 3.55, "x1": 5.45, "y1": 5.45},
        },
        {
            "key": "thighs",
            "name": ZONE_NAMES["thighs"]["ar"],
            "mean": float(mat[10:13, 2:6].mean()),
            "max": float(mat[10:13, 2:6].max()),
            "shape": {"type": "rect", "x0": 1.55, "y0": 9.55, "x1": 5.45, "y1": 12.45},
        },
        {
            "key": "head",
            "name": ZONE_NAMES["head"]["ar"],
            "mean": float(mat[0:2, 2:6].mean()),
            "max": float(mat[0:2, 2:6].max()),
            "shape": {"type": "rect", "x0": 1.55, "y0": -0.45, "x1": 5.45, "y1": 1.45},
        },
    ]


# ───────── استخراج السمات (Feature Engineering) ─────────

def extract_features(reading: SensorReading) -> np.ndarray:
    """
    يحوّل قراءة واحدة إلى متجه من 7 سمات متوازنة تغطي كافة مناطق الضغط والحرارة والرطوبة.
    """
    mat = reading.pressure_matrix
    zones = _zone_slices(mat)
    max_zone_mean = max(z["mean"] for z in zones)

    return np.array([
        float(max_zone_mean),
        float(np.percentile(mat, 97)),
        float(mat.max()),
        float(reading.temperature),
        float(abs(reading.temperature - 34.0)),
        float(reading.humidity),
        float(abs(reading.humidity - 55.0)),
    ])


def extract_features_batch(readings: list[SensorReading]) -> np.ndarray:
    return np.vstack([extract_features(r) for r in readings])


# ───────── التحليل التشريحي والسريري وبناء النصوص المترجمة ─────────

def _pressure_recommendation(zone_key: str, lang: LangType = "ar") -> str:
    if zone_key in PRESSURE_RECOMMENDATIONS:
        return PRESSURE_RECOMMENDATIONS[zone_key][lang]
    zname = ZONE_NAMES.get(zone_key, {"ar": zone_key, "en": zone_key})[lang]
    if lang == "en":
        return f"Reposition the patient now and redistribute support points to relieve pressure on the {zname}."
    return f"تغيير وضعية المريض الآن وإعادة توزيع نقاط الارتكاز لتخفيف الضغط عن منطقة {zname}."


def _build_clinical_texts(
    risk_key: str,
    elevated_keys: list[str],
    top_zone_key: str,
    primary_mean: float,
    primary_max: float,
    temp: float,
    hum: float,
    high_temp: bool,
    high_hum: bool,
    lang: LangType,
    sustained_minutes: int = 0,
    is_transient: bool = False,
    transient_seconds: int = 0,
    alert_mode: bool = False,
) -> tuple[str, str, str, str]:
    """يُرجع (region_str, risk_label, cause, recommendation) باللغة المطلوبة."""
    sep = "، " if lang == "ar" else ", "
    affected_labels = [ZONE_NAMES[k][lang] for k in elevated_keys]

    if is_transient:
        risk_label = RISK_LABELS["transient_motion"][lang]
        region_str = sep.join(affected_labels) if affected_labels else ZONE_NAMES["balanced"][lang]
        secs = transient_seconds if transient_seconds > 0 else 30
        if lang == "en":
            cause = (
                f"Brief natural movement on {region_str} ({secs}s momentary spike) — "
                "Ignored by false-alarm filter (0 sustained min)"
            )
            recommendation = (
                "No nursing action required; brief posture shifts are healthy natural movements "
                "and do not increment the sustained time counter."
            )
        else:
            cause = (
                f"حركة طبيعية عابرة على {region_str} (ارتفاع لحظي لمدة {secs} ثانية) — "
                "تم تجاهلها لمنع الإنذار الكاذب (٠ دقيقة مستمرة)"
            )
            recommendation = (
                "لا يلزم أي تدخل تمريضي؛ التقلّب العابر القصير حركة طبيعية ولا يزيد عدّاد الوقت المستمر."
            )
        return region_str, risk_label, cause, recommendation

    risk_label = RISK_LABELS.get(risk_key, RISK_LABELS["normal"])[lang]
    eff_mins = max(int(sustained_minutes), DEFAULT_SUSTAINED_MINUTES) if alert_mode else int(sustained_minutes)
    dur_str = format_sustained_duration(eff_mins, lang) if eff_mins > 0 else ""

    if risk_key == "compound":
        region_str = sep.join(affected_labels)
        if lang == "en":
            env_parts = []
            if high_temp:
                env_parts.append(f"elevated heat ({temp:.1f} °C)")
            if high_hum:
                env_parts.append(f"sweating & high moisture ({hum:.1f}%)")
            if alert_mode:
                cause = (
                    f"Sustained pressure on {region_str} for {dur_str} "
                    f"with {' and '.join(env_parts)} — Reposition patient now"
                )
            else:
                cause = (
                    f"Compound skin-ulcer risk: sustained pressure on {region_str} "
                    f"({primary_mean:.1f} mmHg) combined with {' and '.join(env_parts)}"
                )
            recommendation = (
                f"Immediate nursing action: reposition patient now to offload {region_str}, "
                "activate safe cooling & ventilation, dry the skin, and inspect contact tissue."
            )
        else:
            env_parts = []
            if high_temp:
                env_parts.append(f"حرارة مرتفعة ({temp:.1f} °م)")
            if high_hum:
                env_parts.append(f"تعرق ورطوبة مرتفعة ({hum:.1f}%)")
            if alert_mode:
                cause = (
                    f"استمر تركز الضغط على {region_str} لمدة {dur_str} متزامناً مع "
                    f"{' و'.join(env_parts)} — يجب تغيير وضعية المريض الآن"
                )
            else:
                cause = (
                    f"خطر مركّب لتقرحات الجلد: ضغط مستمر على {region_str} "
                    f"({primary_mean:.1f} ملم زئبق) متزامن مع {' و'.join(env_parts)}"
                )
            recommendation = (
                f"تدخّل تمريضي فوري: تغيير وضعية المريض الآن لتخفيف الضغط عن {region_str}، "
                "وتفعيل التهوية والتبريد الآمن مع تجفيف الجلد وفحص موضع التلامس."
            )

    elif risk_key == "localized_pressure":
        region_str = sep.join(affected_labels)
        primary_key = elevated_keys[0]
        if lang == "en":
            if alert_mode:
                cause = f"Sustained pressure on {region_str} for {dur_str} — Reposition patient now"
            else:
                cause = (
                    f"Elevated pressure on {region_str} "
                    f"(mean {primary_mean:.1f} mmHg, peak {primary_max:.1f} mmHg)"
                )
        else:
            if alert_mode:
                cause = f"استمر تركز الضغط على {region_str} لمدة {dur_str} — يجب تغيير وضعية المريض الآن"
            else:
                cause = (
                    f"تركّز ضغط مرتفع على منطقة {region_str} "
                    f"(متوسط {primary_mean:.1f} ملم زئبق، الأقصى {primary_max:.1f} ملم زئبق)"
                )
        recommendation = _pressure_recommendation(primary_key, lang)

    elif risk_key == "heat_humidity":
        region_str = ZONE_NAMES["full_surface"][lang]
        if lang == "en":
            if alert_mode:
                cause = (
                    f"Sustained elevated moisture/heat on {region_str} ({hum:.1f}% · {temp:.1f} °C) "
                    f"for {dur_str} — Activate safe cooling & inspect skin now"
                )
            elif high_temp and high_hum:
                cause = f"Sweating with elevated surface moisture ({hum:.1f}%) and temperature ({temp:.1f} °C)"
            elif high_hum:
                cause = f"Elevated mattress surface moisture ({hum:.1f}%)"
            else:
                cause = f"Elevated mattress surface temperature ({temp:.1f} °C)"
            recommendation = (
                "Activate safe mattress ventilation & cooling, and verify skin and linen dryness to prevent maceration."
            )
        else:
            if alert_mode:
                cause = (
                    f"استمر ارتفاع رطوبة وحرارة {region_str} ({hum:.1f}% · {temp:.1f} °م) "
                    f"لمدة {dur_str} — يجب تفعيل التهوية والتبريد الآمن وفحص الجلد الآن"
                )
            elif high_temp and high_hum:
                cause = f"تعرق وارتفاع ملحوظ في رطوبة السطح ({hum:.1f}%) وحرارته ({temp:.1f} °م)"
            elif high_hum:
                cause = f"ارتفاع ملحوظ في رطوبة سطح المفرش ({hum:.1f}%)"
            else:
                cause = f"ارتفاع ملحوظ في حرارة سطح المفرش ({temp:.1f} °م)"
            recommendation = "تفعيل التهوية والتبريد الآمن للمفرش، والتحقق من جفاف الجلد والأغطية للوقاية من تهيّج البشرة."

    elif risk_key == "cold_surface":
        region_str = ZONE_NAMES["full_surface"][lang]
        if lang == "en":
            if alert_mode:
                cause = (
                    f"Sustained low temperature on {region_str} ({temp:.1f} °C) "
                    f"for {dur_str} — Activate safe warming & check patient now"
                )
            else:
                cause = f"Unusual coolness and drop in mattress surface temperature ({temp:.1f} °C)"
            recommendation = "Activate safe mattress warming and check the patient's blankets and core thermal comfort."
        else:
            if alert_mode:
                cause = (
                    f"استمر انخفاض حرارة {region_str} ({temp:.1f} °م) "
                    f"لمدة {dur_str} — يجب تفعيل التدفئة الآمنة وتفقّد المريض الآن"
                )
            else:
                cause = f"برودة وانخفاض غير معتاد في حرارة سطح المفرش ({temp:.1f} °م)"
            recommendation = "تفعيل التدفئة الآمنة للمفرش وتفقّد أغطية المريض ودرجة حرارته الحيوية."

    elif risk_key == "general_anomaly":
        top_name = ZONE_NAMES[top_zone_key][lang]
        if lang == "en":
            region_str = f"Peak Zone: {top_name}"
            if alert_mode:
                cause = f"Sustained pattern deviation near {top_name} for {dur_str} — Reposition patient now"
            else:
                cause = "Pattern deviation from the patient's learned normal baseline"
            recommendation = "Check patient posture and review surface pressure, temperature, and humidity indicators."
        else:
            region_str = f"أعلى ضغط: {top_name}"
            if alert_mode:
                cause = f"استمر انحراف نمط القراءات عند {top_name} لمدة {dur_str} — يجب تغيير وضعية المريض الآن"
            else:
                cause = "انحراف في نمط القراءات عن خط الأساس الطبيعي للمريض"
            recommendation = "تفقّد وضعية المريض ومراجعة مؤشرات الضغط والحرارة والرطوبة."

    else:
        region_str = ZONE_NAMES["balanced"][lang]
        if lang == "en":
            cause = "Sensor readings are within the patient's balanced normal baseline"
            recommendation = "Continue routine periodic monitoring; no immediate intervention required."
        else:
            cause = "القراءات ضمن النمط الطبيعي المتوازن للمريض"
            recommendation = "استمرار المراقبة الدورية المعتادة دون حاجة لتدخل إضافي."

    return region_str, risk_label, cause, recommendation


def analyze_reading(reading: SensorReading, lang: LangType = "ar") -> dict:
    """يحلّل القراءة ويُرجع المفاتيح الداخلية مع النصوص حسب اللغة."""
    zones = _zone_slices(reading.pressure_matrix)
    zones_sorted = sorted(zones, key=lambda z: z["mean"], reverse=True)
    top_zone = zones_sorted[0]

    elevated = [z for z in zones_sorted if z["mean"] >= ZONE_PRESSURE_THRESHOLD]
    if any(z["key"] in ("right_lateral", "left_lateral") for z in elevated):
        elevated = [z for z in elevated if z["key"] != "shoulders"]

    high_temp = reading.temperature >= HIGH_TEMP_THRESHOLD
    low_temp = reading.temperature <= LOW_TEMP_THRESHOLD
    high_hum = reading.humidity >= HIGH_HUMIDITY_THRESHOLD

    has_pressure = len(elevated) > 0
    elevated_keys = [z["key"] for z in elevated]
    highlight_shapes = [z["shape"] for z in elevated]

    dur_mins = int(getattr(reading, "duration_minutes", STEP_MINUTES))
    is_transient = bool(getattr(reading, "is_transient", False) or dur_mins < STEP_MINUTES)
    transient_seconds = int(getattr(reading, "transient_seconds", 0))

    if has_pressure and (high_temp or high_hum):
        risk_key = "compound"
    elif has_pressure:
        risk_key = "localized_pressure"
    elif high_temp or high_hum:
        risk_key = "heat_humidity"
    elif low_temp:
        risk_key = "cold_surface"
    else:
        risk_key = "normal"

    primary_mean = elevated[0]["mean"] if has_pressure else top_zone["mean"]
    primary_max = elevated[0]["max"] if has_pressure else top_zone["max"]

    region_str, risk_label, cause, recommendation = _build_clinical_texts(
        risk_key=risk_key,
        elevated_keys=elevated_keys,
        top_zone_key=top_zone["key"],
        primary_mean=primary_mean,
        primary_max=primary_max,
        temp=reading.temperature,
        hum=reading.humidity,
        high_temp=high_temp,
        high_hum=high_hum,
        lang=lang,
        is_transient=is_transient,
        transient_seconds=transient_seconds,
        alert_mode=False,
    )

    return {
        "risk_key": risk_key,
        "risk_type": risk_label,
        "region": region_str,
        "elevated_keys": elevated_keys,
        "affected_regions": [ZONE_NAMES[k][lang] for k in elevated_keys],
        "highlight_shapes": highlight_shapes,
        "top_zone_key": top_zone["key"],
        "top_zone_name": ZONE_NAMES[top_zone["key"]][lang],
        "top_zone_mean": round(top_zone["mean"], 1),
        "top_zone_max": round(top_zone["max"], 1),
        "primary_mean": round(primary_mean, 1),
        "primary_max": round(primary_max, 1),
        "temperature": reading.temperature,
        "humidity": reading.humidity,
        "high_temp": high_temp,
        "high_hum": high_hum,
        "is_transient": is_transient,
        "transient_seconds": transient_seconds,
        "cause": cause,
        "recommendation": recommendation,
        "is_compound": bool(has_pressure and (high_temp or high_hum) and not is_transient),
    }


def localize_analysis(analysis: dict, lang: LangType = "ar") -> dict:
    """يعيد ترجمة قاموس التحليل المخزّن في حالة الجلسة إلى اللغة المطلوبة فوراً."""
    region_str, risk_label, cause, recommendation = _build_clinical_texts(
        risk_key=analysis["risk_key"],
        elevated_keys=analysis["elevated_keys"],
        top_zone_key=analysis["top_zone_key"],
        primary_mean=analysis["primary_mean"],
        primary_max=analysis["primary_max"],
        temp=analysis["temperature"],
        hum=analysis["humidity"],
        high_temp=analysis["high_temp"],
        high_hum=analysis["high_hum"],
        lang=lang,
        is_transient=bool(analysis.get("is_transient", False)),
        transient_seconds=int(analysis.get("transient_seconds", 0)),
        alert_mode=False,
    )
    out = dict(analysis)
    out["risk_type"] = risk_label
    out["region"] = region_str
    out["affected_regions"] = [ZONE_NAMES[k][lang] for k in analysis["elevated_keys"]]
    out["top_zone_name"] = ZONE_NAMES[analysis["top_zone_key"]][lang]
    out["cause"] = cause
    out["recommendation"] = recommendation
    return out


# ───────── نموذج كشف الشذوذ وسجل التنبيهات ─────────

@dataclass
class AlertRecord:
    """سجل تنبيه واحد يدعم الترجمة الديناميكية عبر المفاتيح الداخلية."""
    timestamp: str
    message: str
    severity: str                                    # النص الافتراضي بالعربية
    severity_key: str = "warn"                       # "warn" | "danger" | "critical"
    risk_key: str = "normal"
    elevated_keys: list[str] = field(default_factory=list)
    top_zone_key: str = "pelvis"
    primary_mean: float = 0.0
    primary_max: float = 0.0
    high_temp: bool = False
    high_hum: bool = False
    sustained_minutes: int = DEFAULT_SUSTAINED_MINUTES
    sustained_periods: int = DEFAULT_SUSTAINED_PERIODS
    simulated_minutes: int = 0
    region: str = "—"
    cause: str = ""
    recommendation: str = ""
    details: dict = field(default_factory=dict)


def localize_alert(alert: AlertRecord, lang: LangType = "ar") -> dict:
    """يُرجع بيانات التنبيه مترجمة بالكامل إلى اللغة المختارة دون تعديل السجل الأصلي."""
    d = alert.details
    temp = float(d.get("الحرارة", 34.0))
    hum = float(d.get("الرطوبة", 55.0))
    sustained_mins = int(alert.sustained_minutes or d.get("دقائق_الاستمرار", DEFAULT_SUSTAINED_MINUTES))
    region_str, _, cause, recommendation = _build_clinical_texts(
        risk_key=alert.risk_key,
        elevated_keys=alert.elevated_keys,
        top_zone_key=alert.top_zone_key,
        primary_mean=alert.primary_mean,
        primary_max=alert.primary_max,
        temp=temp,
        hum=hum,
        high_temp=alert.high_temp,
        high_hum=alert.high_hum,
        lang=lang,
        sustained_minutes=sustained_mins,
        alert_mode=True,
    )
    if alert.simulated_minutes > 0:
        hrs, rem = divmod(alert.simulated_minutes, 60)
        time_display = f"+{hrs:02d}:{rem:02d} ({alert.timestamp})"
    else:
        time_display = alert.timestamp

    return {
        "severity_key": alert.severity_key,
        "severity": SEVERITY_LABELS.get(alert.severity_key, SEVERITY_LABELS["warn"])[lang],
        "region": region_str,
        "cause": cause,
        "recommendation": recommendation,
        "timestamp": time_display,
        "sustained_minutes": sustained_mins,
        "sustained_duration_label": format_sustained_duration(sustained_mins, lang),
        "consecutive": alert.sustained_periods,
        "peak_pressure": d.get("أقصى_ضغط", "—"),
        "temperature": d.get("الحرارة", "—"),
        "humidity": d.get("الرطوبة", "—"),
        "score": d.get("درجة_الشذوذ", "—"),
    }


class SmartMattressModel:
    """
    يغلّف Isolation Forest المتدرّب على 600 قراءة طبيعية مع:
      • فلترة الحركات العابرة اللحظية (ثوانٍ/دقائق) لمنع الإنذارات الكاذبة.
      • تتبّع مدة استمرار الضغط أو الشذوذ على نفس المنطقة بفترات مدة كل منها 15 دقيقة،
        وإطلاق التنبيه التمريضي فور بلوغ 4 فترات متواصلة (ساعة متواصلة = 60 دقيقة).
    """

    def __init__(
        self,
        sustained_minutes_threshold: int = DEFAULT_SUSTAINED_MINUTES,
        contamination: float = 0.03,
        consecutive_threshold: int | None = None,
    ):
        if consecutive_threshold is not None:
            sustained_minutes_threshold = consecutive_threshold * STEP_MINUTES
        self.sustained_minutes_threshold: int = sustained_minutes_threshold
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        self._sustained_minutes: int = 0
        self._sustained_periods: int = 0
        self._active_target_key: str | None = None
        self._simulated_elapsed_minutes: int = 0
        self.alerts: list[AlertRecord] = []
        self._is_trained: bool = False

    @property
    def sustained_periods_threshold(self) -> int:
        return max(1, self.sustained_minutes_threshold // STEP_MINUTES)

    @property
    def consecutive_threshold(self) -> int:
        return self.sustained_periods_threshold

    def train(self, n_normal: int = 600) -> None:
        """يتدرب على 600 قراءة من السيناريو الطبيعي فقط."""
        normal_data = generate_batch("normal", n=n_normal)
        X = extract_features_batch(normal_data)
        self.model.fit(X)
        self._is_trained = True

    def predict(self, reading: SensorReading) -> dict:
        """
        يفحص قراءة واحدة:
          • إذا كانت حركة عابرة لحظية (is_transient أو duration_minutes < 15): يتجاهلها كحركة طبيعية
            ولا يزيد عدّاد الوقت المستمر ولا يُطلق إنذاراً.
          • إذا كانت فترة مراقبة مستمرة (15 دقيقة): يتحقق من استمرار الشذوذ على نفس المنطقة،
            ويُطلق التنبيه التمريضي فور بلوغ 60 دقيقة متواصلة (4 فترات = ساعة متواصلة).
        """
        if not self._is_trained:
            self.train()

        features = extract_features(reading).reshape(1, -1)
        label = self.model.predict(features)[0]
        score = float(self.model.decision_function(features)[0])
        raw_is_anomaly = bool(label == -1)
        analysis = analyze_reading(reading, lang="ar")

        dur_mins = int(getattr(reading, "duration_minutes", STEP_MINUTES))
        is_transient_spike = bool(getattr(reading, "is_transient", False) or dur_mins < STEP_MINUTES)
        transient_secs = int(getattr(reading, "transient_seconds", 0))

        if is_transient_spike:
            # حركة طبيعية عابرة (ثوانٍ/دقائق): لا تزيد عدّاد الوقت المستمر ولا تُطلق إنذاراً
            if getattr(reading, "scenario", "") == "transient_motion":
                self._sustained_minutes = 0
                self._sustained_periods = 0
                self._active_target_key = None
            is_anomaly = False
            transient_ignored = True
            alert_triggered = False
        else:
            transient_ignored = False
            self._simulated_elapsed_minutes += dur_mins
            is_anomaly = raw_is_anomaly

            if is_anomaly:
                if analysis["elevated_keys"]:
                    target_key = f"zone:{analysis['elevated_keys'][0]}"
                else:
                    rk = analysis["risk_key"] if analysis["risk_key"] != "normal" else "general_anomaly"
                    target_key = f"env:{rk}"

                if self._active_target_key == target_key:
                    self._sustained_minutes += dur_mins
                    self._sustained_periods += 1
                else:
                    # تغيّرت المنطقة أو بدأ شذوذ جديد: يبدأ العدّ من أول فترة (15 دقيقة) لهذه المنطقة
                    self._active_target_key = target_key
                    self._sustained_minutes = dur_mins
                    self._sustained_periods = 1
            else:
                self._sustained_minutes = 0
                self._sustained_periods = 0
                self._active_target_key = None

            alert_triggered = self._sustained_minutes >= self.sustained_minutes_threshold

        alert = None
        if alert_triggered:
            eff_risk_key = analysis["risk_key"]
            if eff_risk_key == "normal":
                eff_risk_key = "general_anomaly"

            if analysis["is_compound"]:
                sev_key = "critical"
            elif analysis["elevated_keys"] or self._sustained_minutes >= self.sustained_minutes_threshold + 30:
                sev_key = "danger"
            else:
                sev_key = "warn"

            reg_ar, _, cause_ar, rec_ar = _build_clinical_texts(
                risk_key=eff_risk_key,
                elevated_keys=analysis["elevated_keys"],
                top_zone_key=analysis["top_zone_key"],
                primary_mean=analysis["primary_mean"],
                primary_max=analysis["primary_max"],
                temp=reading.temperature,
                hum=reading.humidity,
                high_temp=analysis["high_temp"],
                high_hum=analysis["high_hum"],
                lang="ar",
                sustained_minutes=self._sustained_minutes,
                alert_mode=True,
            )
            sev_ar = SEVERITY_LABELS[sev_key]["ar"]

            alert = AlertRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                message=cause_ar,
                severity=sev_ar,
                severity_key=sev_key,
                risk_key=eff_risk_key,
                elevated_keys=list(analysis["elevated_keys"]),
                top_zone_key=analysis["top_zone_key"],
                primary_mean=analysis["primary_mean"],
                primary_max=analysis["primary_max"],
                high_temp=analysis["high_temp"],
                high_hum=analysis["high_hum"],
                sustained_minutes=self._sustained_minutes,
                sustained_periods=self._sustained_periods,
                simulated_minutes=self._simulated_elapsed_minutes,
                region=reg_ar,
                cause=cause_ar,
                recommendation=rec_ar,
                details={
                    "المنطقة": reg_ar,
                    "أقصى_ضغط": round(float(reading.pressure_matrix.max()), 1),
                    "متوسط_أعلى_منطقة": analysis["top_zone_mean"],
                    "الحرارة": reading.temperature,
                    "الرطوبة": reading.humidity,
                    "درجة_الشذوذ": round(score, 4),
                    "دقائق_الاستمرار": self._sustained_minutes,
                    "فترات_متتالية": self._sustained_periods,
                },
            )
            self.alerts.append(alert)

        return {
            "is_anomaly": is_anomaly,
            "raw_is_anomaly": raw_is_anomaly,
            "transient_ignored": transient_ignored,
            "transient_seconds": transient_secs,
            "sustained_minutes": self._sustained_minutes,
            "sustained_periods": self._sustained_periods,
            "consecutive": self._sustained_periods,
            "simulated_elapsed_minutes": self._simulated_elapsed_minutes,
            "alert_triggered": alert_triggered,
            "alert": alert,
            "score": round(score, 4),
            "analysis": analysis,
        }

    def reset_streak(self) -> None:
        self._sustained_minutes = 0
        self._sustained_periods = 0
        self._active_target_key = None

    def reset(self) -> None:
        self._sustained_minutes = 0
        self._sustained_periods = 0
        self._active_target_key = None
        self._simulated_elapsed_minutes = 0
        self.alerts.clear()

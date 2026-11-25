"""
model.py – نموذج كشف الشذوذ والتشخيص الذكي للمناطق والتوصيات (ثنائي اللغة ar / en)

يستخدم Isolation Forest من scikit-learn:
  • يتدرب على بيانات السيناريو الطبيعي فقط.
  • يكشف الشذوذ في توزيع الضغط (في أي منطقة من الجسم) أو الحرارة أو الرطوبة.
  • يحدد تلقائيًا المنطقة المتأثرة بالضغط وسبب التنبيه وتوصية الإجراء للممرض بمفاتيح داخلية ثابتة
    قابلة للعرض الفوري بالعربية أو الإنجليزية دون المساس بحالة الجلسة.
  • لا يُطلق تنبيهًا إلا إذا استمر الشذوذ لعدة قراءات متتالية لتجنب التنبيهات الكاذبة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

from simulator import LangType, SensorReading, generate_batch

# ───────── عتبات تفسير المؤشرات السطحية (للتشخيص الداعم للقرار) ─────────

ZONE_PRESSURE_THRESHOLD = 45.0   # mmHg
HIGH_TEMP_THRESHOLD = 36.0       # °C
LOW_TEMP_THRESHOLD = 32.2        # °C
HIGH_HUMIDITY_THRESHOLD = 68.0   # %

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
        "ar": "تغيير وضعية المريض (إمالة جانبية بزاوية ٣٠°) لتخفيف الضغط المستمر عن منطقة الحوض والعجز.",
        "en": "Reposition the patient (30° lateral tilt) to relieve sustained pressure off the pelvis and sacrum.",
    },
    "heels_legs": {
        "ar": "رفع الكعبين عن سطح المفرش بوضع وسادة داعمة أسفل الساقين وتخفيف الحمل عن القدمين.",
        "en": "Offload the heels by placing a supportive pillow beneath the calves to relieve lower-limb pressure.",
    },
    "right_lateral": {
        "ar": "تغيير وضعية المريض من الاستلقاء الجانبي إلى وضعية متوازنة وتحرير الكتف والذراع واليد من الضغط.",
        "en": "Shift the patient from prolonged side-lying to a balanced posture and relieve the right shoulder, arm, and hand.",
    },
    "left_lateral": {
        "ar": "تغيير وضعية المريض من الاستلقاء الجانبي إلى وضعية متوازنة وتحرير الكتف والذراع واليد من الضغط.",
        "en": "Shift the patient from prolonged side-lying to a balanced posture and relieve the left shoulder, arm, and hand.",
    },
    "shoulders": {
        "ar": "تعديل زاوية الجزء العلوي للجسم وإعادة توزيع الارتكاز عن منطقة الكتفين.",
        "en": "Adjust upper-body elevation angle and redistribute weight away from the shoulders.",
    },
    "back": {
        "ar": "إمالة المريض جزئيًا بوسائد ساندة لتخفيف الضغط المتواصل عن منطقة الظهر.",
        "en": "Apply gentle lateral support pillows to relieve continuous pressure along the back.",
    },
}


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
        return f"Reposition the patient and redistribute support points to relieve pressure on the {zname}."
    return f"تغيير وضعية المريض وإعادة توزيع نقاط الارتكاز لتخفيف الضغط عن منطقة {zname}."


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
) -> tuple[str, str, str, str]:
    """يُرجع (region_str, risk_label, cause, recommendation) باللغة المطلوبة."""
    sep = "، " if lang == "ar" else ", "
    affected_labels = [ZONE_NAMES[k][lang] for k in elevated_keys]
    risk_label = RISK_LABELS[risk_key][lang]

    if risk_key == "compound":
        region_str = sep.join(affected_labels)
        if lang == "en":
            env_parts = []
            if high_temp:
                env_parts.append(f"elevated heat ({temp:.1f} °C)")
            if high_hum:
                env_parts.append(f"sweating & high moisture ({hum:.1f}%)")
            cause = (
                f"Compound skin-ulcer risk: sustained pressure on {region_str} "
                f"({primary_mean:.1f} mmHg) combined with {' and '.join(env_parts)}"
            )
            recommendation = (
                f"Immediate nursing action: reposition patient to offload {region_str}, "
                "activate safe cooling & ventilation, dry the skin, and inspect contact tissue."
            )
        else:
            env_parts = []
            if high_temp:
                env_parts.append(f"حرارة مرتفعة ({temp:.1f} °م)")
            if high_hum:
                env_parts.append(f"تعرق ورطوبة مرتفعة ({hum:.1f}%)")
            cause = (
                f"خطر مركّب لتقرحات الجلد: ضغط مستمر على {region_str} "
                f"({primary_mean:.1f} ملم زئبق) متزامن مع {' و'.join(env_parts)}"
            )
            recommendation = (
                f"تدخّل تمريضي فوري: تغيير وضعية المريض لتخفيف الضغط عن {region_str}، "
                "وتفعيل التهوية والتبريد الآمن مع تجفيف الجلد وفحص موضع التلامس."
            )
    elif risk_key == "localized_pressure":
        region_str = sep.join(affected_labels)
        primary_key = elevated_keys[0]
        if lang == "en":
            cause = (
                f"Sustained elevated pressure on {region_str} "
                f"(mean {primary_mean:.1f} mmHg, peak {primary_max:.1f} mmHg)"
            )
        else:
            cause = (
                f"ارتفاع مستمر للضغط على منطقة {region_str} "
                f"(متوسط {primary_mean:.1f} ملم زئبق، الأقصى {primary_max:.1f} ملم زئبق)"
            )
        recommendation = _pressure_recommendation(primary_key, lang)
    elif risk_key == "heat_humidity":
        region_str = ZONE_NAMES["full_surface"][lang]
        if lang == "en":
            if high_temp and high_hum:
                cause = f"Sweating with elevated surface moisture ({hum:.1f}%) and temperature ({temp:.1f} °C)"
            elif high_hum:
                cause = f"Elevated mattress surface moisture ({hum:.1f}%)"
            else:
                cause = f"Elevated mattress surface temperature ({temp:.1f} °C)"
            recommendation = (
                "Activate safe mattress ventilation & cooling, and verify skin and linen dryness to prevent maceration."
            )
        else:
            if high_temp and high_hum:
                cause = f"تعرق وارتفاع ملحوظ في رطوبة السطح ({hum:.1f}%) وحرارته ({temp:.1f} °م)"
            elif high_hum:
                cause = f"ارتفاع ملحوظ في رطوبة سطح المفرش ({hum:.1f}%)"
            else:
                cause = f"ارتفاع ملحوظ في حرارة سطح المفرش ({temp:.1f} °م)"
            recommendation = "تفعيل التهوية والتبريد الآمن للمفرش، والتحقق من جفاف الجلد والأغطية للوقاية من تهيّج البشرة."
    elif risk_key == "cold_surface":
        region_str = ZONE_NAMES["full_surface"][lang]
        if lang == "en":
            cause = f"Unusual coolness and drop in mattress surface temperature ({temp:.1f} °C)"
            recommendation = "Activate safe mattress warming and check the patient's blankets and core thermal comfort."
        else:
            cause = f"برودة وانخفاض غير معتاد في حرارة سطح المفرش ({temp:.1f} °م)"
            recommendation = "تفعيل التدفئة الآمنة للمفرش وتفقّد أغطية المريض ودرجة حرارته الحيوية."
    elif risk_key == "general_anomaly":
        top_name = ZONE_NAMES[top_zone_key][lang]
        if lang == "en":
            region_str = f"Peak Zone: {top_name}"
            cause = "Repeated pattern deviation from the patient's learned normal baseline"
            recommendation = "Check patient posture and review surface pressure, temperature, and humidity indicators."
        else:
            region_str = f"أعلى ضغط: {top_name}"
            cause = "انحراف متكرر في نمط القراءات عن خط الأساس الطبيعي للمريض"
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
        "cause": cause,
        "recommendation": recommendation,
        "is_compound": bool(has_pressure and (high_temp or high_hum)),
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
    severity: str                    # النص الافتراضي بالعربية
    severity_key: str = "warn"       # "warn" | "danger" | "critical"
    risk_key: str = "normal"
    elevated_keys: list[str] = field(default_factory=list)
    top_zone_key: str = "pelvis"
    primary_mean: float = 0.0
    primary_max: float = 0.0
    high_temp: bool = False
    high_hum: bool = False
    region: str = "—"
    cause: str = ""
    recommendation: str = ""
    details: dict = field(default_factory=dict)


def localize_alert(alert: AlertRecord, lang: LangType = "ar") -> dict:
    """يُرجع بيانات التنبيه مترجمة بالكامل إلى اللغة المختارة دون تعديل السجل الأصلي."""
    d = alert.details
    temp = float(d.get("الحرارة", 34.0))
    hum = float(d.get("الرطوبة", 55.0))
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
    )
    return {
        "severity_key": alert.severity_key,
        "severity": SEVERITY_LABELS.get(alert.severity_key, SEVERITY_LABELS["warn"])[lang],
        "region": region_str,
        "cause": cause,
        "recommendation": recommendation,
        "timestamp": alert.timestamp,
        "consecutive": d.get("قراءات_متتالية", "—"),
        "peak_pressure": d.get("أقصى_ضغط", "—"),
        "temperature": d.get("الحرارة", "—"),
        "humidity": d.get("الرطوبة", "—"),
        "score": d.get("درجة_الشذوذ", "—"),
    }


class SmartMattressModel:
    """
    يغلّف Isolation Forest المتدرّب على النمط الطبيعي فقط مع منطق التنبيه المتتالي.
    """

    def __init__(self, consecutive_threshold: int = 3,
                 contamination: float = 0.03):
        self.consecutive_threshold = consecutive_threshold
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        self._consecutive_anomalies: int = 0
        self.alerts: list[AlertRecord] = []
        self._is_trained: bool = False

    def train(self, n_normal: int = 600) -> None:
        """يتدرب على بيانات السيناريو الطبيعي فقط."""
        normal_data = generate_batch("normal", n=n_normal)
        X = extract_features_batch(normal_data)
        self.model.fit(X)
        self._is_trained = True

    def predict(self, reading: SensorReading) -> dict:
        """
        يفحص قراءة واحدة، ويحدّث عدّاد الشذوذ المتتالي، ولا يُطلق التنبيه
        إلا عند استمرار الشذوذ لـ consecutive_threshold قراءات متتالية.
        """
        if not self._is_trained:
            self.train()

        features = extract_features(reading).reshape(1, -1)
        label = self.model.predict(features)[0]
        score = float(self.model.decision_function(features)[0])
        is_anomaly = bool(label == -1)
        analysis = analyze_reading(reading, lang="ar")

        if is_anomaly:
            self._consecutive_anomalies += 1
        else:
            self._consecutive_anomalies = 0

        alert_triggered = self._consecutive_anomalies >= self.consecutive_threshold

        alert = None
        if alert_triggered:
            eff_risk_key = analysis["risk_key"]
            if eff_risk_key == "normal":
                eff_risk_key = "general_anomaly"

            if analysis["is_compound"]:
                sev_key = "critical"
            elif self._consecutive_anomalies >= self.consecutive_threshold + 2:
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
            )
            sev_ar = SEVERITY_LABELS[sev_key]["ar"]

            alert = AlertRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                message=f"{cause_ar} (مستمر منذ {self._consecutive_anomalies} قراءات متتالية)",
                severity=sev_ar,
                severity_key=sev_key,
                risk_key=eff_risk_key,
                elevated_keys=list(analysis["elevated_keys"]),
                top_zone_key=analysis["top_zone_key"],
                primary_mean=analysis["primary_mean"],
                primary_max=analysis["primary_max"],
                high_temp=analysis["high_temp"],
                high_hum=analysis["high_hum"],
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
                    "قراءات_متتالية": self._consecutive_anomalies,
                },
            )
            self.alerts.append(alert)

        return {
            "is_anomaly": is_anomaly,
            "consecutive": self._consecutive_anomalies,
            "alert_triggered": alert_triggered,
            "alert": alert,
            "score": round(score, 4),
            "analysis": analysis,
        }

    def reset_streak(self) -> None:
        self._consecutive_anomalies = 0

    def reset(self) -> None:
        self._consecutive_anomalies = 0
        self.alerts.clear()

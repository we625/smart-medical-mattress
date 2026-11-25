"""
model.py – نموذج كشف الشذوذ والتشخيص الذكي للمناطق والتوصيات (المفرش الطبي الذكي)

يستخدم Isolation Forest من scikit-learn:
  • يتدرب على بيانات السيناريو الطبيعي فقط.
  • يكشف الشذوذ في توزيع الضغط (في أي منطقة من الجسم) أو الحرارة أو الرطوبة.
  • يحدد تلقائيًا المنطقة المتأثرة بالضغط وسبب التنبيه وتوصية الإجراء للممرض.
  • لا يُطلق تنبيهًا إلا إذا استمر الشذوذ لعدة قراءات متتالية لتجنب التنبيهات الكاذبة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

from simulator import SensorReading, generate_batch

# ───────── عتبات تفسير المؤشرات السطحية (للتشخيص الداعم للقرار) ─────────

ZONE_PRESSURE_THRESHOLD = 45.0   # ملم زئبق (متوسط المنطقة لاعتبارها تحت ضغط مرتفع)
HIGH_TEMP_THRESHOLD = 36.0       # °م
LOW_TEMP_THRESHOLD = 32.2        # °م
HIGH_HUMIDITY_THRESHOLD = 68.0   # %


def _zone_slices(mat: np.ndarray) -> list[dict]:
    """
    يُرجع قائمة المناطق التشريحية مع قيم الضغط وإحداثيات التظليل على الخريطة الحرارية.
    """
    right_lateral = np.concatenate([mat[2:4, 0:3].ravel(), mat[4:9, 0:2].ravel()])
    left_lateral = np.concatenate([mat[2:4, 5:8].ravel(), mat[4:9, 6:8].ravel()])
    heels_legs = mat[13:16, 2:6]
    heels_only = mat[14:16, 2:6]

    return [
        {
            "key": "pelvis",
            "name": "الحوض",
            "mean": float(mat[6:10, 2:6].mean()),
            "max": float(mat[6:10, 2:6].max()),
            "shape": {
                "type": "rect",
                "x0": 1.55, "y0": 5.55, "x1": 5.45, "y1": 9.45,
            },
        },
        {
            "key": "heels_legs",
            "name": "الكعبين والساقين",
            # نأخذ الأعلى بين متوسط الكعبين ومتوسط الساقين+الكعبين لرصد تركز الكعبين بدقة
            "mean": float(max(heels_legs.mean(), heels_only.mean() * 0.95)),
            "max": float(heels_legs.max()),
            "shape": {
                "type": "rect",
                "x0": 1.55, "y0": 12.55, "x1": 5.45, "y1": 15.45,
            },
        },
        {
            "key": "right_lateral",
            "name": "الكتف والذراع واليد (الأيمن)",
            "mean": float(right_lateral.mean()),
            "max": float(right_lateral.max()),
            "shape": {
                "type": "path",
                "path": "M -0.45,1.55 L 2.45,1.55 L 2.45,3.45 L 1.45,3.45 L 1.45,8.45 L -0.45,8.45 Z",
            },
        },
        {
            "key": "left_lateral",
            "name": "الكتف والذراع واليد (الأيسر)",
            "mean": float(left_lateral.mean()),
            "max": float(left_lateral.max()),
            "shape": {
                "type": "path",
                "path": "M 5.55,1.55 L 7.45,1.55 L 7.45,8.45 L 5.55,8.45 L 5.55,3.45 L 4.55,3.45 L 4.55,1.55 Z",
            },
        },
        {
            "key": "shoulders",
            "name": "الكتفين",
            "mean": float(mat[2:4, 1:7].mean()),
            "max": float(mat[2:4, 1:7].max()),
            "shape": {
                "type": "rect",
                "x0": 0.55, "y0": 1.55, "x1": 6.45, "y1": 3.45,
            },
        },
        {
            "key": "back",
            "name": "الظهر",
            "mean": float(mat[4:6, 2:6].mean()),
            "max": float(mat[4:6, 2:6].max()),
            "shape": {
                "type": "rect",
                "x0": 1.55, "y0": 3.55, "x1": 5.45, "y1": 5.45,
            },
        },
        {
            "key": "thighs",
            "name": "الفخذين",
            "mean": float(mat[10:13, 2:6].mean()),
            "max": float(mat[10:13, 2:6].max()),
            "shape": {
                "type": "rect",
                "x0": 1.55, "y0": 9.55, "x1": 5.45, "y1": 12.45,
            },
        },
        {
            "key": "head",
            "name": "الرأس",
            "mean": float(mat[0:2, 2:6].mean()),
            "max": float(mat[0:2, 2:6].max()),
            "shape": {
                "type": "rect",
                "x0": 1.55, "y0": -0.45, "x1": 5.45, "y1": 1.45,
            },
        },
    ]


# ───────── استخراج السمات (Feature Engineering) ─────────

def extract_features(reading: SensorReading) -> np.ndarray:
    """
    يحوّل قراءة واحدة إلى متجه سمات متوازن يغطي كافة مناطق الضغط والحرارة والرطوبة.

    السمات (7 سمات متوازنة):
      0  أعلى متوسط ضغط بين المناطق التشريحية الثماني
      1  المئين 97 للضغط (يرصد البؤر الموضعية الدقيقة كالكعبين والكتف والذراع)
      2  أقصى ضغط مسجّل على المفرش
      3  درجة الحرارة السطحية
      4  الانحراف المطلق للحرارة عن خط الأساس الطبيعي (34.0 °م)
      5  الرطوبة النسبية
      6  الانحراف المطلق للرطوبة عن خط الأساس الطبيعي (55.0 %)
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
    """يُرجع مصفوفة السمات بحجم (n_samples, n_features)."""
    return np.vstack([extract_features(r) for r in readings])


# ───────── التحليل التشريحي والسريري للقراءة ─────────

def _pressure_recommendation_for_key(zone_key: str, zone_name: str) -> str:
    if zone_key == "pelvis":
        return "تغيير وضعية المريض (إمالة جانبية بزاوية ٣٠°) لتخفيف الضغط المستمر عن منطقة الحوض والعجز."
    if zone_key == "heels_legs":
        return "رفع الكعبين عن سطح المفرش بوضع وسادة داعمة أسفل الساقين وتخفيف الحمل عن القدمين."
    if zone_key in ("right_lateral", "left_lateral"):
        return "تغيير وضعية المريض من الاستلقاء الجانبي إلى وضعية متوازنة وتحرير الكتف والذراع واليد من الضغط."
    if zone_key == "shoulders":
        return "تعديل زاوية الجزء العلوي للجسم وإعادة توزيع الارتكاز عن منطقة الكتفين."
    if zone_key == "back":
        return "إمالة المريض جزئيًا بوسائد ساندة لتخفيف الضغط المتواصل عن منطقة الظهر."
    return f"تغيير وضعية المريض وإعادة توزيع نقاط الارتكاز لتخفيف الضغط عن منطقة {zone_name}."


def analyze_reading(reading: SensorReading) -> dict:
    """
    يحلّل مصفوفة الضغط والحرارة والرطوبة لتحديد المنطقة المتأثرة وسبب التنبيه وتوصية الممرض.
    """
    zones = _zone_slices(reading.pressure_matrix)
    zones_sorted = sorted(zones, key=lambda z: z["mean"], reverse=True)
    top_zone = zones_sorted[0]

    elevated = [z for z in zones_sorted if z["mean"] >= ZONE_PRESSURE_THRESHOLD]
    # إذا شُخّص ضغط جانبي على الكتف والذراع واليد، نتجنب تكرار منطقة "الكتفين" المتداخلة معه
    if any(z["key"] in ("right_lateral", "left_lateral") for z in elevated):
        elevated = [z for z in elevated if z["key"] != "shoulders"]

    high_temp = reading.temperature >= HIGH_TEMP_THRESHOLD
    low_temp = reading.temperature <= LOW_TEMP_THRESHOLD
    high_hum = reading.humidity >= HIGH_HUMIDITY_THRESHOLD

    has_pressure = len(elevated) > 0
    affected_names = [z["name"] for z in elevated]
    highlight_shapes = [z["shape"] for z in elevated]

    if has_pressure and (high_temp or high_hum):
        primary_zone = elevated[0]
        region_str = "، ".join(affected_names)
        risk_type = "خطر مركّب"
        env_parts = []
        if high_temp:
            env_parts.append(f"حرارة مرتفعة ({reading.temperature:.1f} °م)")
        if high_hum:
            env_parts.append(f"تعرق ورطوبة مرتفعة ({reading.humidity:.1f}%)")
        cause = (
            f"خطر مركّب لتقرحات الجلد: ضغط مستمر على {region_str} "
            f"({primary_zone['mean']:.1f} ملم زئبق) متزامن مع {' و'.join(env_parts)}"
        )
        recommendation = (
            f"تدخّل تمريضي فوري: تغيير وضعية المريض لتخفيف الضغط عن {region_str}، "
            "وتفعيل التهوية والتبريد الآمن مع تجفيف الجلد وفحص موضع التلامس."
        )
    elif has_pressure:
        primary_zone = elevated[0]
        region_str = "، ".join(affected_names)
        risk_type = "ضغط موضعي"
        cause = (
            f"ارتفاع مستمر للضغط على منطقة {region_str} "
            f"(متوسط {primary_zone['mean']:.1f} ملم زئبق، الأقصى {primary_zone['max']:.1f} ملم زئبق)"
        )
        recommendation = _pressure_recommendation_for_key(primary_zone["key"], primary_zone["name"])
    elif high_temp or high_hum:
        region_str = "سطح المفرش العام"
        risk_type = "حرارة ورطوبة"
        if high_temp and high_hum:
            cause = f"تعرق وارتفاع ملحوظ في رطوبة السطح ({reading.humidity:.1f}%) وحرارته ({reading.temperature:.1f} °م)"
        elif high_hum:
            cause = f"ارتفاع ملحوظ في رطوبة سطح المفرش ({reading.humidity:.1f}%)"
        else:
            cause = f"ارتفاع ملحوظ في حرارة سطح المفرش ({reading.temperature:.1f} °م)"
        recommendation = "تفعيل التهوية والتبريد الآمن للمفرش، والتحقق من جفاف الجلد والأغطية للوقاية من تهيّج البشرة."
    elif low_temp:
        region_str = "سطح المفرش العام"
        risk_type = "برودة سطحية"
        cause = f"برودة وانخفاض غير معتاد في حرارة سطح المفرش ({reading.temperature:.1f} °م)"
        recommendation = "تفعيل التدفئة الآمنة للمفرش وتفقّد أغطية المريض ودرجة حرارته الحيوية."
    else:
        region_str = "توزيع متوازن"
        risk_type = "طبيعي"
        cause = "القراءات ضمن النمط الطبيعي المتوازن للمريض"
        recommendation = "استمرار المراقبة الدورية المعتادة دون حاجة لتدخل إضافي."

    return {
        "risk_type": risk_type,
        "region": region_str,
        "affected_regions": affected_names,
        "highlight_shapes": highlight_shapes,
        "top_zone_name": top_zone["name"],
        "top_zone_mean": round(top_zone["mean"], 1),
        "top_zone_max": round(top_zone["max"], 1),
        "cause": cause,
        "recommendation": recommendation,
        "is_compound": bool(has_pressure and (high_temp or high_hum)),
    }


# ───────── نموذج كشف الشذوذ ─────────

@dataclass
class AlertRecord:
    """سجل تنبيه واحد مع المنطقة المتأثرة والسبب وتوصية الممرض."""
    timestamp: str
    message: str
    severity: str               # "تحذير" أو "خطر" أو "خطر حرج"
    region: str = "—"
    cause: str = ""
    recommendation: str = ""
    details: dict = field(default_factory=dict)


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
        normal_data = generate_batch("طبيعي", n=n_normal)
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
        label = self.model.predict(features)[0]       # 1 = طبيعي, -1 = شاذ
        score = float(self.model.decision_function(features)[0])
        is_anomaly = bool(label == -1)
        analysis = analyze_reading(reading)

        if is_anomaly:
            self._consecutive_anomalies += 1
        else:
            self._consecutive_anomalies = 0

        alert_triggered = self._consecutive_anomalies >= self.consecutive_threshold

        alert = None
        if alert_triggered:
            # في حال رصد النموذج شذوذًا دون تجاوز عتبة فرعية محددة، نضع سببًا وتوصية عامة واضحة
            cause = analysis["cause"]
            recommendation = analysis["recommendation"]
            region = analysis["region"]
            if analysis["risk_type"] == "طبيعي":
                region = f"أعلى ضغط: {analysis['top_zone_name']}"
                cause = "انحراف متكرر في نمط القراءات عن خط الأساس الطبيعي للمريض"
                recommendation = "تفقّد وضعية المريض ومراجعة مؤشرات الضغط والحرارة والرطوبة."

            if analysis["is_compound"]:
                severity = "خطر حرج"
            elif self._consecutive_anomalies >= self.consecutive_threshold + 2:
                severity = "خطر"
            else:
                severity = "تحذير"

            alert = AlertRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                message=f"{cause} (مستمر منذ {self._consecutive_anomalies} قراءات متتالية)",
                severity=severity,
                region=region,
                cause=cause,
                recommendation=recommendation,
                details={
                    "المنطقة": region,
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
        """يصفّر عدّاد القراءات الشاذة المتتالية فقط دون مسح السجل."""
        self._consecutive_anomalies = 0

    def reset(self) -> None:
        """يصفّر العدّاد وسجل التنبيهات."""
        self._consecutive_anomalies = 0
        self.alerts.clear()

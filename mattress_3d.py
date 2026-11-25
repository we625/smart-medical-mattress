"""
mattress_3d.py – مكوّن العرض التفاعلي ثلاثي الأبعاد (Three.js) للمفرش الطبي الذكي

يبني مشهد WebGL طبيًا واقعيًا وتفاعليًا يضم:
  • سرير مستشفى حديث أبيض مع المرتبة الأساسية
  • المفرش الطبي الذكي المكوّن من 5 طبقات هندسية قابلة للتفكيك السلس (Exploded View)
  • شريحة حساسات 16×8 مرتبطة لحظيًا بمصفوفة الضغط مع تموّج ثلاثي الأبعاد
  • تيارات وجسيمات هواء زرقاء عند التبريد/التهوية وتوهج دافئ عند التدفئة في الطبقة الثالثة
  • وحدة تحكم جانبية متصلة بسلك رفيع مع فتحة تهوية ومؤشر LED متغير اللون حسب الحالة
  • مجسم مريض ثلاثي الأبعاد بمظهر زجاجي أبيض مطفي شفاف (Frosted Glass) مع زر إظهار/إخفاء
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import streamlit as st

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


@st.cache_data(show_spinner=False)
def _load_vendor_scripts() -> tuple[str, str]:
    """يقرأ مكتبتي Three.js و OrbitControls محليًا لضمان سرعة واستقرار العرض."""
    three_js = (VENDOR_DIR / "three.min.js").read_text(encoding="utf-8")
    orbit_js = (VENDOR_DIR / "OrbitControls.js").read_text(encoding="utf-8")
    return three_js, orbit_js


VIEWER_3D_I18N: dict[str, dict[str, str]] = {
    "ar": {
        "dir": "rtl",
        "align": "right",
        "scale_deg": "90deg",
        "unit_prefix": "وحدة التحكم: ",
        "led_alert": "تنبيه نشط (أحمر)",
        "led_cool_alert": "تهوية وتبريد آمن + تنبيه",
        "led_cool": "تهوية وتبريد آمن (أزرق)",
        "led_heat_alert": "تدفئة آمنة + تنبيه",
        "led_heat": "تدفئة آمنة (برتقالي دافئ)",
        "led_normal": "وضع مستقر (أخضر)",
        "btn_explode_open": "عرض الطبقات الداخلية",
        "btn_explode_close": "دمج الطبقات",
        "slider_label": "تفكيك الطبقات",
        "btn_hide_patient": "إخفاء مجسم المريض",
        "btn_show_patient": "إظهار مجسم المريض",
        "btn_reset_cam": "إعادة ضبط الزاوية",
        "hint": "اسحب بالماوس للتدوير · عجلة الماوس للتقريب · حرّك شريط الطبقات لفحص البنية الهندسية",
        "low_pressure": "ضغط منخفض",
        "high_pressure": "ضغط مرتفع",
        "l1_title": "١) الغطاء القطني الطبي المبطّن",
        "l1_sub": "مربعات ناعمة لطيفة على الجلد وقابلة للغسل",
        "l2_title": "٢) شريحة الحساسات المرنة (١٦×٨)",
        "l2_sub": "١٢٨ نقطة قياس لحظية للضغط والتموّج الحراري",
        "l3_title": "٣) شبكة التهوية (3D Spacer Mesh)",
        "l3_sub": "قنوات التبريد والتدفئة الآمنة وتصريف الرطوبة",
        "l4_title": "٤) الإسفنج الطبي المثقّب",
        "l4_sub": "توزيع متوازن للضغط ومسارات تهوية",
        "l5_title": "٥) القاعدة المانعة للانزلاق",
        "l5_sub": "ثبات على المرتبة + ٤ أحزمة تثبيت مطاطية",
    },
    "en": {
        "dir": "ltr",
        "align": "left",
        "scale_deg": "270deg",
        "unit_prefix": "Control Unit: ",
        "led_alert": "Active Alert (Red)",
        "led_cool_alert": "Safe Cooling + Alert",
        "led_cool": "Safe Cooling & Vent (Blue)",
        "led_heat_alert": "Safe Warming + Alert",
        "led_heat": "Safe Warming (Warm Orange)",
        "led_normal": "Stable Mode (Green)",
        "btn_explode_open": "Explode Internal Layers",
        "btn_explode_close": "Merge Layers",
        "slider_label": "Layer Separation",
        "btn_hide_patient": "Hide Patient Model",
        "btn_show_patient": "Show Patient Model",
        "btn_reset_cam": "Reset View",
        "hint": "Drag to rotate · Scroll to zoom · Adjust slider to inspect the 5 internal layers",
        "low_pressure": "Low Pressure",
        "high_pressure": "High Pressure",
        "l1_title": "1) Quilted Medical Cotton Cover",
        "l1_sub": "Soft skin-friendly squares · Removable & washable",
        "l2_title": "2) Flexible Sensor Array (16×8)",
        "l2_sub": "128 real-time pressure nodes & 3D relief",
        "l3_title": "3) 3D Spacer Ventilation Mesh",
        "l3_sub": "Safe cooling/warming channels & moisture airflow",
        "l4_title": "4) Perforated Medical Foam",
        "l4_sub": "Balanced pressure redistribution & air pores",
        "l5_title": "5) Anti-Slip Base Layer",
        "l5_sub": "Stable bed grip + 4 elastic corner straps",
    },
}


def build_mattress_3d_html(
    pressure_matrix: np.ndarray | None,
    climate_visual: str = "standby",   # "standby" | "cool" | "heat" | "off"
    alert_triggered: bool = False,
    affected_region: str = "توزيع متوازن",
    temperature: float = 34.0,
    humidity: float = 55.0,
    lang: str = "ar",
) -> str:
    """
    يولّد كود HTML متكاملاً لمشهد Three.js التفاعلي باللغة المطلوبة (ar / en).
    """
    three_js, orbit_js = _load_vendor_scripts()
    tr = VIEWER_3D_I18N.get(lang, VIEWER_3D_I18N["ar"])

    if pressure_matrix is None:
        mat_list = [[10.0] * 8 for _ in range(16)]
    else:
        mat_list = np.round(pressure_matrix, 1).tolist()

    if alert_triggered and climate_visual not in ("cool", "heat"):
        led_state = "alert"
        led_label = tr["led_alert"]
        led_hex = "#EF4444"
    elif climate_visual == "cool":
        led_state = "cool_alert" if alert_triggered else "cool"
        led_label = tr["led_cool_alert"] if alert_triggered else tr["led_cool"]
        led_hex = "#0EA5E9"
    elif climate_visual == "heat":
        led_state = "heat_alert" if alert_triggered else "heat"
        led_label = tr["led_heat_alert"] if alert_triggered else tr["led_heat"]
        led_hex = "#F97316"
    elif alert_triggered:
        led_state = "alert"
        led_label = tr["led_alert"]
        led_hex = "#EF4444"
    else:
        led_state = "normal"
        led_label = tr["led_normal"]
        led_hex = "#10B981"

    payload = json.dumps(
        {
            "matrix": mat_list,
            "climate": climate_visual,
            "alert": bool(alert_triggered),
            "ledState": led_state,
            "ledLabel": led_label,
            "ledHex": led_hex,
            "region": affected_region,
            "temp": round(float(temperature), 1),
            "hum": round(float(humidity), 1),
            "i18n": tr,
        },
        ensure_ascii=False,
    )

    return f"""<!DOCTYPE html>
<html lang="{lang}" dir="{tr['dir']}">
<head>
<meta charset="utf-8" />
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@500;600;700&family=Inter:wght@500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
  html, body {{
    width: 100%; height: 100%; overflow: hidden;
    background: #F8FAFC;
    font-family: 'Cairo', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #0F172A;
    direction: {tr['dir']};
    text-align: {tr['align']};
  }}
  #stage {{
    position: relative;
    width: 100%;
    height: 540px;
    border-radius: 14px;
    overflow: hidden;
    background: radial-gradient(circle at 50% 38%, #FFFFFF 0%, #F8FAFC 65%, #EEF2F6 100%);
    border: 1px solid #E2E8F0;
  }}
  canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}

  /* شريط الأدوات التفاعلي داخل مشهد الـ 3D */
  .hud-top {{
    position: absolute; top: 12px; right: 12px; left: 12px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; flex-wrap: wrap; pointer-events: none; z-index: 10;
  }}
  .hud-group {{
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    pointer-events: auto;
  }}
  .hud-btn {{
    appearance: none; border: 1px solid #E2E8F0;
    background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(8px);
    color: #0F172A; font-family: inherit; font-size: 12.5px; font-weight: 700;
    padding: 7px 13px; border-radius: 10px; cursor: pointer;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    transition: all 0.16s ease;
    display: inline-flex; align-items: center; gap: 6px;
  }}
  .hud-btn:hover {{ border-color: #0284C7; color: #0284C7; background: #FFFFFF; }}
  .hud-btn.active {{
    background: #0284C7; border-color: #0284C7; color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
  }}
  .slider-pill {{
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(8px);
    border: 1px solid #E2E8F0; border-radius: 10px; padding: 5px 12px;
    font-size: 12px; font-weight: 600; color: #475569;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
  }}
  .slider-pill input[type="range"] {{
    width: 88px; accent-color: #0284C7; cursor: pointer; direction: ltr;
  }}

  /* شارة حالة وحدة التحكم الجانبية والبيانات اللحظية */
  .unit-status {{
    pointer-events: auto;
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(8px);
    border: 1px solid #E2E8F0; border-radius: 999px;
    padding: 6px 13px; font-size: 12px; font-weight: 700; color: #0F172A;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
  }}
  .led-dot {{
    width: 9px; height: 9px; border-radius: 50%; display: inline-block;
    box-shadow: 0 0 8px currentColor;
  }}

  /* شريط الإرشاد السفلي ومفتاح الألوان */
  .hud-bottom {{
    position: absolute; bottom: 10px; right: 12px; left: 12px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; flex-wrap: wrap; pointer-events: none; z-index: 10;
  }}
  .hint-pill {{
    background: rgba(255, 255, 255, 0.88); backdrop-filter: blur(6px);
    border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 4px 10px; font-size: 11.5px; color: #64748B; font-weight: 600;
  }}
  .scale-bar {{
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(6px);
    border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 4px 10px; font-size: 11.5px; color: #475569; font-weight: 600;
  }}
  .scale-grad {{
    width: 84px; height: 8px; border-radius: 999px;
    background: linear-gradient({tr['scale_deg']}, #991B1B 0%, #DC2626 18%, #F97316 38%, #FACC15 58%, #7DD3FC 82%, #E0F2FE 100%);
  }}

  /* تسميات الطبقات الخمس عند التفكيك */
  #labels-layer {{
    position: absolute; inset: 0; pointer-events: none; z-index: 8; overflow: hidden;
  }}
  .layer-tag {{
    position: absolute; transform: translate(-100%, -50%);
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #CBD5E1; border-inline-start: 3px solid #0284C7;
    border-radius: 8px; padding: 4px 10px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    white-space: nowrap; opacity: 0; transition: opacity 0.2s ease;
    direction: {tr['dir']}; text-align: {tr['align']};
  }}
  .layer-tag .lt-title {{ font-size: 11.5px; font-weight: 700; color: #0F172A; display: block; line-height: 1.3; }}
  .layer-tag .lt-sub {{ font-size: 10.5px; font-weight: 500; color: #64748B; display: block; line-height: 1.2; }}
</style>
</head>
<body>
<div id="stage">
  <div class="hud-top">
    <div class="hud-group">
      <button id="btnExplode" class="hud-btn" type="button">{tr['btn_explode_open']}</button>
      <div class="slider-pill">
        <span>{tr['slider_label']}</span>
        <input id="rngExplode" type="range" min="0" max="100" value="0" />
      </div>
      <button id="btnPatient" class="hud-btn active" type="button">{tr['btn_hide_patient']}</button>
      <button id="btnResetCam" class="hud-btn" type="button">{tr['btn_reset_cam']}</button>
    </div>
    <div class="unit-status" id="unitBadge">
      <span class="led-dot" id="ledDot"></span>
      <span id="ledText"></span>
    </div>
  </div>

  <div id="labels-layer"></div>

  <div class="hud-bottom">
    <div class="hint-pill">{tr['hint']}</div>
    <div class="scale-bar">
      <span>{tr['low_pressure']}</span>
      <span class="scale-grad"></span>
      <span>{tr['high_pressure']}</span>
    </div>
  </div>
</div>

<script>{three_js}</script>
<script>{orbit_js}</script>
<script>
(function() {{
  const DATA = {payload};
  const TR = DATA.i18n;
  const stage = document.getElementById('stage');
  const labelsLayer = document.getElementById('labels-layer');
  const ledDot = document.getElementById('ledDot');
  const ledText = document.getElementById('ledText');

  ledDot.style.color = DATA.ledHex;
  ledDot.style.background = DATA.ledHex;
  ledText.textContent = TR.unit_prefix + DATA.ledLabel;

  // ─────────── إعداد المشهد والكاميرا والإضاءة ───────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xF8FAFC);

  const camera = new THREE.PerspectiveCamera(36, stage.clientWidth / stage.clientHeight, 0.1, 40);
  const DEFAULT_CAM = new THREE.Vector3(-2.38, 1.76, 2.48);
  const DEFAULT_TARGET = new THREE.Vector3(0.05, 0.56, 0.0);
  camera.position.copy(DEFAULT_CAM);

  const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.06;
  stage.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.065;
  controls.target.copy(DEFAULT_TARGET);
  controls.minDistance = 1.35;
  controls.maxDistance = 5.2;
  controls.maxPolarAngle = Math.PI * 0.485;

  // إضاءة استوديو طبية ناعمة
  const hemiLight = new THREE.HemisphereLight(0xFFFFFF, 0xE2E8F0, 0.88);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xFFFFFF, 0.72);
  keyLight.position.set(-2.4, 3.8, 2.2);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 10;
  keyLight.shadow.camera.left = -2.2;
  keyLight.shadow.camera.right = 2.2;
  keyLight.shadow.camera.top = 2.2;
  keyLight.shadow.camera.bottom = -2.2;
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xE0F2FE, 0.42);
  fillLight.position.set(2.5, 2.0, -1.8);
  scene.add(fillLight);

  // أرضية استوديو لاستقبال الظلال الناعمة
  const floorGeo = new THREE.CircleGeometry(3.2, 64);
  const floorMat = new THREE.ShadowMaterial({{ opacity: 0.08 }});
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.001;
  floor.receiveShadow = true;
  scene.add(floor);

  const ringGeo = new THREE.RingGeometry(1.55, 1.565, 64);
  const ringMat = new THREE.MeshBasicMaterial({{ color: 0xE2E8F0, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }});
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.002;
  scene.add(ring);

  // ─────────── دوال مساعدة للألوان والخامات الإجرائية ───────────
  function pressureColorRGB(val) {{
    const t = Math.max(0, Math.min(1, val / 100));
    const stops = [
      [0.00, [240, 249, 255]],
      [0.18, [224, 242, 254]],
      [0.34, [125, 211, 252]],
      [0.48, [254, 240, 138]],
      [0.62, [250, 204, 21]],
      [0.76, [249, 115, 22]],
      [0.90, [220, 38, 38]],
      [1.00, [153, 27, 27]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {{
      if (t >= stops[i][0] && t <= stops[i + 1][0]) {{
        const u = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        const a = stops[i][1], b = stops[i + 1][1];
        return [
          Math.round(a[0] + (b[0] - a[0]) * u),
          Math.round(a[1] + (b[1] - a[1]) * u),
          Math.round(a[2] + (b[2] - a[2]) * u),
        ];
      }}
    }}
    return [153, 27, 27];
  }}

  function sampleMatrixBilinear(u, v) {{
    // u in [0,1] across cols 0..7, v in [0,1] across rows 0..15
    const gx = Math.max(0, Math.min(7, u * 7));
    const gy = Math.max(0, Math.min(15, v * 15));
    const c0 = Math.floor(gx), c1 = Math.min(7, c0 + 1);
    const r0 = Math.floor(gy), r1 = Math.min(15, r0 + 1);
    const tx = gx - c0, ty = gy - r0;
    const m = DATA.matrix;
    const v00 = m[r0][c0], v10 = m[r0][c1];
    const v01 = m[r1][c0], v11 = m[r1][c1];
    const top = v00 * (1 - tx) + v10 * tx;
    const bot = v01 * (1 - tx) + v11 * tx;
    return top * (1 - ty) + bot * ty;
  }}

  // 1) خامة الغطاء القطني المبطّن بنقشة مربعات ناعمة (مع خيار دمج حراري ناعم)
  function makeQuiltedCoverTexture(withHeatmapBlend) {{
    const cw = 256, ch = 512;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const ctx = cv.getContext('2d');

    for (let y = 0; y < ch; y += 2) {{
      for (let x = 0; x < cw; x += 2) {{
        const u = x / (cw - 1), v = y / (ch - 1);
        let r = 252, g = 253, b = 255;
        if (withHeatmapBlend) {{
          const p = sampleMatrixBilinear(u, v);
          const hc = pressureColorRGB(p);
          const blend = p > 40 ? 0.62 : 0.24;
          r = Math.round(r * (1 - blend) + hc[0] * blend);
          g = Math.round(g * (1 - blend) + hc[1] * blend);
          b = Math.round(b * (1 - blend) + hc[2] * blend);
        }}
        ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
        ctx.fillRect(x, y, 2, 2);
      }}
    }}

    // خياطة مربعات قطنية ناعمة
    ctx.strokeStyle = withHeatmapBlend ? 'rgba(148, 163, 184, 0.30)' : 'rgba(148, 163, 184, 0.38)';
    ctx.lineWidth = 1.2;
    const step = 24;
    for (let x = 0; x <= cw; x += step) {{
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
    }}
    for (let y = 0; y <= ch; y += step) {{
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    }}
    // إطار محيطي أنيق
    ctx.strokeStyle = 'rgba(2, 132, 199, 0.35)';
    ctx.lineWidth = 6;
    ctx.strokeRect(4, 4, cw - 8, ch - 8);

    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 4;
    return tex;
  }}

  // 2) خامة شريحة الحساسات المرنة (16×8) باللون الفضي والأزرق السماوي
  function makeSensorCircuitTexture() {{
    const cw = 256, ch = 512;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const ctx = cv.getContext('2d');

    ctx.fillStyle = '#E2E8F0';
    ctx.fillRect(0, 0, cw, ch);

    // مسارات فضية وسماوية دقيقة للشبكة 16×8
    for (let c = 0; c < 8; c++) {{
      const x = ((c + 0.5) / 8) * cw;
      ctx.strokeStyle = '#94A3B8';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, ch - 8); ctx.stroke();
    }}
    for (let r = 0; r < 16; r++) {{
      const y = ((r + 0.5) / 16) * ch;
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(8, y); ctx.lineTo(cw - 8, y); ctx.stroke();
    }}

    // تلوين الخلايا حسب قيم مصفوفة الضغط اللحظية
    for (let r = 0; r < 16; r++) {{
      for (let c = 0; c < 8; c++) {{
        const val = DATA.matrix[r][c];
        const rgb = pressureColorRGB(val);
        const cx = ((c + 0.5) / 8) * cw;
        const cy = ((r + 0.5) / 16) * ch;
        const grd = ctx.createRadialGradient(cx, cy, 2, cx, cy, 16);
        grd.addColorStop(0, `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},0.98)`);
        grd.addColorStop(1, `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},0.22)`);
        ctx.fillStyle = grd;
        ctx.fillRect(cx - 16, cy - 16, 32, 32);

        ctx.strokeStyle = 'rgba(255,255,255,0.75)';
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - 10, cy - 10, 20, 20);
      }}
    }}

    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 4;
    return tex;
  }}

  // 3) خامة شبكة التهوية المسامية ثلاثية الأبعاد (3D Spacer Mesh)
  function makeSpacerMeshTexture(climate) {{
    const cw = 256, ch = 512;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const ctx = cv.getContext('2d');

    if (climate === 'cool') ctx.fillStyle = '#E0F2FE';
    else if (climate === 'heat') ctx.fillStyle = '#FFEDD5';
    else ctx.fillStyle = '#E2E8F0';
    ctx.fillRect(0, 0, cw, ch);

    // قنوات طولية للتبريد/التدفئة
    const chColor = climate === 'cool' ? 'rgba(14, 165, 233, 0.32)'
                  : climate === 'heat' ? 'rgba(249, 115, 22, 0.34)'
                  : 'rgba(100, 116, 139, 0.16)';
    for (let i = 1; i <= 5; i++) {{
      const x = (i / 6) * cw;
      ctx.strokeStyle = chColor;
      ctx.lineWidth = 12;
      ctx.beginPath(); ctx.moveTo(x, 14); ctx.lineTo(x, ch - 14); ctx.stroke();
    }}

    // ثقوب المسام السداسية الدقيقة
    ctx.fillStyle = 'rgba(51, 65, 85, 0.25)';
    for (let y = 8; y < ch; y += 10) {{
      const offset = ((y / 10) % 2) * 5;
      for (let x = 8 + offset; x < cw; x += 10) {{
        ctx.beginPath(); ctx.arc(x, y, 2.1, 0, Math.PI * 2); ctx.fill();
      }}
    }}

    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }}

  // 4) خامة الإسفنج الطبي المثقّب (بيج فاتح بارد)
  function makePerforatedFoamTexture() {{
    const cw = 256, ch = 512;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const ctx = cv.getContext('2d');

    ctx.fillStyle = '#ECE7DC';
    ctx.fillRect(0, 0, cw, ch);

    for (let y = 16; y < ch - 10; y += 20) {{
      for (let x = 16; x < cw - 10; x += 20) {{
        ctx.fillStyle = '#C9C0B1';
        ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#B5AA99';
        ctx.beginPath(); ctx.arc(x + 0.8, y + 0.8, 2.6, 0, Math.PI * 2); ctx.fill();
      }}
    }}
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }}

  // 5) خامة القاعدة الرمادية المانعة للانزلاق
  function makeAntiSlipTexture() {{
    const cw = 128, ch = 256;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#CBD5E1';
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = '#94A3B8';
    for (let y = 6; y < ch; y += 8) {{
      for (let x = 6; x < cw; x += 8) {{
        ctx.fillRect(x, y, 3, 3);
      }}
    }}
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }}

  // ─────────── 1. بناء هيكل سرير المستشفى الحديث والمرتبة الأساسية ───────────
  const bedGroup = new THREE.Group();
  scene.add(bedGroup);

  const W = 0.96, L = 2.10; // أبعاد المفرش الذكي (عرض × طول)
  const BED_W = 1.04, BED_L = 2.22;
  const whiteFrameMat = new THREE.MeshStandardMaterial({{ color: 0xFFFFFF, roughness: 0.35, metalness: 0.08 }});
  const metalLegMat = new THREE.MeshStandardMaterial({{ color: 0xCBD5E1, roughness: 0.3, metalness: 0.55 }});
  const accentBlueMat = new THREE.MeshStandardMaterial({{ color: 0x0284C7, roughness: 0.35, metalness: 0.1 }});
  const baseMattressMat = new THREE.MeshStandardMaterial({{ color: 0xF1F5F9, roughness: 0.72, metalness: 0.02 }});

  // منصة السرير السفلية
  const deck = new THREE.Mesh(new THREE.BoxGeometry(BED_W, 0.07, BED_L), whiteFrameMat);
  deck.position.y = 0.33;
  deck.castShadow = true; deck.receiveShadow = true;
  bedGroup.add(deck);

  // شريط طبي أزرق رفيع حول إطار السرير
  const trimStrip = new THREE.Mesh(new THREE.BoxGeometry(BED_W + 0.008, 0.014, BED_L + 0.008), accentBlueMat);
  trimStrip.position.y = 0.33;
  bedGroup.add(trimStrip);

  // المرتبة الأساسية لسرير المستشفى (فوقها يوضع المفرش الطبي الذكي)
  const baseMattress = new THREE.Mesh(new THREE.BoxGeometry(W + 0.02, 0.155, L + 0.02), baseMattressMat);
  baseMattress.position.y = 0.442;
  baseMattress.castShadow = true; baseMattress.receiveShadow = true;
  bedGroup.add(baseMattress);

  // لوح الرأس ولوح القدمين بتصميم طبي مبسّط
  function makeBedBoard(zPos, height) {{
    const g = new THREE.Group();
    const panel = new THREE.Mesh(new THREE.BoxGeometry(BED_W, height, 0.045), whiteFrameMat);
    panel.position.y = 0.33 + height / 2 - 0.02;
    panel.castShadow = true;
    g.add(panel);
    const handleSlot = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.035, 0.052), accentBlueMat);
    handleSlot.position.y = 0.33 + height - 0.085;
    g.add(handleSlot);
    g.position.z = zPos;
    return g;
  }}
  bedGroup.add(makeBedBoard(-BED_L / 2 - 0.02, 0.46)); // لوح الرأس
  bedGroup.add(makeBedBoard(BED_L / 2 + 0.02, 0.34));  // لوح القدمين

  // قوائم السرير والعجلات الطبية الأربع
  const wheelMat = new THREE.MeshStandardMaterial({{ color: 0x64748B, roughness: 0.5 }});
  [[-0.44, -0.96], [0.44, -0.96], [-0.44, 0.96], [0.44, 0.96]].forEach(([lx, lz]) => {{
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.26, 16), metalLegMat);
    leg.position.set(lx, 0.17, lz);
    leg.castShadow = true;
    bedGroup.add(leg);

    const caster = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.026, 16), wheelMat);
    caster.rotation.z = Math.PI / 2;
    caster.position.set(lx, 0.038, lz);
    bedGroup.add(caster);
  }});

  // حواجز جانبية منخفضة أنيقة
  [-1, 1].forEach((side) => {{
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.055, 0.95), whiteFrameMat);
    rail.position.set(side * (BED_W / 2 + 0.012), 0.45, -0.22);
    bedGroup.add(rail);
  }});

  // ─────────── 2. بناء الطبقات الخمس الواقعية للمفرش الطبي الذكي ───────────
  const BED_TOP_Y = 0.52;
  // ترتيب الطبقات من الأسفل (الطبقة 5) إلى الأعلى (الطبقة 1)
  const layersGroup = new THREE.Group();
  scene.add(layersGroup);

  // الطبقة 5: قاعدة مانعة للانزلاق + أحزمة التثبيت المطاطية في الزوايا
  const layer5Group = new THREE.Group();
  const antiSlipTex = makeAntiSlipTexture();
  const l5Mesh = new THREE.Mesh(
    new THREE.BoxGeometry(W, 0.008, L),
    new THREE.MeshStandardMaterial({{ map: antiSlipTex, roughness: 0.85, color: 0xE2E8F0 }})
  );
  l5Mesh.castShadow = true; l5Mesh.receiveShadow = true;
  layer5Group.add(l5Mesh);

  // الأحزمة المطاطية الأربعة في الزوايا المثبّتة حول المرتبة الأساسية
  const strapMat = new THREE.MeshStandardMaterial({{ color: 0x94A3B8, roughness: 0.6, metalness: 0.05 }});
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sz]) => {{
    const strap = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.095, 0.016), strapMat);
    strap.position.set(sx * (W / 2 - 0.05), -0.048, sz * (L / 2 + 0.011));
    strap.rotation.y = sx * sz * 0.32;
    layer5Group.add(strap);

    const strapSide = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.095, 0.14), strapMat);
    strapSide.position.set(sx * (W / 2 + 0.011), -0.048, sz * (L / 2 - 0.05));
    layer5Group.add(strapSide);
  }});
  layersGroup.add(layer5Group);

  // الطبقة 4: إسفنج طبي مثقّب (بيج فاتح بارد) لتوزيع الضغط والتهوية
  const layer4Group = new THREE.Group();
  const foamTopMat = new THREE.MeshStandardMaterial({{ map: makePerforatedFoamTexture(), roughness: 0.78 }});
  const foamSideMat = new THREE.MeshStandardMaterial({{ color: 0xE7E0D3, roughness: 0.82 }});
  const l4Mesh = new THREE.Mesh(
    new THREE.BoxGeometry(W - 0.006, 0.013, L - 0.006),
    [foamSideMat, foamSideMat, foamTopMat, foamSideMat, foamSideMat, foamSideMat]
  );
  l4Mesh.castShadow = true; l4Mesh.receiveShadow = true;
  layer4Group.add(l4Mesh);
  layersGroup.add(layer4Group);

  // الطبقة 3: شبكة التهوية ثلاثية الأبعاد (3D Spacer Mesh) وقنوات التبريد/التدفئة
  const layer3Group = new THREE.Group();
  const spacerTopMat = new THREE.MeshStandardMaterial({{
    map: makeSpacerMeshTexture(DATA.climate),
    roughness: 0.55,
    emissive: DATA.climate === 'heat' ? new THREE.Color(0xF97316)
            : DATA.climate === 'cool' ? new THREE.Color(0x0EA5E9)
            : new THREE.Color(0x000000),
    emissiveIntensity: DATA.climate === 'heat' ? 0.22 : DATA.climate === 'cool' ? 0.12 : 0.0,
  }});
  const spacerSideMat = new THREE.MeshStandardMaterial({{
    color: DATA.climate === 'cool' ? 0xBAE6FD : DATA.climate === 'heat' ? 0xFED7AA : 0xCBD5E1,
    roughness: 0.5, transparent: true, opacity: 0.92,
  }});
  const l3Mesh = new THREE.Mesh(
    new THREE.BoxGeometry(W - 0.012, 0.010, L - 0.012),
    [spacerSideMat, spacerSideMat, spacerTopMat, spacerSideMat, spacerSideMat, spacerSideMat]
  );
  layer3Group.add(l3Mesh);

  // قنوات التكييف الداخلية وتيارات الجسيمات الهوائية / التوهج الدافئ في الطبقة الثالثة
  const airflowParticles = [];
  const heatTubes = [];
  for (let i = -2; i <= 2; i++) {{
    const tubeGeo = new THREE.CylinderGeometry(0.007, 0.007, L - 0.14, 10);
    tubeGeo.rotateX(Math.PI / 2);
    const tubeMat = new THREE.MeshStandardMaterial({{
      color: DATA.climate === 'cool' ? 0x38BDF8 : DATA.climate === 'heat' ? 0xFB923C : 0x94A3B8,
      emissive: DATA.climate === 'cool' ? 0x0284C7 : DATA.climate === 'heat' ? 0xEA580C : 0x000000,
      emissiveIntensity: (DATA.climate === 'cool' || DATA.climate === 'heat') ? 0.45 : 0.0,
      transparent: true, opacity: 0.85,
    }});
    const chTube = new THREE.Mesh(tubeGeo, tubeMat);
    chTube.position.set(i * 0.16, 0.002, 0);
    layer3Group.add(chTube);
    heatTubes.push(tubeMat);
  }}

  if (DATA.climate === 'cool') {{
    const pGeo = new THREE.SphereGeometry(0.011, 8, 8);
    const pMat = new THREE.MeshBasicMaterial({{ color: 0x38BDF8, transparent: true, opacity: 0.85 }});
    for (let k = 0; k < 56; k++) {{
      const sp = new THREE.Mesh(pGeo, pMat.clone());
      const lane = ((k % 5) - 2) * 0.16 + (Math.random() - 0.5) * 0.05;
      sp.userData = {{
        lane: lane,
        phase: Math.random(),
        speed: 0.24 + Math.random() * 0.16,
        rise: 0.018 + Math.random() * 0.032,
      }};
      layer3Group.add(sp);
      airflowParticles.push(sp);
    }}
  }} else if (DATA.climate === 'heat') {{
    const pGeo = new THREE.SphereGeometry(0.013, 8, 8);
    const pMat = new THREE.MeshBasicMaterial({{ color: 0xFB923C, transparent: true, opacity: 0.65 }});
    for (let k = 0; k < 36; k++) {{
      const sp = new THREE.Mesh(pGeo, pMat.clone());
      const lane = ((k % 5) - 2) * 0.16 + (Math.random() - 0.5) * 0.07;
      sp.userData = {{
        lane: lane,
        phase: Math.random(),
        speed: 0.12 + Math.random() * 0.08,
        rise: 0.015 + Math.random() * 0.022,
      }};
      layer3Group.add(sp);
      airflowParticles.push(sp);
    }}
  }}
  layersGroup.add(layer3Group);

  // الطبقة 2: شريحة الحساسات المرنة (16×8) مع تموّج ثلاثي الأبعاد واضح عند مناطق الضغط
  const layer2Group = new THREE.Group();
  const sensorBase = new THREE.Mesh(
    new THREE.BoxGeometry(W - 0.016, 0.004, L - 0.016),
    new THREE.MeshStandardMaterial({{ map: makeSensorCircuitTexture(), roughness: 0.32, metalness: 0.35 }})
  );
  layer2Group.add(sensorBase);

  // سطح تموّج ثلاثي الأبعاد للضغط (Relief Surface) + نقاط الحساسات الـ 128
  const segX = 40, segZ = 80;
  const waveGeo = new THREE.PlaneGeometry(W - 0.03, L - 0.03, segX, segZ);
  waveGeo.rotateX(-Math.PI / 2);
  const posAttr = waveGeo.attributes.position;
  const colors = new Float32Array(posAttr.count * 3);
  for (let i = 0; i < posAttr.count; i++) {{
    const vx = posAttr.getX(i);
    const vz = posAttr.getZ(i);
    const u = (vx + (W - 0.03) / 2) / (W - 0.03);
    const v = (vz + (L - 0.03) / 2) / (L - 0.03);
    const pVal = sampleMatrixBilinear(u, v);
    const extra = Math.max(0, pVal - 26);
    const height = 0.003 + Math.pow(extra / 70, 1.25) * 0.052;
    posAttr.setY(i, height);

    const rgb = pressureColorRGB(pVal);
    colors[i * 3] = rgb[0] / 255;
    colors[i * 3 + 1] = rgb[1] / 255;
    colors[i * 3 + 2] = rgb[2] / 255;
  }}
  waveGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  waveGeo.computeVertexNormals();
  const waveMesh = new THREE.Mesh(
    waveGeo,
    new THREE.MeshStandardMaterial({{
      vertexColors: true, roughness: 0.36, metalness: 0.08,
      transparent: true, opacity: 0.88, side: THREE.DoubleSide,
    }})
  );
  layer2Group.add(waveMesh);

  // عقد الحساسات الـ 128 (16×8)
  const padGeo = new THREE.BoxGeometry(0.042, 0.005, 0.042);
  for (let r = 0; r < 16; r++) {{
    for (let c = 0; c < 8; c++) {{
      const val = DATA.matrix[r][c];
      const rgb = pressureColorRGB(val);
      const colHex = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
      const extra = Math.max(0, val - 26);
      const lift = 0.004 + Math.pow(extra / 70, 1.25) * 0.052;
      const pad = new THREE.Mesh(
        padGeo,
        new THREE.MeshStandardMaterial({{
          color: colHex,
          roughness: 0.28,
          metalness: 0.25,
          emissive: val >= 48 ? colHex : 0x000000,
          emissiveIntensity: val >= 48 ? 0.28 : 0.0,
        }})
      );
      const px = -W / 2 + ((c + 0.5) / 8) * W;
      const pz = -L / 2 + ((r + 0.5) / 16) * L;
      pad.position.set(px, lift, pz);
      layer2Group.add(pad);
    }}
  }}
  layersGroup.add(layer2Group);

  // الطبقة 1 (العليا): غطاء قطني طبي أبيض مبطّن بنقشة مربعات (يعكس حرارة الضغط بنعومة في الوضع المدمج)
  const layer1Group = new THREE.Group();
  const coverTexMerged = makeQuiltedCoverTexture(true);
  const coverTexPure = makeQuiltedCoverTexture(false);

  const coverGeo = new THREE.PlaneGeometry(W, L, 36, 72);
  coverGeo.rotateX(-Math.PI / 2);
  const cPos = coverGeo.attributes.position;
  const baseCoverHeights = new Float32Array(cPos.count);
  for (let i = 0; i < cPos.count; i++) {{
    const vx = cPos.getX(i), vz = cPos.getZ(i);
    const u = (vx + W / 2) / W, v = (vz + L / 2) / L;
    const pVal = sampleMatrixBilinear(u, v);
    const extra = Math.max(0, pVal - 30);
    baseCoverHeights[i] = Math.pow(extra / 72, 1.3) * 0.024;
    cPos.setY(i, 0.0045 + baseCoverHeights[i]);
  }}
  coverGeo.computeVertexNormals();

  const coverSideShell = new THREE.Mesh(
    new THREE.BoxGeometry(W, 0.008, L),
    new THREE.MeshStandardMaterial({{ map: coverTexPure, roughness: 0.65, color: 0xFFFFFF }})
  );
  coverSideShell.castShadow = true; coverSideShell.receiveShadow = true;
  layer1Group.add(coverSideShell);

  const coverHeatmapOverlay = new THREE.Mesh(
    coverGeo,
    new THREE.MeshStandardMaterial({{
      map: coverTexMerged, roughness: 0.58, transparent: true, opacity: 0.94,
    }})
  );
  layer1Group.add(coverHeatmapOverlay);
  layersGroup.add(layer1Group);

  // ─────────── 3. وحدة التحكم الجانبية المتصلة بحافة المفرش ───────────
  const ctrlUnitGroup = new THREE.Group();
  // نثبتها بأناقة بجانب حافة السرير من جهة الرأس
  const unitX = BED_W / 2 + 0.13, unitY = 0.49, unitZ = -0.58;
  ctrlUnitGroup.position.set(unitX, unitY, unitZ);

  // جسم الوحدة (أبيض مطفي مع واجهة سماوية)
  const unitBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.11, 0.15, 0.19),
    new THREE.MeshStandardMaterial({{ color: 0xF8FAFC, roughness: 0.42, metalness: 0.06 }})
  );
  unitBody.castShadow = true;
  ctrlUnitGroup.add(unitBody);

  const unitBezel = new THREE.Mesh(
    new THREE.BoxGeometry(0.114, 0.032, 0.194),
    new THREE.MeshStandardMaterial({{ color: 0x38BDF8, roughness: 0.35, metalness: 0.12 }})
  );
  unitBezel.position.y = 0.045;
  ctrlUnitGroup.add(unitBezel);

  // فتحة التهوية الصغيرة (شرائح غائرة دقيقة)
  const slatMat = new THREE.MeshStandardMaterial({{ color: 0x475569, roughness: 0.7 }});
  for (let s = -2; s <= 2; s++) {{
    const slat = new THREE.Mesh(new THREE.BoxGeometry(0.116, 0.005, 0.085), slatMat);
    slat.position.set(0, -0.028 + s * 0.012, 0.0);
    ctrlUnitGroup.add(slat);
  }}

  // مؤشر إضاءة LED متغير اللون حسب الحالة
  const ledColorObj = new THREE.Color(DATA.ledHex);
  const ledMat = new THREE.MeshStandardMaterial({{
    color: ledColorObj,
    emissive: ledColorObj,
    emissiveIntensity: 1.4,
    roughness: 0.2,
  }});
  const ledBulb = new THREE.Mesh(new THREE.SphereGeometry(0.013, 16, 16), ledMat);
  ledBulb.position.set(0, 0.078, -0.05);
  ctrlUnitGroup.add(ledBulb);

  const ledGlowLight = new THREE.PointLight(ledColorObj, 0.45, 0.45);
  ledGlowLight.position.set(0, 0.095, -0.05);
  ctrlUnitGroup.add(ledGlowLight);

  // ذراع تثبيت جانبي صغير بالسرير
  const mountArm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.018, 0.03), metalLegMat);
  mountArm.position.set(-0.08, -0.04, 0);
  ctrlUnitGroup.add(mountArm);
  scene.add(ctrlUnitGroup);

  // السلك الرفيع المتصل بين وحدة التحكم وحافة المفرش الذكي
  const cableCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(unitX - 0.04, unitY + 0.02, unitZ + 0.06),
    new THREE.Vector3(BED_W / 2 + 0.04, 0.50, unitZ + 0.12),
    new THREE.Vector3(W / 2 - 0.01, BED_TOP_Y + 0.018, unitZ + 0.16),
  ]);
  const cableMesh = new THREE.Mesh(
    new THREE.TubeGeometry(cableCurve, 24, 0.0065, 10, false),
    new THREE.MeshStandardMaterial({{ color: 0x38BDF8, roughness: 0.4 }})
  );
  scene.add(cableMesh);

  // ─────────── 4. مجسم المريض الشفاف (Frosted Glass Mannequin) ───────────
  const patientGroup = new THREE.Group();
  const frostedMat = new THREE.MeshPhysicalMaterial({{
    color: 0xE6F4FE,
    emissive: 0xBAE6FD,
    emissiveIntensity: 0.10,
    transparent: true,
    opacity: 0.48,
    roughness: 0.22,
    metalness: 0.04,
    clearcoat: 0.75,
    clearcoatRoughness: 0.18,
    depthWrite: false,
  }});

  function addBodyPart(geo, x, y, z, sx, sy, sz, rx, ry, rz) {{
    const mesh = new THREE.Mesh(geo, frostedMat);
    mesh.position.set(x, y, z);
    mesh.scale.set(sx, sy, sz);
    if (rx) mesh.rotation.x = rx;
    if (ry) mesh.rotation.y = ry;
    if (rz) mesh.rotation.z = rz;
    patientGroup.add(mesh);
    return mesh;
  }}

  const sph = new THREE.SphereGeometry(1, 24, 20);
  const cyl = new THREE.CylinderGeometry(0.88, 0.95, 1, 20);
  cyl.rotateX(Math.PI / 2);

  // الرأس والرقبة (فوق الصفوف 1–2: z ≈ -0.88)
  addBodyPart(sph, 0, 0.085, -0.87, 0.095, 0.092, 0.115, 0, 0, 0);
  addBodyPart(cyl, 0, 0.062, -0.73, 0.052, 0.048, 0.09, 0, 0, 0);

  // الكتفان وأعلى الصدر (فوق الصفوف 3–4: z ≈ -0.62)
  addBodyPart(sph, 0, 0.074, -0.59, 0.225, 0.068, 0.115, 0, 0, 0);

  // الظهر والجذع الأوسط (فوق الصفوف 5–6: z ≈ -0.38)
  addBodyPart(sph, 0, 0.072, -0.36, 0.178, 0.072, 0.175, 0, 0, 0);

  // منطقة الحوض (فوق الصفوف 7–10: z ≈ -0.04)
  addBodyPart(sph, 0, 0.072, -0.04, 0.192, 0.072, 0.165, 0, 0, 0);

  // الذراعان واليدان في الأعمدة الطرفية (فوق الصفوف 4–9: z = -0.60 .. +0.08, x = ±0.32)
  [-1, 1].forEach((side) => {{
    // مفصل الكتف والعضد
    addBodyPart(sph, side * 0.255, 0.066, -0.60, 0.058, 0.052, 0.062, 0, 0, 0);
    addBodyPart(cyl, side * 0.305, 0.052, -0.43, 0.046, 0.040, 0.28, 0, side * -0.16, 0);
    // الساعد
    addBodyPart(cyl, side * 0.335, 0.044, -0.14, 0.040, 0.035, 0.27, 0, side * -0.05, 0);
    // اليد
    addBodyPart(sph, side * 0.342, 0.036, 0.05, 0.040, 0.022, 0.068, 0, 0, 0);
  }});

  // الفخذان (فوق الصفوف 11–13: z ≈ +0.30) والساقان والكعبان/القدمان (فوق الصفوف 14–16: z ≈ +0.72 .. +0.94)
  [-1, 1].forEach((side) => {{
    // الفخذ
    addBodyPart(cyl, side * 0.108, 0.062, 0.29, 0.076, 0.062, 0.39, 0, 0, 0);
    // الركبة
    addBodyPart(sph, side * 0.105, 0.054, 0.51, 0.058, 0.050, 0.058, 0, 0, 0);
    // الساق
    addBodyPart(cyl, side * 0.102, 0.046, 0.71, 0.055, 0.046, 0.35, 0, 0, 0);
    // الكعب والقدم بارتفاع طبيعي للأصابع
    addBodyPart(sph, side * 0.102, 0.036, 0.90, 0.044, 0.034, 0.048, 0, 0, 0);
    addBodyPart(sph, side * 0.102, 0.088, 0.93, 0.045, 0.082, 0.036, -0.18, 0, 0);
  }});

  scene.add(patientGroup);

  // ─────────── 5. تفكيك الطبقات التفاعلي والتسميات المترجمة ───────────
  const layerMeta = [
    {{
      group: layer5Group,
      baseY: BED_TOP_Y + 0.004,
      explodeDelta: 0.00,
      title: TR.l5_title,
      sub: TR.l5_sub,
    }},
    {{
      group: layer4Group,
      baseY: BED_TOP_Y + 0.0145,
      explodeDelta: 0.145,
      title: TR.l4_title,
      sub: TR.l4_sub,
    }},
    {{
      group: layer3Group,
      baseY: BED_TOP_Y + 0.026,
      explodeDelta: 0.295,
      title: TR.l3_title,
      sub: TR.l3_sub,
    }},
    {{
      group: layer2Group,
      baseY: BED_TOP_Y + 0.033,
      explodeDelta: 0.445,
      title: TR.l2_title,
      sub: TR.l2_sub,
    }},
    {{
      group: layer1Group,
      baseY: BED_TOP_Y + 0.040,
      explodeDelta: 0.595,
      title: TR.l1_title,
      sub: TR.l1_sub,
    }},
  ];

  const tagElements = layerMeta.map((lm) => {{
    const el = document.createElement('div');
    el.className = 'layer-tag';
    el.innerHTML = `<span class="lt-title">${{lm.title}}</span><span class="lt-sub">${{lm.sub}}</span>`;
    labelsLayer.appendChild(el);
    return el;
  }});

  let currentExplode = 0.0;
  let targetExplode = 0.0;
  let patientVisible = true;

  const btnExplode = document.getElementById('btnExplode');
  const rngExplode = document.getElementById('rngExplode');
  const btnPatient = document.getElementById('btnPatient');
  const btnResetCam = document.getElementById('btnResetCam');

  function syncExplodeUI(val) {{
    rngExplode.value = Math.round(val * 100);
    if (val > 0.5) {{
      btnExplode.classList.add('active');
      btnExplode.textContent = TR.btn_explode_close;
    }} else {{
      btnExplode.classList.remove('active');
      btnExplode.textContent = TR.btn_explode_open;
    }}
  }}

  btnExplode.addEventListener('click', () => {{
    targetExplode = targetExplode > 0.5 ? 0.0 : 1.0;
    syncExplodeUI(targetExplode);
  }});

  rngExplode.addEventListener('input', (e) => {{
    targetExplode = parseFloat(e.target.value) / 100.0;
    syncExplodeUI(targetExplode);
  }});

  btnPatient.addEventListener('click', () => {{
    patientVisible = !patientVisible;
    patientGroup.visible = patientVisible;
    btnPatient.classList.toggle('active', patientVisible);
    btnPatient.textContent = patientVisible ? TR.btn_hide_patient : TR.btn_show_patient;
  }});

  btnResetCam.addEventListener('click', () => {{
    camera.position.copy(DEFAULT_CAM);
    controls.target.set(DEFAULT_TARGET.x, DEFAULT_TARGET.y + currentExplode * 0.24, DEFAULT_TARGET.z);
    controls.update();
  }});

  // تحديث مواضع الطبقات والتسميات وحركة الجسيمات
  const projVec = new THREE.Vector3();
  const clock = new THREE.Clock();

  function applyExplodeState(e) {{
    layerMeta.forEach((lm, idx) => {{
      lm.group.position.y = lm.baseY + lm.explodeDelta * e;

      const tag = tagElements[idx];
      if (e > 0.18) {{
        projVec.set(-W / 2 - 0.04, lm.group.position.y, 0.18);
        projVec.project(camera);
        const sx = (projVec.x * 0.5 + 0.5) * stage.clientWidth;
        const sy = (-projVec.y * 0.5 + 0.5) * stage.clientHeight;
        const tagW = tag.offsetWidth || 195;
        const clampedX = Math.max(tagW + 10, Math.min(stage.clientWidth - 10, sx));
        const clampedY = Math.max(82, Math.min(stage.clientHeight - 46, sy));
        tag.style.left = `${{clampedX}}px`;
        tag.style.top = `${{clampedY}}px`;
        tag.style.opacity = Math.min(1, (e - 0.18) / 0.35).toFixed(2);
      }} else {{
        tag.style.opacity = '0';
      }}
    }});

    // في الوضع المدمج يظهر انعكاس الضغط الحراري بنعومة فوق الغطاء، وعند الفصل يظهر الغطاء الأبيض النقي
    coverHeatmapOverlay.material.opacity = Math.max(0.0, 0.94 * (1.0 - e * 1.15));
    coverHeatmapOverlay.visible = coverHeatmapOverlay.material.opacity > 0.02;

    // رفع مجسم المريض بسلاسة فوق الطبقة العليا
    patientGroup.position.y = BED_TOP_Y + 0.044 + 0.73 * e;
  }}

  applyExplodeState(0);

  function animate() {{
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.getElapsedTime();

    if (Math.abs(targetExplode - currentExplode) > 0.0008) {{
      const prev = currentExplode;
      currentExplode += (targetExplode - currentExplode) * Math.min(1, dt * 6.5);
      controls.target.y += (currentExplode - prev) * 0.24;
    }} else {{
      currentExplode = targetExplode;
    }}
    applyExplodeState(currentExplode);

    // تحريك جسيمات التهوية الزرقاء أو التوهج الحراري في الطبقة الثالثة
    if (airflowParticles.length > 0) {{
      for (let i = 0; i < airflowParticles.length; i++) {{
        const p = airflowParticles[i];
        const u = (p.userData.phase + elapsed * p.userData.speed) % 1.0;
        const z = -L / 2 + 0.10 + u * (L - 0.20);
        const yLift = Math.sin(u * Math.PI) * (p.userData.rise + currentExplode * 0.028);
        p.position.set(p.userData.lane, 0.008 + yLift, z);
        p.material.opacity = Math.sin(u * Math.PI) * (DATA.climate === 'cool' ? 0.85 : 0.62);
      }}
    }}

    if (DATA.climate === 'heat') {{
      const glow = 0.20 + 0.10 * Math.sin(elapsed * 2.6);
      spacerTopMat.emissiveIntensity = glow;
      for (let i = 0; i < heatTubes.length; i++) heatTubes[i].emissiveIntensity = 0.35 + glow;
    }}

    // نبض مؤشر LED في حالة التنبيه
    if (DATA.alert) {{
      const pulse = 0.85 + 0.75 * (0.5 + 0.5 * Math.sin(elapsed * 5.2));
      if (DATA.ledState === 'cool_alert') {{
        // تناوب بصري ناعم بين أزرق التبريد وأحمر التنبيه عند الخطر المركّب مع التبريد
        const mix = 0.5 + 0.5 * Math.sin(elapsed * 4.5);
        ledMat.color.setRGB(0.05 * (1 - mix) + 0.94 * mix, 0.65 * (1 - mix) + 0.27 * mix, 0.91 * (1 - mix) + 0.27 * mix);
        ledMat.emissive.copy(ledMat.color);
      }}
      ledMat.emissiveIntensity = pulse * 1.3;
      ledGlowLight.intensity = 0.32 + pulse * 0.25;
    }}

    controls.update();
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    if (!stage.clientWidth) return;
    camera.aspect = stage.clientWidth / stage.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(stage.clientWidth, stage.clientHeight);
  }});
}})();
</script>
</body>
</html>"""

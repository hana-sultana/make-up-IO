"""
Glam Vault — Python Flask Backend
===================================
Runs locally on http://localhost:5000
Requires: pip install flask flask-cors mediapipe opencv-python Pillow numpy scikit-learn requests

Usage:
  python app.py
Then open http://localhost:5000 in your browser.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json
import os
import io
import numpy as np
import requests
import traceback

# ── PIL / OpenCV / MediaPipe ──
from PIL import Image
import cv2
import mediapipe as mp
from sklearn.cluster import KMeans

app = Flask(__name__, static_folder="static", template_folder="templates")

# Allow requests from your Netlify domain AND localhost for local dev
# Replace YOUR_NETLIFY_SITE with your actual Netlify URL once deployed e.g. https://glamvault.netlify.app
ALLOWED_ORIGINS = [
    os.environ.get("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
CORS(app, origins=ALLOWED_ORIGINS)

# ─────────────────────────────────────────────
# ANTHROPIC CONFIG — set your key here or via env var
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Key landmark indices for feature analysis
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LIPS_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LEFT_EYEBROW = [70,63,105,66,107,55,65,52,53,46]
RIGHT_EYEBROW = [300,293,334,296,336,285,295,282,283,276]
NOSE_TIP = [4]
LEFT_CHEEK = [234,93,132,58,172,136,150,149,176,148,152]
RIGHT_CHEEK = [454,361,323,152,377,400,379,365,397,288,356]


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL to an OpenCV BGR numpy array."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def img_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def landmarks_to_points(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def region_mean_color(img_rgb: np.ndarray, points: list) -> tuple:
    """Average RGB color in a convex hull region."""
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    mean = cv2.mean(img_rgb, mask=mask)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def rgb_to_hex(r, g, b) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def dominant_colors(img_rgb: np.ndarray, n=3) -> list:
    """Return n dominant colors using KMeans."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    # Sample for speed
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]
    km = KMeans(n_clusters=n, n_init=5, random_state=42)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    return [rgb_to_hex(*c) for c in centers]


# ─────────────────────────────────────────────
# FACE SHAPE ANALYSIS (MediaPipe geometry)
# ─────────────────────────────────────────────

def compute_face_shape(landmarks, w, h) -> dict:
    """
    Use facial landmark geometry to classify face shape.
    Measurements: face width, jaw width, face height, forehead width.
    """
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    # Key points
    top = pt(10)          # top of forehead
    bottom = pt(152)      # chin tip
    left_jaw = pt(234)    # left jaw
    right_jaw = pt(454)   # right jaw
    left_cheek = pt(93)   # left cheekbone
    right_cheek = pt(361) # right cheekbone
    left_fore = pt(70)    # left forehead
    right_fore = pt(300)  # right forehead
    left_jaw2 = pt(172)   # lower jaw left
    right_jaw2 = pt(397)  # lower jaw right

    face_height = float(np.linalg.norm(top - bottom))
    face_width = float(np.linalg.norm(left_cheek - right_cheek))
    jaw_width = float(np.linalg.norm(left_jaw2 - right_jaw2))
    forehead_width = float(np.linalg.norm(left_fore - right_fore))

    ratio_hw = face_height / (face_width + 1e-6)
    jaw_ratio = jaw_width / (face_width + 1e-6)
    fore_ratio = forehead_width / (face_width + 1e-6)

    # Classification logic based on proportional analysis
    if ratio_hw > 1.55:
        shape = "oblong"
    elif ratio_hw > 1.35:
        if jaw_ratio < 0.75:
            shape = "heart"
        else:
            shape = "oval"
    elif ratio_hw > 1.15:
        if jaw_ratio > 0.85 and fore_ratio > 0.85:
            shape = "square"
        elif jaw_ratio < 0.72:
            shape = "heart"
        else:
            shape = "oval"
    else:
        if jaw_ratio > 0.82:
            shape = "square"
        else:
            shape = "round"

    return {
        "shape": shape,
        "faceHeight": round(face_height, 1),
        "faceWidth": round(face_width, 1),
        "jawWidth": round(jaw_width, 1),
        "foreheadWidth": round(forehead_width, 1),
        "heightWidthRatio": round(ratio_hw, 3),
        "jawRatio": round(jaw_ratio, 3),
        "foreheadRatio": round(fore_ratio, 3),
    }


# ─────────────────────────────────────────────
# EYE SHAPE ANALYSIS
# ─────────────────────────────────────────────

def compute_eye_shape(landmarks, w, h) -> str:
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    # Left eye key points: inner corner (133), outer corner (33), top (159), bottom (145)
    inner = pt(133)
    outer = pt(33)
    top = pt(159)
    bottom = pt(145)

    eye_width = float(np.linalg.norm(outer - inner))
    eye_height = float(np.linalg.norm(top - bottom))

    # Tilt: compare inner vs outer y position
    tilt = float(outer[1] - inner[1])  # negative = upturned, positive = downturned
    tilt_ratio = tilt / (eye_width + 1e-6)

    hw_ratio = eye_height / (eye_width + 1e-6)

    # Hooded: check if upper lid landmark is close to crease
    lid_top = pt(386)
    crease_dist = float(np.linalg.norm(top - lid_top))
    hooded = crease_dist < eye_height * 0.3

    if hooded:
        return "hooded"
    elif hw_ratio < 0.22:
        return "monolid"
    elif tilt_ratio < -0.08:
        return "upturned"
    elif tilt_ratio > 0.08:
        return "downturned"
    elif hw_ratio > 0.38:
        return "round"
    else:
        return "almond"


# ─────────────────────────────────────────────
# SKIN UNDERTONE (via color science)
# ─────────────────────────────────────────────

def compute_undertone(img_rgb: np.ndarray, landmarks, w, h) -> dict:
    """
    Sample skin color from cheek regions, analyze RGB/HSV to classify undertone.
    Warm: high red/yellow, Cool: high blue/pink, Neutral/Olive: balanced with green tones.
    """
    lm = landmarks.landmark
    cheek_pts_l = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_CHEEK]
    cheek_pts_r = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_CHEEK]

    l_color = region_mean_color(img_rgb, cheek_pts_l)
    r_color = region_mean_color(img_rgb, cheek_pts_r)
    skin_r = int((l_color[0] + r_color[0]) / 2)
    skin_g = int((l_color[1] + r_color[1]) / 2)
    skin_b = int((l_color[2] + r_color[2]) / 2)

    # Convert to HSV for hue analysis
    skin_bgr = np.uint8([[[skin_b, skin_g, skin_r]]])
    hsv = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hue = int(hsv[0])
    sat = int(hsv[1])

    # Ratios
    rg_ratio = skin_r / (skin_g + 1e-6)
    rb_ratio = skin_r / (skin_b + 1e-6)
    gb_ratio = skin_g / (skin_b + 1e-6)

    # Classification
    if gb_ratio > 1.15 and rg_ratio > 1.05:
        undertone = "warm"       # strong red + yellow tones
    elif skin_b > skin_r * 0.72 and hue > 165:
        undertone = "cool"       # pink/blue dominance
    elif abs(rg_ratio - 1.0) < 0.08 and gb_ratio < 1.1:
        undertone = "neutral"
    elif skin_g > skin_r * 0.82 and skin_g > skin_b * 1.05:
        undertone = "olive"
    else:
        undertone = "warm"

    return {
        "undertone": undertone,
        "skinHex": rgb_to_hex(skin_r, skin_g, skin_b),
        "skinRGB": [skin_r, skin_g, skin_b],
        "hue": hue,
        "saturation": sat,
    }


# ─────────────────────────────────────────────
# LIP SHAPE ANALYSIS
# ─────────────────────────────────────────────

def compute_lip_shape(landmarks, w, h) -> str:
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    left_corner = pt(61)
    right_corner = pt(291)
    top_center = pt(0)
    bottom_center = pt(17)
    cupid_peak_l = pt(37)
    cupid_peak_r = pt(267)

    lip_width = float(np.linalg.norm(right_corner - left_corner))
    lip_height = float(np.linalg.norm(top_center - bottom_center))
    hw = lip_height / (lip_width + 1e-6)

    # Cupid's bow detection: peaks above center
    center_y = top_center[1]
    cupid_dip = (cupid_peak_l[1] + cupid_peak_r[1]) / 2 - center_y

    if hw > 0.42:
        return "full"
    elif hw < 0.25:
        return "thin"
    elif cupid_dip > 3:
        return "cupid's bow"
    elif lip_width > w * 0.32:
        return "wide"
    else:
        return "heart-shaped"


# ─────────────────────────────────────────────
# CHEEKBONE PROMINENCE
# ─────────────────────────────────────────────

def compute_cheekbones(landmarks, w, h) -> str:
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    cheek_l = pt(93)
    cheek_r = pt(361)
    jaw_l = pt(172)
    jaw_r = pt(397)

    cheek_w = float(np.linalg.norm(cheek_r - cheek_l))
    jaw_w = float(np.linalg.norm(jaw_r - jaw_l))
    ratio = cheek_w / (jaw_w + 1e-6)

    if ratio > 1.18:
        return "high"
    elif ratio > 1.05:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────────
# BROW SHAPE
# ─────────────────────────────────────────────

def compute_brow_shape(landmarks, w, h) -> str:
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    # Left brow: inner (46), peak (66), outer (70)
    inner = pt(46)
    peak = pt(66)
    outer = pt(70)

    # Arch height relative to brow width
    brow_width = float(np.linalg.norm(outer - inner))
    # Vertical position of peak relative to line between inner and outer
    line_y = inner[1] + (outer[1] - inner[1]) * ((peak[0] - inner[0]) / (brow_width + 1e-6))
    arch_height = float(line_y - peak[1])

    arch_ratio = arch_height / (brow_width + 1e-6)

    if arch_ratio > 0.12:
        return "high arched"
    elif arch_ratio > 0.06:
        return "softly arched"
    elif abs(arch_ratio) < 0.04:
        return "straight"
    else:
        return "flat"


# ─────────────────────────────────────────────
# SKIN TONE (lightness-based classification)
# ─────────────────────────────────────────────

def compute_skin_tone(img_rgb: np.ndarray, landmarks, w, h) -> str:
    lm = landmarks.landmark
    forehead_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in [10, 67, 109, 338, 297]]
    cheek_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_CHEEK]

    try:
        f_color = region_mean_color(img_rgb, forehead_pts)
        c_color = region_mean_color(img_rgb, cheek_pts)
        avg_r = int((f_color[0] + c_color[0]) / 2)
        avg_g = int((f_color[1] + c_color[1]) / 2)
        avg_b = int((f_color[2] + c_color[2]) / 2)
        luminance = 0.2126 * avg_r + 0.7152 * avg_g + 0.0722 * avg_b
    except Exception:
        luminance = 150

    if luminance > 210:
        return "fair"
    elif luminance > 185:
        return "light"
    elif luminance > 160:
        return "light-medium"
    elif luminance > 135:
        return "medium"
    elif luminance > 110:
        return "medium-tan"
    elif luminance > 85:
        return "tan"
    elif luminance > 60:
        return "deep"
    else:
        return "rich"


# ─────────────────────────────────────────────
# MAKEUP TIPS DATABASE (research-backed)
# ─────────────────────────────────────────────

FACE_SHAPE_TIPS = {
    "oval": {
        "headline": "You have the most versatile face shape",
        "contour": "Light contouring under cheekbones keeps your balanced look. Almost any technique works.",
        "blush": "Apply blush to the apples of your cheeks and blend slightly upward.",
        "brow": "Most brow shapes flatter you — a soft arch is universally flattering.",
        "pinterest_keyword": "oval face makeup looks"
    },
    "round": {
        "headline": "Soft, symmetrical features with natural fullness",
        "contour": "Contour along the sides of the face and under cheekbones to add definition and length.",
        "blush": "Apply blush slightly above the hollows of your cheeks, blending toward temples.",
        "brow": "A higher, more angular brow arch adds length and lifts the face.",
        "pinterest_keyword": "round face makeup contouring looks"
    },
    "square": {
        "headline": "Strong jaw and defined structure — so striking",
        "contour": "Soften the corners of your jaw and forehead with bronzer/contour.",
        "blush": "Apply blush in circular motions on the apples of cheeks to soften angles.",
        "brow": "Softer, more rounded brows balance angular features beautifully.",
        "pinterest_keyword": "square face soft glam makeup looks"
    },
    "heart": {
        "headline": "Wider forehead tapering to a delicate chin — ethereal",
        "contour": "Contour temples lightly. Add highlighter to the chin to balance proportions.",
        "blush": "Apply below cheekbones and blend down slightly to add width to lower face.",
        "brow": "Softer, less dramatic arches balance a wider forehead.",
        "pinterest_keyword": "heart face shape makeup tutorial"
    },
    "oblong": {
        "headline": "Elegant long face with refined features",
        "contour": "Add width with blush on the sides of your face. Contour lightly at hairline and chin.",
        "blush": "Apply horizontally across cheeks to add width — avoid vertical blending.",
        "brow": "Straighter, flatter brows visually shorten the face beautifully.",
        "pinterest_keyword": "long oblong face makeup tutorial"
    },
    "diamond": {
        "headline": "Striking high cheekbones and narrow forehead and jaw",
        "contour": "Highlight forehead and jawline to add width. Contour cheekbones lightly.",
        "blush": "Apply blush on cheekbones sweeping outward.",
        "brow": "Full, straight brows add width to the forehead area.",
        "pinterest_keyword": "diamond face shape makeup looks"
    },
}

EYE_SHAPE_TIPS = {
    "almond": "Lucky you — almost any eye look flatters almond eyes. Try a classic smoky cut crease.",
    "round": "Elongate with a flicked liner wing. Shade outer corners darker to add length.",
    "hooded": "Apply eyeshadow slightly above the crease so it's visible when eyes are open. Tight-line the upper lash line.",
    "monolid": "Build shadow above the lash line and blend high. Bold liner and lash-focused looks are stunning.",
    "upturned": "Balance with shadow at outer corners angled downward. Avoid heavy upper wings.",
    "downturned": "Wing liner upward at outer corners. Highlight inner corners to lift.",
    "deep-set": "Highlight the brow bone and lid. Avoid heavy dark shadow on the lid — it deepens eyes further.",
}

UNDERTONE_TIPS = {
    "warm": "Choose foundation and blush with peach, golden, or orange undertones. Gold jewelry and warm browns complement you beautifully.",
    "cool": "Choose foundation with pink or rosy undertones. Berry, mauve, and cool-toned nudes are your best friends.",
    "neutral": "Lucky — you can pull off both warm and cool shades. Focus on the overall depth rather than undertone.",
    "olive": "Look for foundations labeled 'olive' or 'neutral.' Avoid anything too pink (ashy) or too orange. Warm mauves and terracottas are gorgeous on you.",
}

LIP_TIPS = {
    "full": "Full lips are gorgeous — enhance with a clear gloss or bold color. Slightly overline corners to maximize.",
    "thin": "Overline slightly outside your natural lip line. A nude-pink slightly lighter than your natural lip makes them appear fuller.",
    "cupid's bow": "Emphasize with a lip liner that follows your natural shape. A glossy center highlight makes the bow pop.",
    "wide": "Avoid lining corners — keep color slightly inside the corners. A slightly deeper shade in corners adds definition.",
    "heart-shaped": "Balance upper and lower with a liner that follows both arches carefully. A glossy center is magical.",
}


# ─────────────────────────────────────────────
# FULL FACE ANALYSIS
# ─────────────────────────────────────────────

def run_ml_face_analysis(img_bgr: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]
    img_rgb = img_to_rgb(img_bgr)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected. Please try a clearer, well-lit front-facing photo."}

    landmarks = results.multi_face_landmarks[0]

    # Run all analyses
    face_shape_data = compute_face_shape(landmarks, w, h)
    face_shape = face_shape_data["shape"]
    eye_shape = compute_eye_shape(landmarks, w, h)
    undertone_data = compute_undertone(img_rgb, landmarks, w, h)
    lip_shape = compute_lip_shape(landmarks, w, h)
    cheekbones = compute_cheekbones(landmarks, w, h)
    brow_shape = compute_brow_shape(landmarks, w, h)
    skin_tone = compute_skin_tone(img_rgb, landmarks, w, h)

    # Gather tips
    shape_tips = FACE_SHAPE_TIPS.get(face_shape, FACE_SHAPE_TIPS["oval"])
    eye_tip = EYE_SHAPE_TIPS.get(eye_shape, "Experiment freely — your eyes are unique!")
    undertone_tip = UNDERTONE_TIPS.get(undertone_data["undertone"], "")
    lip_tip = LIP_TIPS.get(lip_shape, "Your lips are beautifully unique.")

    features = [
        {"label": "Face Shape", "value": face_shape.title(), "tip": shape_tips["contour"]},
        {"label": "Skin Undertone", "value": undertone_data["undertone"].title(), "tip": undertone_tip},
        {"label": "Eye Shape", "value": eye_shape.title(), "tip": eye_tip},
        {"label": "Lip Shape", "value": lip_shape.title(), "tip": lip_tip},
        {"label": "Cheekbones", "value": cheekbones.title(), "tip": shape_tips["blush"]},
        {"label": "Brow Shape", "value": brow_shape.title(), "tip": shape_tips["brow"]},
        {"label": "Skin Tone", "value": skin_tone.replace("-", " ").title(), "tip": "Match foundation undertone to your detected undertone for a seamless look."},
    ]

    return {
        "faceShape": face_shape,
        "eyeShape": eye_shape,
        "lipShape": lip_shape,
        "skinUndertone": undertone_data["undertone"],
        "skinTone": skin_tone,
        "skinHex": undertone_data["skinHex"],
        "cheekbones": cheekbones,
        "browShape": brow_shape,
        "features": features,
        "shapeTips": shape_tips,
        "pinterestKeyword": shape_tips.get("pinterest_keyword", f"{face_shape} face makeup"),
        "measurements": {
            "heightWidthRatio": face_shape_data["heightWidthRatio"],
            "jawRatio": face_shape_data["jawRatio"],
            "foreheadRatio": face_shape_data["foreheadRatio"],
        },
        "mlAnalysis": True,  # flag: this came from real ML
    }


# ─────────────────────────────────────────────
# CLAUDE AI: generate narrative + looks
# ─────────────────────────────────────────────

def call_claude(system_prompt: str, user_prompt: str, max_tokens=1500) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    r = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["content"][0]["text"]


def generate_face_narrative(ml_data: dict, shade_profile: dict) -> dict:
    """Ask Claude to generate a warm narrative + 6 curated looks from ML data."""
    prompt = f"""You are a warm, expert makeup artist. Using these precise ML measurements from MediaPipe facial analysis, write personalized makeup guidance.

ML Analysis Results:
- Face shape: {ml_data['faceShape']} (height/width ratio: {ml_data['measurements']['heightWidthRatio']}, jaw ratio: {ml_data['measurements']['jawRatio']})
- Eye shape: {ml_data['eyeShape']}
- Lip shape: {ml_data['lipShape']}
- Skin undertone: {ml_data['skinUndertone']} (detected skin hex: {ml_data['skinHex']})
- Skin tone: {ml_data['skinTone']}
- Cheekbones: {ml_data['cheekbones']}
- Brow shape: {ml_data['browShape']}

User's shade preferences:
- Skin tone (self-reported): {shade_profile.get('skinTone', 'not specified')}
- Lip preferences: {', '.join(shade_profile.get('lip', [])) or 'not specified'}
- Eye preferences: {', '.join(shade_profile.get('eye', [])) or 'not specified'}
- Finish: {', '.join(shade_profile.get('finish', [])) or 'not specified'}
- Foundation shade: {shade_profile.get('foundationShade', 'not specified')}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "headline": "6-8 word flattering summary of their face",
  "subtext": "One warm encouraging sentence about their natural beauty",
  "narrative": "3-4 sentences: what makes their face unique, specific techniques for their exact face shape and features, one thing to embrace",
  "looks": [
    {{
      "name": "Look name",
      "vibe": "Everyday|Date Night|Work|Festival|Editorial|Bridal",
      "emoji": "2-3 emojis",
      "bannerClass": "one of: banner-oval banner-round banner-square banner-heart banner-oblong banner-diamond banner-default",
      "description": "2 sentences describing the look and why it suits their specific features",
      "whyItWorks": "1 sentence — the science/technique behind why this flatters their face shape/eye shape",
      "steps": ["step 1 (10-12 words specific to their features)", "step 2", "step 3", "step 4", "step 5"],
      "keyProducts": ["product type 1", "product type 2", "product type 3"],
      "tags": ["tag1", "tag2", "tag3"],
      "pinterestQuery": "specific Pinterest search query 4-8 words"
    }}
  ]
}}
Generate exactly 6 looks. Make them diverse (everyday, glam, editorial, etc.)."""

    text = call_claude(
        "You are an expert makeup artist and beauty consultant. Always respond with valid JSON only.",
        prompt,
        max_tokens=2500
    )
    return json.loads(text.strip())


def generate_product_recommendations(shade_profile: dict, face_data: dict, inventory: list) -> list:
    """Generate 6 personalized product recommendations."""
    owned = [f"{p.get('brand','')} {p.get('name','')}" for p in inventory]
    face_ctx = f"\nFace shape: {face_data.get('faceShape','')}\nEye shape: {face_data.get('eyeShape','')}\nUndertone (ML): {face_data.get('skinUndertone','')}" if face_data else ""

    prompt = f"""Recommend 6 makeup products for this person. Include brands like MAC, NARS, Urban Decay, Charlotte Tilbury, Fenty Beauty, Too Faced, Rare Beauty, Huda Beauty, ABH.

Profile:
Skin tone: {shade_profile.get('skinTone','not specified')}
Foundation shade: {shade_profile.get('foundationShade','not specified')}
Lip preferences: {', '.join(shade_profile.get('lip',[]))}
Eye preferences: {', '.join(shade_profile.get('eye',[]))}
Finish: {', '.join(shade_profile.get('finish',[]))}
Blush: {', '.join(shade_profile.get('blush',[]))}
Products already owned: {', '.join(owned) if owned else 'none'}{face_ctx}

Respond ONLY with a JSON array of 6 objects:
[{{"brand":"","name":"","category":"","shade":"","reason":"1-2 sentences mentioning their specific profile","emoji":"","shadeHex":""}}]"""

    text = call_claude(
        "You are a luxury makeup consultant. Respond with valid JSON only.",
        prompt, max_tokens=1000
    )
    return json.loads(text.strip())


def identify_makeup_product(image_b64: str, media_type: str, known_brands: list) -> dict:
    """Use Claude Vision to identify a makeup product."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": 800,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": f"""You are a makeup product expert. Identify this product.
Known brands: {', '.join(known_brands)}.
Respond ONLY with JSON (no markdown):
{{"brand":"","name":"","category":"","shade":"","shadeHex":"","brandKnown":true,"description":""}}
Category must be one of: Lipstick, Lip Gloss, Lip Liner, Foundation, Concealer, Blush, Bronzer, Highlighter, Eyeshadow Palette, Eyeliner, Mascara, Setting Powder, Setting Spray, Primer, Contour, Other"""}
            ]
        }]
    }
    r = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    text = r.json()["content"][0]["text"]
    return json.loads(text.strip())


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/analyze-face", methods=["POST"])
def api_analyze_face():
    """
    POST { imageData: "data:image/...", shadeProfile: {...} }
    Returns full ML + AI analysis.
    """
    try:
        data = request.json
        image_data_url = data.get("imageData", "")
        shade_profile = data.get("shadeProfile", {})

        if not image_data_url:
            return jsonify({"error": "No image provided"}), 400

        # Step 1: MediaPipe ML analysis
        img_bgr = decode_image(image_data_url)
        ml_result = run_ml_face_analysis(img_bgr)

        if "error" in ml_result:
            return jsonify(ml_result), 422

        # Step 2: Claude generates narrative + looks from ML data
        ai_result = generate_face_narrative(ml_result, shade_profile)

        # Merge everything
        response = {
            **ml_result,
            "headline": ai_result.get("headline", "Your unique face profile"),
            "subtext": ai_result.get("subtext", ""),
            "narrative": ai_result.get("narrative", ""),
            "looks": ai_result.get("looks", []),
        }
        return jsonify(response)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI response parse error: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/identify-product", methods=["POST"])
def api_identify_product():
    """
    POST { imageData: "data:image/...", knownBrands: [...] }
    Returns product identification.
    """
    try:
        data = request.json
        image_data_url = data.get("imageData", "")
        known_brands = data.get("knownBrands", [])

        if not image_data_url:
            return jsonify({"error": "No image provided"}), 400

        header, b64 = image_data_url.split(",", 1)
        media_type = header.split(";")[0].split(":")[1]

        result = identify_makeup_product(b64, media_type, known_brands)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations", methods=["POST"])
def api_recommendations():
    """
    POST { shadeProfile: {...}, faceData: {...}, inventory: [...] }
    Returns 6 product recommendations.
    """
    try:
        data = request.json
        shade_profile = data.get("shadeProfile", {})
        face_data = data.get("faceData", {})
        inventory = data.get("inventory", [])

        recs = generate_product_recommendations(shade_profile, face_data, inventory)
        return jsonify({"recommendations": recs})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ml": "mediapipe+opencv+sklearn", "ai": "claude-sonnet"})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None  # debug only in local dev
    print("\n✨ Glam Vault Backend Starting...")
    print("━" * 45)
    print("  ML Engine : MediaPipe + OpenCV + scikit-learn")
    print("  AI Engine : Claude Sonnet")
    print(f"  URL       : http://localhost:{port}")
    print("━" * 45)
    if ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  Set your API key:")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        print("   or edit app.py line 20\n")
    app.run(debug=debug, host="0.0.0.0", port=port)

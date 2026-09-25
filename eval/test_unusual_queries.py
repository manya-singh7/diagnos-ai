import re

_GENERIC_MATCH_WORDS = {
    "device", "devices", "phone", "phones", "mobile", "samsung", "galaxy", "settings", "setting",
    "options", "option", "feature", "features", "screen", "screens", "component", "components",
    "item", "items", "hardware", "action", "actions", "troubleshooting", "configuration",
    "issue", "issues", "problem", "problems", "glitch", "glitches", "bug", "bugs", "handset"
}

_DEVICE_DOMAIN_WORDS = set(_GENERIC_MATCH_WORDS) | {
    # Battery & Power
    "battery", "batteries", "battry", "batery", "drain", "draining", "drained", "drains", "drainage",
    "charge", "charging", "charger", "chargers", "charged", "overheat", "overheating", "overheated",
    "heat", "heating", "hot", "warm", "warmth", "deplete", "depleting", "depleted", "depletion",
    "dying", "dead", "percentage", "percent", "power", "powers", "powered", "shutdown", "shutting",
    # Display & Screen
    "display", "displays", "touchscreen", "oled", "amoled", "lcd", "panel", "brightness", "bright",
    "dim", "dimmer", "dimming", "dimmed", "dark", "darkmode", "refresh", "hz", "tint", "tinted",
    "flicker", "flickering", "flickers", "burn-in", "pixel", "pixels", "blackout", "unresponsive",
    "touch", "touches", "touching", "tap", "tapping", "tapped", "ghost",
    # Navigation & UI Gestures
    "navigation", "nav", "swipe", "swiping", "swiped", "gesture", "gestures", "navbar", "scroll", "scrolling",
    # Camera & Imaging
    "camera", "cameras", "cam", "cams", "webcam", "selfie", "selfies", "photo", "photos", "photograph",
    "photographs", "picture", "pictures", "pic", "pics", "image", "images", "shot", "shots", "snap", "snaps",
    "flash", "torch", "lens", "lenses", "preview", "viewfinder", "blur", "blurry", "blurred", "focus",
    "focusing", "focused", "unfocused", "shutter", "zoom", "zooming", "zoomed", "exposure", "hdr",
    "video", "videos", "recording", "record", "records", "recorded", "fps",
    # Performance & System
    "lag", "lagging", "laggy", "lags", "freeze", "freezing", "freezes", "froze", "frozen", "hang",
    "hanging", "hangs", "stutter", "stuttering", "stutters", "sluggish", "slow", "slowness", "slower",
    "ram", "memory", "storage", "space", "full", "performance", "crash", "crashes", "crashing", "crashed",
    # Audio & Sound
    "audio", "sound", "sounds", "volume", "speaker", "speakers", "earpiece", "earbud", "earbuds",
    "headphone", "headphones", "headset", "headsets", "mic", "microphone", "microphones", "ringtone",
    "ringtones", "buzz", "buzzing", "static", "muffled", "silent", "mute", "muted",
    # Connectivity
    "wifi", "wi-fi", "bluetooth", "hotspot", "nfc", "cellular", "data", "network", "sim", "esim",
    "signal", "reception", "airplane", "pairing", "paired", "disconnect", "disconnected", "disconnecting",
    # OS, Apps & Security
    "reboot", "restart", "boot", "reset", "update", "updates", "updating", "updated", "app", "apps",
    "application", "applications", "firmware", "install", "installing", "uninstall", "notification",
    "notifications", "fingerprint", "biometric", "biometrics", "facial", "face", "button", "buttons",
    "key", "keys", "keyboard", "keypad", "wallpaper", "lockscreen", "security", "privacy",
    "accessibility", "sensitivity", "sensor", "sensors", "vibrate", "vibration", "vibrating", "haptic", "haptics"
}

_REAL_SETTINGS_SCREEN_PATTERNS = [
    r"\bbattery\b",
    r"\bsound\b",
    r"\bvolume\b",
    r"\bnotifications?\b",
    r"\bwi-?fi\b",
    r"\bbluetooth\b",
    r"\bnetwork\b",
    r"\bconnections?\b",
    r"\bwallpaper\b",
    r"\block\s*screen\b",
    r"\bbiometrics?\b",
    r"\bsecurity\b",
    r"\bprivacy\b",
    r"\blocations?\b",
    r"\baccounts?\b",
    r"\bapps?\b",
    r"\bdevice\s*care\b",
    r"\bstorage\b",
    r"\bmemory\b",
    r"\bbrightness\b",
    r"\bnavigation\s*bar\b",
    r"\btoggle\b",
    r"\bsensitivity\b",
    r"\baccessibility\b",
    r"\bsoftware\s*update\b",
]

def has_device_keywords(query: str) -> bool:
    t = (query or "").lower()
    words = set(re.findall(r"\b[a-z0-9'-]+\b", t))
    if words & _DEVICE_DOMAIN_WORDS:
        return True
    return any(re.search(p, t) for p in _REAL_SETTINGS_SCREEN_PATTERNS)

test_queries = [
    # Non-device queries (MUST be False)
    ("book me a flight to Paris", False),
    ("reserve a table at an Italian restaurant", False),
    ("what is the capital of France", False),
    ("tell me a funny joke about cats", False),
    ("who won the world cup in 2022", False),
    ("can you write a poem about autumn leaves", False),
    # Legitimate device queries phrased unusually (MUST be True)
    ("battery is dying super quickly after the new update", True),
    ("taking photos looks super blurry and distorted when zooming in", True),
    ("camera preview is completely black and shutter button freezes", True),
    ("my handset gets extremely hot and drains juice while idle in pocket", True),
    ("pictures keep coming out out-of-focus in low light", True),
    ("why is the rear lens foggy and refusing to focus", True),
    ("depleting percentage rapidly within two hours of light browsing", True),
    ("screen started flickering green lines after waking up", True),
    ("sound is muffled through the speakers when playing music", True),
]

if __name__ == "__main__":
    passed = 0
    for q, expected in test_queries:
        actual = has_device_keywords(q)
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] actual={actual!s:5} expected={expected!s:5} -> {q}")

    print(f"\nTotal: {passed}/{len(test_queries)} passed")

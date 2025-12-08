# app.py

import os
import re
import shutil
import subprocess
import tempfile
import uuid
import json

from flask import Flask, request, jsonify
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# ========== FLASK APP ==========
app = Flask(__name__)

# ========== ENV + MODEL LOAD ==========

# Load .env (for HF_TOKEN, DIARIZE_MODEL)
load_dotenv()


def get_token():
    return os.getenv("HF_TOKEN")


def get_model_id():
    return os.getenv("DIARIZE_MODEL") or "pyannote/speaker-diarization-3.1"


token = get_token()
if not token:
    raise RuntimeError("HF_TOKEN not found in environment variables")

model_id = get_model_id()

print("🧠 Loading Pyannote model... (first time may take a while)")
pipeline = Pipeline.from_pretrained(
    model_id,
    use_auth_token=token,
)
print("✅ Pyannote model loaded.")


# ========== FFMPEG / FFPROBE HELPERS ==========

def run_cmd(cmd, timeout=300):
    """Run a shell command and raise if it fails."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    stdout = p.stdout.decode(errors="ignore")
    stderr = p.stderr.decode(errors="ignore")

    if p.returncode != 0:
        raise RuntimeError(stderr or stdout)

    return stdout, stderr


def convert_to_wav(in_path, out_path):
    """Convert any audio to 16kHz mono 16-bit PCM WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-vn",
        "-hide_banner",
        "-loglevel", "error",
        out_path,
    ]
    run_cmd(cmd)


def ffprobe_info(path):
    """Return basic ffprobe info as a dict."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate,sample_fmt,duration",
        "-of", "default=noprint_wrappers=1",
        path,
    ]
    out, _ = run_cmd(cmd)
    info = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


# ========== DIARIZATION ==========

def run_diarization(wav_path: str):
    """Run Pyannote diarization on a WAV file and return segments list."""
    print(f"🎙️  Running diarization on {wav_path}...")
    diarization = pipeline(wav_path)
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2),
        })
    print(f"✅ Diarization complete! Found {len(segments)} segments")
    return segments


# ========== ROUTES ==========

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "msg": "server running"}), 200


@app.route("/upload", methods=["POST"])
def upload():
    """
    1. Receive audio file (field name: 'audio')
    2. Save to temp file
    3. Convert to WAV (ffmpeg)
    4. Validate with ffprobe
    5. Run diarization
    6. Save WAV into ./audio and JSON into ./diarization
    7. Return everything as JSON
    """
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "no_file_field_audio"}), 400

    f = request.files["audio"]
    if not f or f.filename == "":
        return jsonify({"ok": False, "error": "empty_filename"}), 400

    # ----- Create a safe base name from original filename -----
    original_name = os.path.splitext(f.filename)[0]
    # keep only letters, digits, _ and -, replace others with _
    safe_base = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name) or "audio"

    # Ensure project folders exist
    os.makedirs("audio", exist_ok=True)
    os.makedirs("diarization", exist_ok=True)

    # Save original upload to a temp file
    suffix = os.path.splitext(f.filename)[1] or ".bin"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        tmp_in.write(f.read())
        tmp_in.flush()
        tmp_in.close()

        # Path for converted WAV (temporary location)
        out_wav = os.path.join(
            tempfile.gettempdir(),
            f"{uuid.uuid4().hex}.wav"
        )

        # --- Convert to WAV ---
        try:
            print(f"🔄 Converting {f.filename} to WAV...")
            convert_to_wav(tmp_in.name, out_wav)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": "conversion_failed",
                "detail": str(e),
            }), 500

        # --- Inspect with ffprobe ---
        try:
            info = ffprobe_info(out_wav)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": "ffprobe_failed",
                "detail": str(e),
            }), 500

        # --- Validate audio format ---
        passes = (
            info.get("codec_name") in ("pcm_s16le", "pcm_s16", "pcm_s16be")
            and info.get("channels") == "1"
            and info.get("sample_rate") in ("16000", "16000.0")
        )

        if not passes:
            return jsonify({
                "ok": False,
                "error": "validation_failed",
                "ffprobe": info,
            }), 400

        # --- Copy final WAV into ./audio folder ---
        saved_wav_path = os.path.join("audio", f"{safe_base}.wav")
        try:
            shutil.copyfile(out_wav, saved_wav_path)
            print(f"💾 Saved WAV to {saved_wav_path}")
        except Exception as e:
            print(f"⚠️ Could not save WAV to audio/: {e}")
            saved_wav_path = None

        # --- Run diarization ---
        try:
            # use the temp wav for processing (same content as saved_wav_path)
            segments = run_diarization(out_wav)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": "diarization_failed",
                "detail": str(e),
            }), 500

        # --- Save segments JSON file into ./diarization ---
        segments_json_path = os.path.join("diarization", f"{safe_base}.json")
        try:
            with open(segments_json_path, "w", encoding="utf-8") as jf:
                json.dump(segments, jf, indent=4)
            print(f"💾 Saved diarization JSON to {segments_json_path}")
        except Exception as e:
            print(f"⚠️ Could not save diarization JSON: {e}")
            segments_json_path = None

        # --- Final response ---
        return jsonify({
            "ok": True,
            "converted_wav_temp": out_wav,      # temp path
            "saved_wav": saved_wav_path,        # ./audio/<name>.wav
            "ffprobe": info,
            "validation": {"passes": passes},
            "segments": segments,
            "segments_json": segments_json_path  # ./diarization/<name>.json
        }), 200

    finally:
        # always clean uploaded temp file
        try:
            os.unlink(tmp_in.name)
        except Exception:
            pass


# ========== MAIN ==========

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)

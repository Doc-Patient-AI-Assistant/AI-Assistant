# diarize.py
import os
import sys
import json
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# ===================== LOAD ENV VARIABLES =====================

load_dotenv()   # Loads HF_TOKEN from .env


def get_token():
    """Get HuggingFace token from environment."""
    return os.getenv("HF_TOKEN")


def get_model_id():
    """Get diarization model ID or default."""
    return os.getenv("DIARIZE_MODEL") or "pyannote/speaker-diarization-3.1"


# ===================== LOAD MODEL ONCE =====================

token = get_token()
if not token:
    raise RuntimeError("HF_TOKEN not found in environment variables")

model_id = get_model_id()

print("🧠 Loading Pyannote model... (first time takes time)")
pipeline = Pipeline.from_pretrained(
    model_id,
    use_auth_token=token
)
print("✅ Pyannote model loaded.")


# ===================== DIARIZATION FUNCTION =====================

def diarize(wav_path: str):
    """
    Run diarization on a WAV file.
    Returns a list of:
    [{ speaker, start, end }, ...]
    """
    diarization_result = pipeline(wav_path)

    segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2)
        })

    return segments


# ===================== SAVE JSON FUNCTION =====================

def save_json(segments, wav_path: str) -> str:
    """
    Save diarization result to:
    diarization/<basename>.json

    If wav_path = "audio/input1.wav" ->
    output file = "diarization/input1.json"
    """
    os.makedirs("diarization", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_file = os.path.join("diarization", f"{base_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4)

    return output_file


# ===================== CLI SUPPORT =====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing_wav_path"}))
        sys.exit(1)

    wav_path = sys.argv[1]

    if not os.path.isfile(wav_path):
        print(json.dumps({"error": f"file_not_found: {wav_path}"}))
        sys.exit(1)

    try:
        segments = diarize(wav_path)
        output_file = save_json(segments, wav_path)

        print(json.dumps({
            "message": "Diarization complete",
            "segments_file": output_file
        }, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

import json
from faster_whisper import WhisperModel

AUDIO_FILE = "../audio/input1.wav"
DIAR_FILE = "../diarization/input1.json"
OUTPUT_FILE = "../output/transcript1.json"
 
# 1. Load diarization segments (already Doctor / Patient)
with open(DIAR_FILE, "r") as f:
    diar_segments = json.load(f)

# sort just in case
diar_segments.sort(key=lambda s: s["start"])

# 2. Load Whisper model
# use base or small for better accuracy than tiny
model = WhisperModel(
    "base",         
    device="cpu",
    compute_type="int8"
)

print("Model loaded. Starting transcription...")

# 3. Transcribe full audio ONCE with word timestamps
segments, info = model.transcribe(
    AUDIO_FILE,
    word_timestamps=True,
    language="en",     # force English
    vad_filter=True
)

# collect all words with timestamps into a flat list
words = []
for seg in segments:
    if seg.words:
        for w in seg.words:
            words.append({
                "start": w.start,
                "end": w.end,
                "text": w.word  # includes the space where needed
            })

print(f"Collected {len(words)} words from ASR.")

# 4. For each diarization segment, grab the words that fall inside it
final_transcript = []

for dseg in diar_segments:
    s_start = dseg["start"]
    s_end = dseg["end"]
    speaker = dseg["speaker"]

    # words whose midpoint is inside this diarization segment
    seg_words = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        if s_start <= mid < s_end:
            seg_words.append(w["text"])

    if not seg_words:
        # no text in this diar span -> skip
        continue

    text = "".join(seg_words).strip()

    final_transcript.append({
        "speaker": speaker,
        "start": s_start,
        "end": s_end,
        "text": text
    })

# 5. Save and print
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_transcript, f, ensure_ascii=False, indent=2)

print(f"Saved transcript to {OUTPUT_FILE}\n")

for seg in final_transcript:
    print(f"{seg['speaker']}: {seg['text']}")
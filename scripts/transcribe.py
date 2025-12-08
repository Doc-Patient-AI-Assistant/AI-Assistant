# scripts/transcribe.py

import os
import sys
import json
from faster_whisper import WhisperModel


def transcribe(base_name: str):
    """
    base_name: file name without extension.
    Uses:
      audio/<base_name>.wav
      diarization/<base_name>.json
      output/<base_name>.json
    """

    AUDIO_FILE = os.path.join("audio", f"{base_name}.wav")
    DIAR_FILE = os.path.join("diarization", f"{base_name}.json")
    OUTPUT_FILE = os.path.join("output", f"transcript{base_name.lstrip('input')}.json")


    # sanity checks
    if not os.path.isfile(AUDIO_FILE):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")
    if not os.path.isfile(DIAR_FILE):
        raise FileNotFoundError(f"Diarization file not found: {DIAR_FILE}")

    os.makedirs("output", exist_ok=True)

    # 1. Load diarization segments
    with open(DIAR_FILE, "r", encoding="utf-8") as f:
        diar_segments = json.load(f)

    diar_segments.sort(key=lambda s: s["start"])

    # 2. Load Whisper model
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("✅ Model loaded. Starting transcription...")

    # 3. Transcribe full audio ONCE with word timestamps
    segments, info = model.transcribe(
        AUDIO_FILE,
        word_timestamps=True,
        language="en",
        vad_filter=True
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "text": w.word
                })

    print(f"✅ Collected {len(words)} words from ASR.")

    # 4. For each diarization segment, grab the words that fall inside it
    final_transcript = []

    for dseg in diar_segments:
        s_start = dseg["start"]
        s_end = dseg["end"]
        speaker = dseg["speaker"]

        seg_words = []
        for w in words:
            mid = (w["start"] + w["end"]) / 2.0
            if s_start <= mid < s_end:
                seg_words.append(w["text"])

        if not seg_words:
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

    print(f"✅ Saved transcript to {OUTPUT_FILE}\n")

    for seg in final_transcript:
        print(f"{seg['speaker']}: {seg['text']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/transcribe.py <base_name_without_extension>")
        sys.exit(1)

    base_name = sys.argv[1]
    transcribe(base_name)

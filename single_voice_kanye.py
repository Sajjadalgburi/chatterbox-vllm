#!/usr/bin/env python3
# single_voice_kanye.py

import os
from pathlib import Path
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# === Config ===
VOICE_PATH = Path("voices/Kanye_West.mp3")   # adjust if your file is elsewhere
OUTPUT_DIR = Path("outputs")
TEXT = (
    "A deep breath of fresh air in the morning always helps me feel more awake and ready for the day. "
    "I love the sound of birds singing, the warmth of sunlight, and the peaceful start that sets the tone for everything ahead."
)

if __name__ == "__main__":
    if not VOICE_PATH.exists():
        raise FileNotFoundError(f"Voice file not found: {VOICE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = ChatterboxTTS.from_pretrained(
        max_model_len=1000,
    )

    # Generate one audio output with the Kanye voice
    audio = model.generate(
        [TEXT],
        audio_prompt_path=str(VOICE_PATH),
        exaggeration=0.6
    )[0].detach().cpu()

    out_path = OUTPUT_DIR / "kanye_demo.wav"   # WAV is safest for saving
    ta.save(str(out_path), audio, model.sr)

    print(f"✅ Saved: {out_path}")
    model.shutdown()

#!/usr/bin/env python3
"""
Demo sử dụng Voice Activity Detection với Hugging Face Models
"""

import os
import sys
from pathlib import Path
from VAD import VoiceActivityDetector


def demo_vad():
    demo_audio = "demo_audio.wav"

    try:
        vad_silero = VoiceActivityDetector(
            model_name="silero-vad",
            threshold=0.5,
            min_speech_duration=0.5,
            device="auto",
        )

        output_dir = "vad_output_silero"
        saved_files = vad_silero.process_audio(demo_audio, output_dir)

        if saved_files:
            print(f"✅ Silero VAD: Tạo {len(saved_files)} đoạn")
        else:
            print("⚠️ Silero VAD: Không tìm thấy đoạn giọng nói")

    except Exception as e:
        print(f"❌ Lỗi Silero VAD: {e}")



def main():
    """Main function"""
    print("=" * 40)
    demo_vad()


if __name__ == "__main__":
    main()
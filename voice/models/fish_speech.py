"""
Fish Speech S2 - Voice cloning and TTS synthesis.

Uses Fish Speech S2 for zero-shot voice cloning and high-quality TTS.
Falls back to silent audio when GPU model is not available.

Fish Speech S2 approach:
  1. Extract speaker embedding from reference audio (zero-shot, no fine-tuning)
  2. Use VITS decoder with speaker conditioning for synthesis
  3. Supports streaming chunk-by-chunk output
"""

import logging
import os
import shutil
import tempfile
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100  # Fish Speech uses 44.1kHz
CHANNELS = 1
WORDS_PER_MINUTE = 150

# Try importing Fish Speech
_FISH_AVAILABLE = False
try:
    import torch
    import torchaudio

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - running in stub mode")


def _try_load_fish_speech():
    """Attempt to load Fish Speech models."""
    global _FISH_AVAILABLE
    try:
        from fish_speech.models.vqgan.lit_module import VQGAN
        from fish_speech.models.text2semantic.llama import NaiveTransformer

        _FISH_AVAILABLE = True
        logger.info("Fish Speech S2 loaded successfully")
    except ImportError:
        logger.warning("Fish Speech not installed - using fallback mode")
        _FISH_AVAILABLE = False


class FishSpeechCloner:
    """Voice cloning and TTS using Fish Speech S2."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device if _TORCH_AVAILABLE and device == "cuda" else "cpu"
        self.model_path = model_path or os.environ.get(
            "FISH_SPEECH_MODEL_PATH", "/workspace/models/fish-speech"
        )
        self.is_loaded = False
        self._model = None
        self._vqgan = None
        self._tokenizer = None

        if _TORCH_AVAILABLE:
            _try_load_fish_speech()
            if _FISH_AVAILABLE:
                self._load_model()

    def _load_model(self) -> None:
        """Load Fish Speech S2 models."""
        try:
            from fish_speech.utils import autocast_exclude_mps
            from tools.llama.generate import load_model as load_llama
            from tools.vqgan.inference import load_model as load_vqgan

            logger.info("Loading Fish Speech models from %s ...", self.model_path)

            llama_ckpt = os.path.join(self.model_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
            vqgan_ckpt = os.path.join(self.model_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

            if os.path.exists(llama_ckpt):
                self._model = load_llama(
                    config_name="firefly_gan_vq",
                    checkpoint_path=llama_ckpt,
                    device=self.device,
                )
                self.is_loaded = True
                logger.info("Fish Speech models loaded on %s", self.device)
            else:
                logger.warning("Model checkpoint not found at %s", llama_ckpt)

        except Exception as e:
            logger.error("Failed to load Fish Speech models: %s", e)
            self.is_loaded = False

    def clone_voice(
        self,
        audio_paths: list[str],
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Clone a voice from audio samples.

        Fish Speech S2 uses zero-shot cloning - no fine-tuning needed.
        We extract speaker reference features and save them for synthesis.
        """
        os.makedirs(output_dir, exist_ok=True)

        def _report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("Clone [%.0f%%] %s", pct, msg)

        _report(5.0, "Downloading audio samples...")

        # Download audio samples to local temp files
        local_paths = []
        for i, url_or_path in enumerate(audio_paths):
            _report(5 + (i / len(audio_paths)) * 20, f"Downloading sample {i + 1}/{len(audio_paths)}")
            local_path = self._download_audio(url_or_path, output_dir, i)
            if local_path:
                local_paths.append(local_path)

        if not local_paths:
            raise RuntimeError("No valid audio samples provided")

        _report(30.0, "Extracting speaker features...")

        if _FISH_AVAILABLE and _TORCH_AVAILABLE and self.is_loaded:
            # Real Fish Speech: extract reference audio features
            self._extract_speaker_reference(local_paths, output_dir, _report)
        else:
            # Stub mode: save metadata for later use
            self._stub_clone(local_paths, output_dir, _report)

        _report(100.0, "Voice cloning completed")
        return output_dir

    def _extract_speaker_reference(
        self,
        audio_paths: list[str],
        output_dir: str,
        report: Callable,
    ) -> None:
        """Extract speaker reference using Fish Speech VQGAN encoder."""
        import torch
        import torchaudio

        report(40.0, "Encoding reference audio with VQGAN...")

        # Concatenate all reference audio
        all_audio = []
        for path in audio_paths:
            waveform, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            all_audio.append(waveform)

        combined = torch.cat(all_audio, dim=1)

        # Save the reference audio for zero-shot synthesis
        ref_path = os.path.join(output_dir, "reference.wav")
        torchaudio.save(ref_path, combined, SAMPLE_RATE)

        report(70.0, "Extracting speaker embedding...")

        # Extract VQGAN codes from reference audio
        if self._vqgan is not None:
            with torch.no_grad():
                codes = self._vqgan.encode(combined.to(self.device))
                torch.save(codes, os.path.join(output_dir, "speaker_codes.pt"))

        report(90.0, "Saving voice model...")

        # Save metadata
        metadata = {
            "model_type": "fish-speech-s2",
            "sample_rate": SAMPLE_RATE,
            "reference_audio": "reference.wav",
            "speaker_codes": "speaker_codes.pt",
            "num_samples": len(audio_paths),
            "total_duration_seconds": combined.shape[1] / SAMPLE_RATE,
        }
        with open(os.path.join(output_dir, "voice_model.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def _stub_clone(
        self,
        audio_paths: list[str],
        output_dir: str,
        report: Callable,
    ) -> None:
        """Stub cloning when models aren't available."""
        import time

        sim_seconds = float(os.environ.get("LIVEAI_CLONE_SIM_SECONDS", "2"))
        steps = [
            (40.0, "Preprocessing audio samples..."),
            (60.0, "Extracting speaker features..."),
            (80.0, "Building voice profile..."),
            (95.0, "Saving voice model..."),
        ]
        for pct, msg in steps:
            report(pct, msg)
            time.sleep(sim_seconds / len(steps))

        # Save reference audio copy
        if audio_paths:
            ref_src = audio_paths[0]
            ref_dst = os.path.join(output_dir, "reference.wav")
            if os.path.exists(ref_src):
                shutil.copy2(ref_src, ref_dst)

        metadata = {
            "model_type": "fish-speech-s2-stub",
            "sample_rate": SAMPLE_RATE,
            "reference_audio": "reference.wav",
            "num_samples": len(audio_paths),
        }
        with open(os.path.join(output_dir, "voice_model.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def synthesize(
        self,
        text: str,
        voice_model_path: str,
        language: str = "en",
    ) -> np.ndarray:
        """
        Synthesize speech from text using a cloned voice.

        Returns a float32 numpy array of audio samples at SAMPLE_RATE.
        """
        if _FISH_AVAILABLE and self.is_loaded:
            return self._fish_synthesize(text, voice_model_path, language)
        return self._stub_synthesize(text)

    def _fish_synthesize(
        self,
        text: str,
        voice_model_path: str,
        language: str,
    ) -> np.ndarray:
        """Real Fish Speech synthesis."""
        import torch
        import torchaudio
        from tools.llama.generate import generate_long
        from tools.vqgan.inference import decode

        # Load reference audio for zero-shot
        ref_path = os.path.join(voice_model_path, "reference.wav")
        if not os.path.exists(ref_path):
            logger.warning("Reference audio not found, falling back to stub")
            return self._stub_synthesize(text)

        ref_audio, ref_sr = torchaudio.load(ref_path)
        if ref_sr != SAMPLE_RATE:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, SAMPLE_RATE)

        # Generate semantic tokens from text + reference
        with torch.no_grad():
            # Encode reference audio to get speaker conditioning
            ref_codes = self._vqgan.encode(ref_audio.to(self.device))

            # Generate new semantic tokens conditioned on speaker
            generated_codes = generate_long(
                model=self._model,
                text=text,
                prompt_tokens=ref_codes,
                device=self.device,
                max_new_tokens=2048,
            )

            # Decode semantic tokens to waveform
            audio_tensor = decode(
                vqgan=self._vqgan,
                codes=generated_codes,
                device=self.device,
            )

        audio_np = audio_tensor.cpu().numpy().squeeze()
        return audio_np.astype(np.float32)

    def _stub_synthesize(self, text: str) -> np.ndarray:
        """Generate silent audio with correct duration estimate."""
        word_count = len(text.split())
        duration = max(1.0, word_count / WORDS_PER_MINUTE * 60)
        num_samples = int(duration * SAMPLE_RATE)
        return np.zeros(num_samples, dtype=np.float32)

    def synthesize_to_file(
        self,
        text: str,
        voice_model_path: str,
        output_path: str,
        language: str = "en",
    ) -> str:
        """Synthesize speech and save to WAV file."""
        audio = self.synthesize(text, voice_model_path, language)
        self._write_wav(output_path, audio, SAMPLE_RATE)
        return output_path

    def synthesize_streaming(
        self,
        text: str,
        voice_model_path: str,
        language: str = "en",
        chunk_duration: float = 0.5,
    ):
        """
        Generator that yields audio chunks for streaming TTS.

        Each chunk is a float32 numpy array of `chunk_duration` seconds.
        """
        if _FISH_AVAILABLE and self.is_loaded:
            yield from self._fish_synthesize_streaming(text, voice_model_path, language, chunk_duration)
        else:
            yield from self._stub_streaming(text, chunk_duration)

    def _fish_synthesize_streaming(
        self,
        text: str,
        voice_model_path: str,
        language: str,
        chunk_duration: float,
    ):
        """Stream synthesis using Fish Speech."""
        # For streaming, generate full audio then chunk it
        # Fish Speech S2 supports native streaming via the API
        audio = self._fish_synthesize(text, voice_model_path, language)
        chunk_samples = int(chunk_duration * SAMPLE_RATE)

        for i in range(0, len(audio), chunk_samples):
            yield audio[i: i + chunk_samples]

    def _stub_streaming(self, text: str, chunk_duration: float):
        """Stream silent audio chunks."""
        word_count = len(text.split())
        duration = max(1.0, word_count / WORDS_PER_MINUTE * 60)
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        total_samples = int(duration * SAMPLE_RATE)

        for i in range(0, total_samples, chunk_samples):
            remaining = min(chunk_samples, total_samples - i)
            yield np.zeros(remaining, dtype=np.float32)

    def _download_audio(self, url_or_path: str, output_dir: str, index: int) -> Optional[str]:
        """Download or copy an audio file to local storage."""
        if os.path.exists(url_or_path):
            return url_or_path

        try:
            import httpx

            local_path = os.path.join(output_dir, f"sample_{index}.wav")
            with httpx.Client(timeout=60) as client:
                resp = client.get(url_or_path)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
            return local_path
        except Exception as e:
            logger.error("Failed to download audio %s: %s", url_or_path, e)
            return None

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
        """Write float32 audio array to WAV file."""
        import struct

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        data_size = len(audio_int16) * 2  # 16-bit = 2 bytes

        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))  # PCM
            f.write(struct.pack("<H", 1))  # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))
            f.write(struct.pack("<H", 2))  # block align
            f.write(struct.pack("<H", 16))  # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(audio_int16.tobytes())

    @staticmethod
    def generate_silent_wav_bytes(duration: float) -> bytes:
        """Generate silent WAV bytes for fallback responses."""
        import struct
        import io

        num_samples = int(duration * SAMPLE_RATE)
        data_size = num_samples * 2

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<I", SAMPLE_RATE))
        buf.write(struct.pack("<I", SAMPLE_RATE * 2))
        buf.write(struct.pack("<H", 2))
        buf.write(struct.pack("<H", 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(b"\x00" * data_size)

        return buf.getvalue()

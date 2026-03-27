"""
Expression Engine - Audio/video to FLAME expression parameter generation.

Two modes of operation:

1. Video-to-Expression: Extract FLAME params from video frames
   - Uses MediaPipe Face Landmarker for landmark detection
   - Converts 478 landmarks to 52 FLAME blendshape weights

2. Audio-to-Expression: Generate lip sync + facial expressions from audio
   - Uses a pretrained model (SadTalker/DiffPoseTalk) to predict
     FLAME expression parameters from audio features
   - Generates natural-looking blinks, head movement, expressions

Dependencies for real implementation:
  pip install mediapipe       (face landmark detection)
  pip install librosa         (audio feature extraction)
  pip install torch torchaudio

Reference models:
  - SadTalker: https://github.com/OpenTalker/SadTalker
  - DiffPoseTalk: audio-driven facial animation
  - EMOTE: https://github.com/radekd91/emote (emotional talking head)
"""

import io
import logging
import math
import os
import struct
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# FLAME expression indices mapped to ARKit blendshape names
ARKIT_BLENDSHAPE_NAMES = [
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "browDownLeft", "browDownRight",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthRollLower", "mouthRollUpper",
    "tongueOut",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
]


class ExpressionEngine:
    """
    Generates FLAME expression parameters from audio or video input.

    The audio-to-expression stub actually analyzes the audio waveform to
    produce visually plausible lip sync and facial animation.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the expression engine.

        Args:
            device: Torch device string.

        TODO (real implementation):
            import torch
            import mediapipe as mp

            self.device = torch.device(device)

            # MediaPipe face mesh for video-to-expression
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Audio-driven expression model (SadTalker or similar)
            # Load pretrained audio2expression model
            from sadtalker import Audio2Expression
            self.audio2exp = Audio2Expression(
                checkpoint_path="pretrained/audio2exp.pth"
            ).to(self.device)
            self.audio2exp.eval()

            # Audio feature extractor
            from transformers import Wav2Vec2FeatureExtractor
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
        """
        self.device = device
        logger.info("ExpressionEngine initialized (device=%s)", device)

    def audio_to_expression(
        self,
        audio_path: str,
        fps: int = 30,
    ) -> list[dict]:
        """
        Generate FLAME expression sequence from audio.

        This stub implementation actually analyzes the audio waveform to
        produce plausible lip-sync animation:
        1. Load audio with the wave module
        2. Compute RMS energy per frame (at target fps)
        3. Map energy to jaw opening
        4. Add periodic blinks every 3-5 seconds
        5. Add subtle sinusoidal head sway

        Args:
            audio_path: Path to audio file (WAV format).
            fps: Target frame rate for the expression sequence.

        Returns:
            List of dicts, one per frame, each containing:
                - expression: list[float] (52 FLAME blendshape weights)
                - jaw_pose: list[float] (3 axis-angle)
                - eye_gaze: list[float] (6 dims: 3 left + 3 right)
                - head_pose: list[float] (6 dims: 3 rotation + 3 translation)

        TODO (real implementation):
            import torchaudio

            # 1. Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # 2. Extract audio features (Wav2Vec2 or Mel spectrogram)
            features = self.audio_processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(self.device)

            # 3. Run audio-to-expression model
            with torch.no_grad():
                exp_sequence = self.audio2exp(features, num_frames=total_frames)

            # 4. Post-process: add blinks, smooth transitions
            exp_sequence = add_natural_blinks(exp_sequence)
            exp_sequence = smooth_sequence(exp_sequence, sigma=0.5)
        """
        # Load audio to get duration and waveform
        audio_data, sample_rate, duration = self._load_audio(audio_path)
        total_frames = max(1, int(duration * fps))

        logger.info(
            "Generating expressions: %.1fs audio -> %d frames at %d fps",
            duration, total_frames, fps,
        )

        # Compute RMS energy per frame
        rms_per_frame = self._compute_rms_per_frame(audio_data, sample_rate, fps, total_frames)

        # Normalize RMS to [0, 1]
        rms_max = max(rms_per_frame) if rms_per_frame else 1.0
        if rms_max > 0:
            rms_normalized = [r / rms_max for r in rms_per_frame]
        else:
            rms_normalized = [0.0] * total_frames

        # Generate blink schedule (every 3-5 seconds, ~0.15s duration)
        rng = np.random.default_rng(42)
        blink_times = []
        t = rng.uniform(2.0, 4.0)
        while t < duration:
            blink_times.append(t)
            t += rng.uniform(3.0, 5.0)

        # Build expression sequence
        frames = []
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            energy = rms_normalized[frame_idx] if frame_idx < len(rms_normalized) else 0.0

            # --- Expression blendshapes (52 dims) ---
            expression = [0.0] * 52

            # Jaw open: proportional to audio energy
            jaw_open_amount = min(1.0, energy * 1.5)
            expression[0] = jaw_open_amount  # jawOpen

            # Mouth shapes driven by energy
            expression[4] = max(0, 0.3 - jaw_open_amount * 0.5)  # mouthClose
            expression[5] = energy * 0.3   # mouthFunnel
            expression[35] = energy * 0.4  # mouthLowerDownLeft
            expression[36] = energy * 0.4  # mouthLowerDownRight
            expression[37] = energy * 0.2  # mouthUpperUpLeft
            expression[38] = energy * 0.2  # mouthUpperUpRight

            # Slight smile during speech
            expression[25] = 0.1 + energy * 0.1  # mouthSmileLeft
            expression[26] = 0.1 + energy * 0.1  # mouthSmileRight

            # Eye blinks
            blink_value = self._compute_blink(t, blink_times)
            expression[9] = blink_value   # eyeBlinkLeft
            expression[10] = blink_value  # eyeBlinkRight

            # Subtle brow movement with speech energy
            expression[17] = energy * 0.15  # browInnerUp

            # --- Jaw pose (3 axis-angle) ---
            jaw_pose = [
                jaw_open_amount * 0.15,  # jaw rotation around x (opening)
                0.0,
                0.0,
            ]

            # --- Eye gaze (6 dims) ---
            # Subtle random gaze shifts
            gaze_x = math.sin(t * 0.7) * 0.05
            gaze_y = math.sin(t * 0.4 + 1.0) * 0.03
            eye_gaze = [gaze_x, gaze_y, 0.0, gaze_x, gaze_y, 0.0]

            # --- Head pose (6 dims: rx, ry, rz, tx, ty, tz) ---
            # Subtle sinusoidal head sway
            head_rx = math.sin(t * 0.3) * 0.03       # gentle nod
            head_ry = math.sin(t * 0.2 + 0.5) * 0.04  # gentle turn
            head_rz = math.sin(t * 0.15 + 1.0) * 0.01 # slight tilt
            head_pose = [head_rx, head_ry, head_rz, 0.0, 0.0, 0.0]

            frames.append({
                "expression": expression,
                "jaw_pose": jaw_pose,
                "eye_gaze": eye_gaze,
                "head_pose": head_pose,
            })

        logger.info("Generated %d expression frames from audio", len(frames))
        return frames

    def video_to_expression(
        self,
        video_path: str,
        fps: int = 30,
    ) -> list[dict]:
        """
        Extract FLAME expression parameters from video frames.

        Args:
            video_path: Path to the source video file.
            fps: Target frame rate for extraction.

        Returns:
            List of FLAME parameter dicts, one per frame.

        TODO (real implementation):
            import cv2

            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps

            # Sample frames at target fps
            frame_interval = max(1, int(video_fps / fps))
            frames = []
            flame_params = []

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    # Detect face landmarks
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        lm_array = np.array([(l.x, l.y, l.z) for l in landmarks.landmark])
                        params = self.landmarks_to_flame(lm_array)
                        flame_params.append(params)
                    else:
                        # No face detected: use interpolated/neutral params
                        flame_params.append(self._neutral_params())
                frame_idx += 1

            cap.release()
            return flame_params
        """
        # STUB: Estimate video duration and generate matching sequence
        duration = self._estimate_video_duration(video_path)
        total_frames = max(1, int(duration * fps))

        logger.info(
            "Extracting expressions from video: %.1fs -> %d frames at %d fps",
            duration, total_frames, fps,
        )

        rng = np.random.default_rng(123)
        frames = []

        # Generate blink schedule
        blink_times = []
        t = rng.uniform(1.5, 3.0)
        while t < duration:
            blink_times.append(t)
            t += rng.uniform(2.5, 5.0)

        for frame_idx in range(total_frames):
            t = frame_idx / fps

            expression = [0.0] * 52

            # Simulate natural-looking expressions with smooth variation
            # Mouth: subtle movement
            expression[0] = max(0, math.sin(t * 3.0) * 0.15 + rng.normal(0, 0.02))
            expression[25] = 0.15 + math.sin(t * 0.5) * 0.05  # smile
            expression[26] = 0.15 + math.sin(t * 0.5) * 0.05

            # Blinks
            blink_value = self._compute_blink(t, blink_times)
            expression[9] = blink_value
            expression[10] = blink_value

            # Brows
            expression[17] = 0.05 + math.sin(t * 0.8) * 0.05

            jaw_pose = [max(0, expression[0] * 0.1), 0.0, 0.0]

            eye_gaze = [
                math.sin(t * 0.5) * 0.04,
                math.sin(t * 0.3) * 0.02,
                0.0,
                math.sin(t * 0.5) * 0.04,
                math.sin(t * 0.3) * 0.02,
                0.0,
            ]

            head_pose = [
                math.sin(t * 0.2) * 0.04,
                math.sin(t * 0.15) * 0.05,
                math.sin(t * 0.1) * 0.01,
                0.0, 0.0, 0.0,
            ]

            frames.append({
                "expression": expression,
                "jaw_pose": jaw_pose,
                "eye_gaze": eye_gaze,
                "head_pose": head_pose,
            })

        logger.info("Extracted %d expression frames from video", len(frames))
        return frames

    def landmarks_to_flame(self, landmarks: np.ndarray) -> dict:
        """
        Convert MediaPipe 478 face landmarks to FLAME expression parameters.

        Uses geometric relationships between landmark positions to estimate
        FLAME blendshape weights.

        Args:
            landmarks: (478, 3) array of MediaPipe face landmark positions,
                normalized to [0, 1].

        Returns:
            dict with expression, jaw_pose, eye_gaze, head_pose.
        """
        landmarks = np.asarray(landmarks)
        if landmarks.shape[0] != 478:
            raise ValueError(f"Expected 478 landmarks, got {landmarks.shape[0]}")

        expression = [0.0] * 52

        # Mouth opening from lip landmarks
        upper_lip = landmarks[13]  # upper lip center
        lower_lip = landmarks[14]  # lower lip center
        mouth_open = float(np.linalg.norm(upper_lip - lower_lip))
        expression[0] = min(1.0, mouth_open * 8.0)  # jawOpen

        # Mouth width
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        mouth_width = float(np.linalg.norm(left_mouth - right_mouth))

        # Smile detection
        mouth_corner_left = landmarks[61]
        mouth_corner_right = landmarks[291]
        mouth_center = (upper_lip + lower_lip) / 2
        left_up = mouth_center[1] - mouth_corner_left[1]
        right_up = mouth_center[1] - mouth_corner_right[1]
        expression[25] = max(0, min(1.0, left_up * 10.0))   # mouthSmileLeft
        expression[26] = max(0, min(1.0, right_up * 10.0))  # mouthSmileRight

        # Eye blink from eyelid landmarks
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        left_eye_open = float(np.linalg.norm(left_eye_top - left_eye_bottom))
        right_eye_open = float(np.linalg.norm(right_eye_top - right_eye_bottom))
        expression[9] = max(0, 1.0 - left_eye_open * 30.0)   # eyeBlinkLeft
        expression[10] = max(0, 1.0 - right_eye_open * 30.0)  # eyeBlinkRight

        # Brow raise from brow landmarks
        left_brow = landmarks[107]
        right_brow = landmarks[336]
        left_eye_center = (left_eye_top + left_eye_bottom) / 2
        right_eye_center = (right_eye_top + right_eye_bottom) / 2
        left_brow_raise = left_eye_center[1] - left_brow[1]
        right_brow_raise = right_eye_center[1] - right_brow[1]
        expression[17] = max(0, min(1.0, (left_brow_raise + right_brow_raise) * 5.0))

        # Jaw pose
        jaw_pose = [expression[0] * 0.15, 0.0, 0.0]

        # Head pose from face center and nose
        nose_tip = landmarks[1]
        face_center = landmarks.mean(axis=0)
        offset = nose_tip - face_center
        head_pose = [
            float(offset[1]) * 3.0,   # pitch
            float(offset[0]) * 3.0,   # yaw
            0.0,                        # roll
            0.0, 0.0, 0.0,            # translation
        ]

        # Eye gaze from iris landmarks (if available, indices 468-477)
        left_iris = landmarks[468] if landmarks.shape[0] > 468 else left_eye_center
        right_iris = landmarks[473] if landmarks.shape[0] > 473 else right_eye_center
        left_gaze_offset = left_iris - left_eye_center
        right_gaze_offset = right_iris - right_eye_center
        eye_gaze = [
            float(left_gaze_offset[0]) * 5.0,
            float(left_gaze_offset[1]) * 5.0,
            0.0,
            float(right_gaze_offset[0]) * 5.0,
            float(right_gaze_offset[1]) * 5.0,
            0.0,
        ]

        return {
            "expression": expression,
            "jaw_pose": jaw_pose,
            "eye_gaze": eye_gaze,
            "head_pose": head_pose,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: str) -> tuple:
        """
        Load audio from a WAV file using the standard library.

        Returns:
            (audio_data, sample_rate, duration_seconds)
            audio_data is a numpy float32 array normalized to [-1, 1].
        """
        if not os.path.exists(audio_path):
            # Return silence for missing files (stub mode)
            logger.warning("Audio file not found: %s (generating 3s silence)", audio_path)
            sample_rate = 22050
            duration = 3.0
            audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
            return audio_data, sample_rate, duration

        try:
            with wave.open(audio_path, "rb") as wf:
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw_data = wf.readframes(num_frames)

            duration = num_frames / sample_rate

            # Convert to numpy
            if sample_width == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            # Convert to mono if stereo
            if num_channels > 1:
                audio = audio.reshape(-1, num_channels).mean(axis=1)

            return audio, sample_rate, duration

        except Exception as e:
            logger.warning("Failed to load audio %s: %s (generating 3s silence)", audio_path, e)
            sample_rate = 22050
            duration = 3.0
            audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
            return audio_data, sample_rate, duration

    @staticmethod
    def _compute_rms_per_frame(
        audio: np.ndarray,
        sample_rate: int,
        fps: int,
        total_frames: int,
    ) -> list[float]:
        """Compute RMS energy for each video frame's time window."""
        samples_per_frame = sample_rate / fps
        rms_values = []

        for frame_idx in range(total_frames):
            start_sample = int(frame_idx * samples_per_frame)
            end_sample = int((frame_idx + 1) * samples_per_frame)
            start_sample = min(start_sample, len(audio))
            end_sample = min(end_sample, len(audio))

            if start_sample >= end_sample:
                rms_values.append(0.0)
            else:
                window = audio[start_sample:end_sample]
                rms = float(np.sqrt(np.mean(window ** 2)))
                rms_values.append(rms)

        return rms_values

    @staticmethod
    def _compute_blink(t: float, blink_times: list[float], duration: float = 0.15) -> float:
        """
        Compute blink value at time t given a list of blink onset times.

        Returns a value in [0, 1] where 1 = fully closed.
        Uses a smooth bell curve centered on each blink time.
        """
        for bt in blink_times:
            dt = abs(t - bt)
            if dt < duration:
                # Smooth bell curve: cos^2 shape
                phase = dt / duration * math.pi
                return (math.cos(phase) + 1.0) / 2.0
        return 0.0

    @staticmethod
    def _estimate_video_duration(video_path: str) -> float:
        """Estimate video duration from file (stub: returns 3.0s)."""
        if os.path.exists(video_path):
            # Very rough estimate from file size
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            # ~1MB per second for typical video
            return max(1.0, size_mb)
        return 3.0

    @staticmethod
    def _neutral_params() -> dict:
        """Return neutral (zero) FLAME parameters."""
        return {
            "expression": [0.0] * 52,
            "jaw_pose": [0.0, 0.0, 0.0],
            "eye_gaze": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "head_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

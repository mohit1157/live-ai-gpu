"""
FLAME Parametric Face Model Fitting.

Fits the FLAME (Faces Learned with an Articulated Model and Expressions)
parametric face model to 2D face images using detected landmarks.

FLAME provides:
  - Shape parameters (300 dims) - identity-specific face shape
  - Expression parameters (100 dims, we use 52 ARKit-compatible subset)
  - Jaw pose (3 dims) - jaw rotation in axis-angle
  - Eye gaze (4 dims) - left/right eye rotation
  - Neck/head pose (6 dims) - neck and global head rotation

Download FLAME model from: https://flame.is.tue.mpg.de/
Required files:
  - generic_model.pkl (FLAME 2020 model)
  - FLAME_texture.npz (UV texture space)
  - landmark_embedding.npy (mapping from FLAME mesh to 2D landmarks)

Dependencies for real implementation:
  - pip install flame-pytorch  (or clone from https://github.com/soubhirgb/FLAME_PyTorch)
  - pip install mediapipe       (for landmark detection)
  - pip install pytorch3d       (for differentiable rendering, optional)
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe face mesh landmark indices that roughly correspond to
# FLAME landmark positions. These 68 indices map the standard facial
# landmark scheme to MediaPipe's 478-point mesh.
_MP_TO_68_INDICES = [
    # Jaw contour (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    # Left eyebrow (5 points)
    107, 66, 105, 63, 70,
    # Right eyebrow (5 points)
    336, 296, 334, 293, 300,
    # Nose bridge (4 points)
    168, 6, 197, 195,
    # Nose bottom (5 points)
    5, 4, 1, 2, 98,
    # Left eye (6 points)
    33, 160, 158, 133, 153, 144,
    # Right eye (6 points)
    362, 385, 387, 263, 373, 380,
    # Outer lip (12 points)
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91,
    # Inner lip (8 points)
    78, 82, 87, 14, 317, 312, 308, 402,
]


class FLAMEFitter:
    """
    Fits FLAME parametric face model to 2D face images.

    Uses MediaPipe Face Landmarker for 478-point landmark detection,
    then optimizes FLAME parameters to match detected landmarks.

    This is a STUB implementation. Each method documents what the real
    implementation requires.
    """

    def __init__(self, flame_model_path: Optional[str] = None):
        """
        Initialize the FLAME fitter.

        Args:
            flame_model_path: Path to the FLAME model directory containing
                generic_model.pkl and landmark_embedding.npy.

        TODO (real implementation):
            import torch
            from flame_pytorch import FLAME, FLAMEConfig

            config = FLAMEConfig(
                flame_model_path=os.path.join(flame_model_path, "generic_model.pkl"),
                n_shape=300,
                n_exp=100,
                use_face_contour=True,
            )
            self.flame = FLAME(config).to("cuda")
            self.landmark_embedding = np.load(
                os.path.join(flame_model_path, "landmark_embedding.npy")
            )

            # MediaPipe face landmarker
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        """
        self.flame_model_path = flame_model_path
        self._initialized = True
        logger.info("FLAMEFitter initialized (flame_model=%s)", flame_model_path or "stub")

    def fit_single(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Fit FLAME parameters to a single face image.

        Args:
            image: BGR image as numpy array (H, W, 3).
            landmarks: Optional pre-detected MediaPipe 478 landmarks (478, 3).
                If None, landmarks are detected from the image.

        Returns:
            dict with FLAME parameters:
                - shape: (300,) identity shape coefficients
                - expression: (52,) expression blendshape weights (ARKit-compatible)
                - jaw_pose: (3,) jaw rotation in axis-angle
                - eye_gaze: (4,) left/right eye gaze rotation
                - neck_pose: (3,) neck rotation
                - head_pose: (3,) global head rotation
                - landmarks_2d: (68, 2) projected 2D landmarks
                - confidence: float, fitting quality score

        TODO (real implementation):
            # 1. Detect landmarks with MediaPipe
            if landmarks is None:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    raise ValueError("No face detected in image")
                lm = results.multi_face_landmarks[0]
                landmarks = np.array([(l.x, l.y, l.z) for l in lm.landmark])

            # 2. Extract 68-point subset for FLAME fitting
            landmarks_68 = landmarks[_MP_TO_68_INDICES]

            # 3. Initialize FLAME parameters
            shape = torch.zeros(1, 300, device="cuda", requires_grad=True)
            expression = torch.zeros(1, 100, device="cuda", requires_grad=True)
            jaw_pose = torch.zeros(1, 3, device="cuda", requires_grad=True)
            ...

            # 4. Optimize with landmark reprojection loss
            optimizer = torch.optim.Adam([shape, expression, jaw_pose, ...], lr=0.01)
            for step in range(200):
                vertices, landmarks_3d = self.flame(
                    shape_params=shape,
                    expression_params=expression,
                    jaw_pose=jaw_pose,
                    ...
                )
                # Project 3D landmarks to 2D
                projected = project(landmarks_3d, camera)
                # L2 loss on landmark positions
                loss = F.mse_loss(projected, target_landmarks)
                # Regularization on shape and expression
                loss += 0.001 * torch.sum(shape ** 2)
                loss += 0.0001 * torch.sum(expression ** 2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 5. Map 100-dim FLAME expressions to 52-dim ARKit blendshapes
            arkit_52 = flame_to_arkit_mapping(expression)
        """
        h, w = image.shape[:2] if image is not None and image.ndim >= 2 else (512, 512)

        # STUB: Generate plausible mock FLAME parameters
        rng = np.random.default_rng()

        # Shape: mostly zeros for a generic face, small random perturbations
        shape = (rng.standard_normal(300) * 0.5).tolist()

        # Expression: 52 ARKit-compatible blendshapes, mostly near zero
        expression = (rng.standard_normal(52) * 0.05).tolist()

        # Jaw: small opening
        jaw_pose = (rng.standard_normal(3) * 0.02).tolist()

        # Eyes: near center gaze
        eye_gaze = (rng.standard_normal(4) * 0.02).tolist()

        # Neck and head: subtle pose
        neck_pose = (rng.standard_normal(3) * 0.02).tolist()
        head_pose = (rng.standard_normal(3) * 0.03).tolist()

        # Mock 2D landmarks (68 points distributed on a face template)
        cx, cy = w / 2, h / 2
        landmarks_2d = []
        for i in range(68):
            angle = (i / 68) * 2 * math.pi
            r = min(w, h) * 0.3
            lx = cx + r * math.cos(angle) * (0.8 + rng.uniform(-0.1, 0.1))
            ly = cy + r * math.sin(angle) * (1.0 + rng.uniform(-0.1, 0.1))
            landmarks_2d.append([lx, ly])

        return {
            "shape": shape,
            "expression": expression,
            "jaw_pose": jaw_pose,
            "eye_gaze": eye_gaze,
            "neck_pose": neck_pose,
            "head_pose": head_pose,
            "landmarks_2d": landmarks_2d,
            "confidence": 0.92 + rng.uniform(-0.05, 0.05),
        }

    def fit_sequence(
        self,
        images: list,
        landmarks: Optional[list] = None,
    ) -> list[dict]:
        """
        Fit FLAME parameters to a video sequence.

        Uses temporal smoothing to ensure consistent identity (shape) across
        frames while allowing expression to vary.

        Args:
            images: List of BGR images as numpy arrays.
            landmarks: Optional list of pre-detected landmark arrays.

        Returns:
            List of FLAME parameter dicts (one per frame).

        TODO (real implementation):
            # 1. Fit first frame to get identity shape
            first_fit = self.fit_single(images[0], landmarks[0] if landmarks else None)
            identity_shape = first_fit["shape"]

            # 2. Fit remaining frames with fixed shape
            results = [first_fit]
            for i in range(1, len(images)):
                lm = landmarks[i] if landmarks else None
                params = self._fit_with_fixed_shape(images[i], identity_shape, lm)
                results.append(params)

            # 3. Temporal smoothing (Gaussian filter on expression/pose)
            results = self._smooth_sequence(results, sigma=1.5)
        """
        results = []

        # Get consistent identity shape from first frame
        first_result = self.fit_single(
            images[0] if images else np.zeros((512, 512, 3), dtype=np.uint8),
            landmarks[0] if landmarks else None,
        )
        identity_shape = first_result["shape"]
        results.append(first_result)

        # Fit remaining frames with the same identity
        for i in range(1, len(images) if images else 0):
            lm = landmarks[i] if landmarks and i < len(landmarks) else None
            params = self.fit_single(images[i], lm)
            params["shape"] = identity_shape  # lock identity
            results.append(params)

        logger.info("Fitted FLAME params to %d frames", len(results))
        return results

    def landmarks_to_flame(self, landmarks_478: np.ndarray) -> dict:
        """
        Convert MediaPipe 478 face landmarks to FLAME parameters.

        This is a fast approximate mapping that avoids full optimization.
        Uses a learned linear regression from landmarks to FLAME params.

        Args:
            landmarks_478: (478, 3) MediaPipe face landmark positions,
                normalized to [0, 1] range.

        Returns:
            dict with FLAME parameters (same format as fit_single).

        TODO (real implementation):
            # 1. Extract the 68-point subset
            lm68 = landmarks_478[_MP_TO_68_INDICES]

            # 2. Normalize landmarks (centering, scaling)
            lm_centered = lm68 - lm68.mean(axis=0)
            lm_norm = lm_centered / np.linalg.norm(lm_centered)

            # 3. Apply pre-trained linear regression
            # This matrix is trained offline:
            #   For N face images:
            #     landmarks_i = detect(image_i)
            #     flame_i = full_optimization(image_i)
            #   regression_matrix = lstsq(landmarks_matrix, flame_matrix)
            flame_vec = self.regression_matrix @ lm_norm.flatten()

            # 4. Unpack into FLAME parameter groups
            shape = flame_vec[:300]
            expression = flame_vec[300:400]
            ...
        """
        landmarks_478 = np.asarray(landmarks_478)
        if landmarks_478.shape[0] != 478:
            raise ValueError(f"Expected 478 landmarks, got {landmarks_478.shape[0]}")

        # STUB: Approximate linear mapping
        # Extract the 68-point subset
        lm68 = landmarks_478[_MP_TO_68_INDICES[:68]] if landmarks_478.shape[0] >= max(_MP_TO_68_INDICES[:68]) + 1 else landmarks_478[:68]

        # Compute some basic features from landmarks
        # Mouth opening: distance between upper and lower lip
        upper_lip_idx, lower_lip_idx = 13, 14  # approximate indices
        if landmarks_478.shape[0] > max(upper_lip_idx, lower_lip_idx):
            mouth_open = float(np.linalg.norm(
                landmarks_478[upper_lip_idx] - landmarks_478[lower_lip_idx]
            ))
        else:
            mouth_open = 0.0

        # Eye openness
        left_eye_top, left_eye_bottom = 159, 145
        right_eye_top, right_eye_bottom = 386, 374
        left_eye_open = float(np.linalg.norm(
            landmarks_478[min(left_eye_top, landmarks_478.shape[0]-1)]
            - landmarks_478[min(left_eye_bottom, landmarks_478.shape[0]-1)]
        )) if landmarks_478.shape[0] > max(left_eye_top, left_eye_bottom) else 0.02
        right_eye_open = float(np.linalg.norm(
            landmarks_478[min(right_eye_top, landmarks_478.shape[0]-1)]
            - landmarks_478[min(right_eye_bottom, landmarks_478.shape[0]-1)]
        )) if landmarks_478.shape[0] > max(right_eye_top, right_eye_bottom) else 0.02

        # Head pose from landmark positions (approximate)
        nose_tip = landmarks_478[1] if landmarks_478.shape[0] > 1 else np.zeros(3)
        face_center = landmarks_478.mean(axis=0)

        # Build FLAME parameters from features
        expression = np.zeros(52)
        expression[0] = mouth_open * 5.0    # jawOpen
        expression[1] = max(0, mouth_open * 3.0 - 0.1)  # mouthClose
        expression[9] = 1.0 - min(1.0, left_eye_open * 20)   # eyeBlinkLeft
        expression[10] = 1.0 - min(1.0, right_eye_open * 20)  # eyeBlinkRight

        jaw_pose = np.array([mouth_open * 2.0, 0.0, 0.0])

        # Head rotation from nose offset
        head_offset = nose_tip - face_center
        head_pose = np.array([
            head_offset[1] * 2.0,  # pitch
            head_offset[0] * 2.0,  # yaw
            0.0,                    # roll
        ])

        return {
            "shape": np.zeros(300).tolist(),
            "expression": expression.tolist(),
            "jaw_pose": jaw_pose.tolist(),
            "eye_gaze": [0.0, 0.0, 0.0, 0.0],
            "neck_pose": [0.0, 0.0, 0.0],
            "head_pose": head_pose.tolist(),
            "confidence": 0.75,  # lower confidence for linear approximation
        }

"""Standalone observation builder for policy deployment.

Extracted from hil/observation.py (HILObservationMixin) and hil/loop.py
(_convert_action_to_dict) into a class with explicit dependencies instead
of mixin self.* lookups.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """Builds policy-format observations from raw robot data.

    Handles:
    - Loading training state names from checkpoint metadata
    - Loading normalization statistics (safetensors or dataset stats.json)
    - Transforming raw observations to policy-expected tensor format
    - Converting policy action tensors back to named motor dicts
    """

    def __init__(
        self,
        checkpoint_path: Path,
        policy,
        policy_type: str = "",
        task: str = "",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.policy = policy
        self.policy_type = policy_type
        self.task = task or "Do the task"

        # Caches (cleared via reset_cache)
        self._dataset_info: Optional[dict] = None
        self._dataset_info_loaded: bool = False
        self._training_state_names: Optional[List[str]] = None
        self._training_state_names_loaded = False
        self._training_action_names: Optional[List[str]] = None
        self._training_action_names_loaded = False
        self._norm_stats: Optional[dict] = None
        self._norm_stats_loaded = False
        self._pi05_tokenizer = None

        # LeRobot preprocessor pipeline (handles all normalization modes)
        self._preprocessor = None
        self._postprocessor = None
        self._inference_frame_count = 0

        try:
            from lerobot.policies.factory import make_pre_post_processors

            self._preprocessor, self._postprocessor = make_pre_post_processors(
                policy.config, pretrained_path=str(checkpoint_path)
            )
            logger.info("Using LeRobot preprocessor pipeline for normalization")
        except Exception as e:
            logger.info(
                "Preprocessor pipeline not available (%s), using manual normalization", e
            )
            self._preprocessor = None
            self._postprocessor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_training_state_names(self) -> Optional[List[str]]:
        """Get observation state names from the policy's training dataset.

        Returns:
            List of state names like ['left_base.pos', ...] or None.
        """
        if self._training_state_names_loaded:
            return self._training_state_names

        self._training_state_names_loaded = True

        info = self._load_dataset_info()
        if info is None:
            return None

        state_names = info.get("features", {}).get("observation.state", {}).get("names")
        if state_names:
            logger.info(
                "Loaded %d state names from training dataset: %s",
                len(state_names),
                state_names,
            )

        self._training_state_names = state_names
        return state_names

    def get_training_action_names(self) -> Optional[List[str]]:
        """Get action names from the policy's training dataset.

        Action names correspond to the policy's output dimensions (e.g. 7
        position targets), which may differ from observation state names
        (e.g. 21 = pos + vel + tau) for extended-state policies.

        Falls back to get_training_state_names() for older datasets that
        don't store action names separately.

        Returns:
            List of action names like ['left_base.pos', ...] or None.
        """
        if self._training_action_names_loaded:
            return self._training_action_names

        self._training_action_names_loaded = True

        info = self._load_dataset_info()
        if info is not None:
            action_names = info.get("features", {}).get("action", {}).get("names")
            if action_names:
                logger.info(
                    "Loaded %d action names from training dataset: %s",
                    len(action_names),
                    action_names,
                )
                self._training_action_names = action_names
                return action_names

        # Backward compat: older datasets without separate action names
        fallback = self.get_training_state_names()
        if fallback:
            has_extended = any(n.endswith((".vel", ".tau")) for n in fallback)
            if has_extended:
                all_count = len(fallback)
                fallback = [n for n in fallback if n.endswith(".pos")]
                logger.warning(
                    "DEPLOY: Action names derived from state names (backward compat). "
                    "If this dataset was recorded with the split-cache fix, action names "
                    "should be stored separately in info.json."
                )
                logger.info(
                    "Filtered %d extended state names to %d position-only",
                    all_count,
                    len(fallback),
                )
            else:
                logger.info(
                    "Action names not found in dataset, falling back to state names"
                )
        self._training_action_names = fallback
        return fallback

    def load_normalization_stats(self) -> Optional[dict]:
        """Load normalization statistics from checkpoint or training dataset.

        Tries the checkpoint safetensors file first, then falls back to
        the training dataset's meta/stats.json (needed for Pi0.5).

        Returns:
            Dict with keys like 'action.min', 'action.max',
            'observation.state.min', 'observation.state.max' as tensors,
            or None if not found.
        """
        if self._norm_stats_loaded:
            return self._norm_stats

        self._norm_stats_loaded = True

        import torch

        # Try checkpoint safetensors first
        stats_path = (
            self.checkpoint_path
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        if stats_path.exists():
            try:
                import safetensors.torch as st

                stats = st.load_file(str(stats_path))
                logger.info("Loaded normalization stats from checkpoint (%d keys)", len(stats))
                self._log_norm_mode(stats)
                self._norm_stats = stats
                return stats
            except Exception as e:
                logger.warning("Failed to load safetensors stats: %s", e)

        # Fallback: training dataset stats.json
        logger.debug("Checkpoint stats not found, trying training dataset...")

        metadata_path = self.checkpoint_path / "policy_metadata.json"
        if not metadata_path.exists():
            for parent_level in range(1, 4):
                parent = self.checkpoint_path
                for _ in range(parent_level):
                    parent = parent.parent
                test_path = parent / "policy_metadata.json"
                if test_path.exists():
                    metadata_path = test_path
                    break

        if not metadata_path.exists():
            logger.warning("Policy metadata not found at: %s", metadata_path)
            self._norm_stats = None
            return None

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            dataset_repo_id = metadata.get("dataset_repo_id")
            use_quantile = metadata.get("config", {}).get(
                "use_quantile_normalization", True
            )

            if not dataset_repo_id:
                logger.warning("No dataset_repo_id in policy metadata")
                self._norm_stats = None
                return None

            # Find project root to locate datasets
            project_root = self.checkpoint_path
            while project_root.name != "nextis_app" and project_root.parent != project_root:
                project_root = project_root.parent

            stats_json_path = project_root / "datasets" / dataset_repo_id / "meta" / "stats.json"
            if not stats_json_path.exists():
                logger.warning("Training dataset stats not found: %s", stats_json_path)
                self._norm_stats = None
                return None

            with open(stats_json_path) as f:
                dataset_stats = json.load(f)

            stats: Dict = {}

            if "action" in dataset_stats:
                action_stats = dataset_stats["action"]
                if use_quantile and "q01" in action_stats and "q99" in action_stats:
                    stats["action.min"] = torch.tensor(action_stats["q01"], dtype=torch.float32)
                    stats["action.max"] = torch.tensor(action_stats["q99"], dtype=torch.float32)
                    logger.info("Loaded action stats (quantiles q01/q99)")
                elif "min" in action_stats and "max" in action_stats:
                    stats["action.min"] = torch.tensor(action_stats["min"], dtype=torch.float32)
                    stats["action.max"] = torch.tensor(action_stats["max"], dtype=torch.float32)
                    logger.info("Loaded action stats (min/max)")

            if "observation.state" in dataset_stats:
                state_stats = dataset_stats["observation.state"]
                if use_quantile and "q01" in state_stats and "q99" in state_stats:
                    stats["observation.state.min"] = torch.tensor(
                        state_stats["q01"], dtype=torch.float32
                    )
                    stats["observation.state.max"] = torch.tensor(
                        state_stats["q99"], dtype=torch.float32
                    )
                elif "min" in state_stats and "max" in state_stats:
                    stats["observation.state.min"] = torch.tensor(
                        state_stats["min"], dtype=torch.float32
                    )
                    stats["observation.state.max"] = torch.tensor(
                        state_stats["max"], dtype=torch.float32
                    )

            if stats:
                self._log_norm_mode(stats)
            self._norm_stats = stats if stats else None
            return self._norm_stats

        except Exception as e:
            logger.warning("Failed to load stats from training dataset: %s", e)
            self._norm_stats = None
            return None

    def prepare_observation(self, raw_obs: dict) -> dict:
        """Transform raw robot observation to policy-expected format.

        Robot returns: ``{'camera_1': np.array(H,W,C), 'left_base.pos': 0.5, ...}``
        Policy expects: ``{'observation.images.camera_1': Tensor(1,C,H,W),
                          'observation.state': Tensor(1,N)}``

        Applies normalization via LeRobot preprocessor pipeline if available,
        otherwise falls back to manual MEAN_STD or MIN_MAX normalization.
        """
        import torch

        device = self.policy.config.device

        if self._preprocessor is not None:
            policy_obs = self._prepare_with_preprocessor(raw_obs)
        else:
            policy_obs = self._prepare_manual(raw_obs, device)

        # Diagnostic logging for first 3 frames (WARNING level to ensure visibility)
        if self._inference_frame_count < 3:
            if "observation.state" in policy_obs:
                s = policy_obs["observation.state"]
                logger.warning(
                    "DEPLOY [frame %d] state tensor stats: min=%.4f max=%.4f mean=%.4f dim=%d",
                    self._inference_frame_count,
                    s.min().item(), s.max().item(), s.mean().item(),
                    s.shape[-1],
                )
            has_images = sum(
                1 for k in policy_obs if k.startswith("observation.images.")
            )
            if self._inference_frame_count == 0:
                logger.warning(
                    "DEPLOY [frame 0] observation keys: %s (%d image features)",
                    [k for k in sorted(policy_obs.keys())],
                    has_images,
                )
        self._inference_frame_count += 1

        # Pi0.5 language tokenization
        if self.policy_type == "pi05" and "observation.state" in policy_obs:
            self._add_pi05_tokenization(policy_obs, device)

        return policy_obs

    def _prepare_with_preprocessor(self, raw_obs: dict) -> dict:
        """Build raw tensors and normalize via LeRobot preprocessor pipeline."""
        import torch

        raw_batch: dict = {}

        # Camera images: uint8 → float32 [0,1] (preprocessor handles normalization)
        if hasattr(self.policy.config, "image_features") and self.policy.config.image_features:
            for key in self.policy.config.image_features:
                cam_name = key.split(".")[-1]
                if cam_name in raw_obs:
                    img = raw_obs[cam_name]
                    img_tensor = (
                        torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                    )
                    raw_batch[key] = img_tensor

        # Observation state: raw float values (preprocessor handles normalization)
        if hasattr(self.policy.config, "robot_state_feature") and self.policy.config.robot_state_feature:
            state_names = self.get_training_state_names()
            if state_names:
                state_values = [float(raw_obs.get(name, 0.0)) for name in state_names]
                self._warn_missing_states(state_names, raw_obs)
                raw_batch["observation.state"] = torch.tensor(
                    state_values, dtype=torch.float32
                )

        # Preprocessor handles: batch dim, device placement, normalization
        return self._preprocessor(raw_batch)

    def _prepare_manual(self, raw_obs: dict, device) -> dict:
        """Build tensors with manual normalization (MEAN_STD or MIN_MAX fallback)."""
        import torch

        policy_obs: dict = {}
        norm_stats = self.load_normalization_stats()

        # 1. Camera images
        if hasattr(self.policy.config, "image_features") and self.policy.config.image_features:
            for key in self.policy.config.image_features:
                cam_name = key.split(".")[-1]
                if cam_name in raw_obs:
                    img = raw_obs[cam_name]
                    img_tensor = (
                        torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                    )
                    if norm_stats:
                        mean_key = f"{key}.mean"
                        std_key = f"{key}.std"
                        if mean_key in norm_stats and std_key in norm_stats:
                            mean = norm_stats[mean_key].view(3, 1, 1)
                            std = norm_stats[std_key].view(3, 1, 1)
                            img_tensor = (img_tensor - mean) / (std + 1e-8)
                    policy_obs[key] = img_tensor.unsqueeze(0).to(device)

        # 2. Observation state
        if hasattr(self.policy.config, "robot_state_feature") and self.policy.config.robot_state_feature:
            state_names = self.get_training_state_names()
            if state_names:
                state_values = [float(raw_obs.get(name, 0.0)) for name in state_names]
                self._warn_missing_states(state_names, raw_obs)
                state_tensor = torch.tensor(state_values, dtype=torch.float32)

                if norm_stats:
                    if (
                        "observation.state.mean" in norm_stats
                        and "observation.state.std" in norm_stats
                    ):
                        mean = norm_stats["observation.state.mean"]
                        std = norm_stats["observation.state.std"]
                        state_tensor = (state_tensor - mean) / (std + 1e-8)
                    elif (
                        "observation.state.min" in norm_stats
                        and "observation.state.max" in norm_stats
                    ):
                        state_min = norm_stats["observation.state.min"]
                        state_max = norm_stats["observation.state.max"]
                        state_range = state_max - state_min
                        dead_motors = state_range.abs() < 1e-6
                        safe_range = torch.where(
                            dead_motors, torch.ones_like(state_range), state_range
                        )
                        state_tensor = (state_tensor - state_min) / safe_range
                        state_tensor = torch.where(
                            dead_motors,
                            torch.full_like(state_tensor, 0.5),
                            state_tensor,
                        )
                        state_tensor = state_tensor * 2.0 - 1.0
                        state_tensor = torch.clamp(state_tensor, -1.0, 1.0)

                policy_obs["observation.state"] = state_tensor.unsqueeze(0).to(device)

        return policy_obs

    def _warn_missing_states(self, state_names: list, raw_obs: dict) -> None:
        """One-time warning if training state names are missing from raw observation."""
        if hasattr(self, "_warned_missing"):
            return
        self._warned_missing = True

        missing = [n for n in state_names if n not in raw_obs]
        if missing:
            logger.warning(
                "OBSERVATION MISMATCH: %d/%d state names missing from robot observation. "
                "Missing: %s. These default to 0.0 — policy input will be degraded! "
                "If the policy was trained with extended state (vel/tau), ensure "
                "record_extended_state=True on the follower robot.",
                len(missing),
                len(state_names),
                missing[:8],
            )
        else:
            logger.warning(
                "DEPLOY [frame 0] observation: all %d states present. "
                "Raw values: %s",
                len(state_names),
                {n: f"{float(raw_obs.get(n, 0.0)):.4f}" for n in state_names},
            )

    def convert_action_to_dict(
        self,
        action,
        raw_obs: dict,
        movement_scale: float = 1.0,
    ) -> dict:
        """Convert policy action tensor to named motor dict.

        1. Handles multi-step diffusion output (3D/2D → 1D)
        2. Denormalizes using postprocessor pipeline or manual MEAN_STD/MIN_MAX
        3. Handles dead motors (min==max → midpoint) in MIN_MAX fallback
        4. Returns dict with named keys from training dataset

        Args:
            action: Policy output tensor or numpy array.
            raw_obs: Raw robot observation (used by callers, not by this method).
            movement_scale: Deprecated, ignored. Velocity limiting is handled
                by SafetyPipeline._limit_velocity() which uses proper delta
                clamping (max_vel * dt) instead of this former asymptotic formula.

        Returns:
            Dict mapping motor names to position values, or empty dict on failure.
        """
        import torch

        if isinstance(action, dict):
            return action

        action_names = self.get_training_action_names()
        if action_names is None:
            logger.warning("No action names available — cannot convert tensor to dict")
            return {}

        if self._postprocessor is not None:
            action_np = self._denormalize_with_postprocessor(action)
        else:
            action_np = self._denormalize_manual(action)

        if action_np is None:
            return {}

        # Handle multi-step output (3D/2D → 1D)
        if action_np.ndim == 3:
            action_np = action_np[0, 0]
        elif action_np.ndim == 2:
            action_np = action_np[0]
        elif action_np.ndim > 3:
            action_np = action_np.squeeze()

        if len(action_np) != len(action_names):
            logger.warning(
                "Action dimension mismatch: action has %d elements, names has %d",
                len(action_np),
                len(action_names),
            )
            return {}

        # Diagnostic logging for first 3 frames (WARNING level for visibility)
        if self._inference_frame_count <= 3:
            logger.warning(
                "DEPLOY [frame %d] denormalized action: min=%.4f max=%.4f mean=%.4f values=%s",
                self._inference_frame_count - 1,
                action_np.min(),
                action_np.max(),
                action_np.mean(),
                {name: f"{float(action_np[i]):.4f}" for i, name in enumerate(action_names)},
            )

        return {name: float(action_np[i]) for i, name in enumerate(action_names)}

    def _denormalize_with_postprocessor(self, action) -> Optional[np.ndarray]:
        """Denormalize action via LeRobot postprocessor pipeline."""
        import torch

        if isinstance(action, torch.Tensor):
            action_tensor = action
        else:
            action_tensor = torch.tensor(action, dtype=torch.float32)

        try:
            # Postprocessor expects a bare tensor (PolicyAction = torch.Tensor),
            # not a dict. Its to_transition converter wraps it into EnvTransition,
            # the unnormalizer denormalizes, and to_output extracts the tensor back.
            result = self._postprocessor(action_tensor)
            if isinstance(result, torch.Tensor):
                return result.cpu().numpy()
            # Fallback: if result is a dict (older LeRobot versions)
            return result["action"].cpu().numpy()
        except Exception as e:
            logger.warning("Postprocessor failed (%s), falling back to manual", e)
            return self._denormalize_manual(action)

    def _denormalize_manual(self, action) -> Optional[np.ndarray]:
        """Denormalize action using manual MEAN_STD or MIN_MAX."""
        import torch

        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)

        norm_stats = self.load_normalization_stats()
        if norm_stats:
            if "action.mean" in norm_stats and "action.std" in norm_stats:
                mean = norm_stats["action.mean"].cpu().numpy()
                std = norm_stats["action.std"].cpu().numpy()
                action_np = action_np * std + mean
            elif "action.min" in norm_stats and "action.max" in norm_stats:
                action_min = norm_stats["action.min"].cpu().numpy()
                action_max = norm_stats["action.max"].cpu().numpy()
                action_range = action_max - action_min
                dead_motors = np.abs(action_range) < 1e-6
                safe_range = np.where(dead_motors, 1.0, action_range)
                action_mid = (action_max + action_min) / 2.0
                action_np = (action_np + 1.0) / 2.0 * safe_range + action_min
                action_np = np.where(dead_motors, action_mid, action_np)

        return action_np

    def reset_cache(self) -> None:
        """Clear cached state names and normalization stats."""
        self._dataset_info = None
        self._dataset_info_loaded = False
        self._training_state_names = None
        self._training_state_names_loaded = False
        self._training_action_names = None
        self._training_action_names_loaded = False
        self._norm_stats = None
        self._norm_stats_loaded = False
        self._inference_frame_count = 0
        if hasattr(self, '_warned_missing'):
            del self._warned_missing

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _log_norm_mode(stats: dict) -> None:
        """Log which normalization mode the loaded stats support."""
        has_mean_std = any(k.endswith(".mean") for k in stats)
        has_min_max = any(k.endswith(".min") for k in stats)
        if has_mean_std:
            logger.info("Norm stats contain MEAN_STD keys")
        if has_min_max:
            logger.info("Norm stats contain MIN_MAX keys")
        if not has_mean_std and not has_min_max:
            logger.warning(
                "Norm stats contain NEITHER mean/std NOR min/max — "
                "normalization will be skipped!"
            )

    def _load_dataset_info(self) -> Optional[dict]:
        """Load and cache the training dataset's meta/info.json.

        Resolves: checkpoint_path/train_config.json → dataset.root → meta/info.json.
        Called by both get_training_state_names() and get_training_action_names()
        so the file is only read once.
        """
        if self._dataset_info_loaded:
            return self._dataset_info

        self._dataset_info_loaded = True

        train_config_path = self.checkpoint_path / "train_config.json"
        if not train_config_path.exists():
            logger.warning("train_config.json not found at %s", train_config_path)
            return None

        with open(train_config_path) as f:
            train_config = json.load(f)

        dataset_root = train_config.get("dataset", {}).get("root")
        if not dataset_root:
            logger.warning("dataset.root not found in train_config")
            return None

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            logger.warning("Training dataset info.json not found: %s", info_path)
            return None

        with open(info_path) as f:
            info = json.load(f)

        self._dataset_info = info
        return info

    def _add_pi05_tokenization(self, policy_obs: dict, device) -> None:
        """Add Pi0.5 language tokenization to the observation."""
        import torch

        state_tensor = policy_obs["observation.state"].squeeze(0).cpu().numpy()
        max_state_dim = getattr(self.policy.config, "max_state_dim", 32)
        if len(state_tensor) < max_state_dim:
            padded = np.zeros(max_state_dim, dtype=np.float32)
            padded[: len(state_tensor)] = state_tensor
            state_tensor = padded

        discretized = (
            np.digitize(state_tensor, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        )
        discretized = np.clip(discretized, 0, 255)

        state_str = " ".join(map(str, discretized))
        cleaned_task = self.task.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned_task}, State: {state_str};\nAction: "

        if self._pi05_tokenizer is None:
            from transformers import AutoTokenizer

            self._pi05_tokenizer = AutoTokenizer.from_pretrained(
                "google/paligemma-3b-pt-224"
            )
            self._pi05_tokenizer.padding_side = "right"
            logger.info("Pi0.5 tokenizer loaded")

        max_length = getattr(self.policy.config, "tokenizer_max_length", 200)
        tokenized = self._pi05_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        policy_obs["observation.language.tokens"] = tokenized["input_ids"].to(device)
        policy_obs["observation.language.attention_mask"] = (
            tokenized["attention_mask"].bool().to(device)
        )

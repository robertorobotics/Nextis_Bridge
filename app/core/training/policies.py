import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .types import PolicyConfig, PolicyInfo

logger = logging.getLogger(__name__)


class PolicyMixin:
    """Mixin providing policy management methods for TrainingService."""

    def list_policies(self) -> List[PolicyInfo]:
        """Scan training/outputs/ directory and return all trained policies."""
        policies = []

        if not self.outputs_path.exists():
            return policies

        for output_dir in self.outputs_path.iterdir():
            if not output_dir.is_dir():
                continue

            policy = self._parse_policy_directory(output_dir)
            if policy:
                policies.append(policy)

        # Also include currently training job
        if self.active_job and self.active_job.output_dir:
            # Check if not already in list
            active_id = self.active_job.output_dir.name
            if not any(p.id == active_id for p in policies):
                policies.append(PolicyInfo(
                    id=active_id,
                    name=self.active_job.config.get("policy_name", active_id),
                    policy_type=self.active_job.policy_type.value,
                    status="training",
                    steps=self.active_job.progress.step,
                    total_steps=self.active_job.progress.total_steps,
                    dataset_repo_id=self.active_job.dataset_repo_id,
                    created_at=self.active_job.created_at.isoformat(),
                    final_loss=self.active_job.progress.loss,
                    checkpoint_path="",
                    loss_history=self.active_job.progress.loss_history,
                    output_dir=str(self.active_job.output_dir),
                ))

        # Sort by created_at descending (newest first)
        policies.sort(key=lambda p: p.created_at, reverse=True)
        return policies

    def _parse_policy_directory(self, output_dir: Path) -> Optional[PolicyInfo]:
        """Parse a training output directory to extract policy info."""
        dir_name = output_dir.name

        # Try to load metadata file first
        metadata_path = output_dir / "policy_metadata.json"
        loss_history_path = output_dir / "loss_history.json"

        metadata = {}
        loss_history = []

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        if loss_history_path.exists():
            try:
                with open(loss_history_path, "r") as f:
                    loss_history = json.load(f)
            except Exception:
                pass

        # Parse directory name: {policy_type}_{job_id}_{timestamp} or {name}_{job_id}_{timestamp}
        parts = dir_name.rsplit("_", 2)
        if len(parts) >= 2:
            # Could be name_jobid_timestamp or policytype_jobid_timestamp
            policy_type = metadata.get("policy_type", "unknown")
            if not policy_type or policy_type == "unknown":
                # Try to infer from directory name
                first_part = parts[0].lower()
                if first_part in ["smolvla", "diffusion", "act", "pi05"]:
                    policy_type = first_part
                else:
                    policy_type = "smolvla"  # Default assumption
        else:
            policy_type = metadata.get("policy_type", "smolvla")

        # Check for checkpoints
        checkpoint_path = ""
        checkpoints_dir = output_dir / "checkpoints"
        last_checkpoint = checkpoints_dir / "last" / "pretrained_model"

        if last_checkpoint.exists():
            checkpoint_path = str(last_checkpoint)
            status = "completed"
        elif checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
            # Has some checkpoints but not "last" - might be interrupted
            status = "failed"
            # Find latest checkpoint
            for ckpt in sorted(checkpoints_dir.iterdir(), reverse=True):
                if ckpt.is_dir() and (ckpt / "pretrained_model").exists():
                    checkpoint_path = str(ckpt / "pretrained_model")
                    break
        else:
            # No checkpoints - might be empty/failed early
            status = "failed"

        # Get steps from train_config.json in checkpoint if available
        steps = metadata.get("final_step", 0)
        total_steps = metadata.get("total_steps", 0)
        final_loss = metadata.get("final_loss")

        if checkpoint_path and not steps:
            # Try to read from checkpoint
            train_config_path = Path(checkpoint_path) / "train_config.json"
            if train_config_path.exists():
                try:
                    with open(train_config_path, "r") as f:
                        train_config = json.load(f)
                        total_steps = train_config.get("steps", 0)
                except Exception:
                    pass

            # Try to get step from training_step.json in training_state
            training_state_dir = Path(checkpoint_path).parent / "training_state"
            step_file = training_state_dir / "training_step.json"
            if step_file.exists():
                try:
                    with open(step_file, "r") as f:
                        step_data = json.load(f)
                        steps = step_data.get("step", 0)
                except Exception:
                    pass

        # Get created_at from metadata or directory timestamp
        created_at = metadata.get("created_at")
        if not created_at:
            # Parse from directory name: ..._YYYYMMDD_HHMMSS
            try:
                timestamp_str = parts[-1] if len(parts) >= 3 else ""
                date_str = parts[-2] if len(parts) >= 3 else ""
                if timestamp_str and date_str:
                    dt = datetime.strptime(f"{date_str}_{timestamp_str}", "%Y%m%d_%H%M%S")
                    created_at = dt.isoformat()
            except Exception:
                pass

        if not created_at:
            # Fallback to directory modification time
            created_at = datetime.fromtimestamp(output_dir.stat().st_mtime).isoformat()

        # Get display name
        name = metadata.get("name", dir_name)

        return PolicyInfo(
            id=dir_name,
            name=name,
            policy_type=policy_type,
            status=status,
            steps=steps,
            total_steps=total_steps,
            dataset_repo_id=metadata.get("dataset_repo_id", ""),
            created_at=created_at,
            final_loss=final_loss,
            checkpoint_path=checkpoint_path,
            loss_history=loss_history,
            output_dir=str(output_dir),
        )

    def get_policy(self, policy_id: str) -> Optional[PolicyInfo]:
        """Get a specific policy by ID."""
        # Check if it's the active training job
        if self.active_job and self.active_job.output_dir and self.active_job.output_dir.name == policy_id:
            return PolicyInfo(
                id=policy_id,
                name=self.active_job.config.get("policy_name", policy_id),
                policy_type=self.active_job.policy_type.value,
                status="training",
                steps=self.active_job.progress.step,
                total_steps=self.active_job.progress.total_steps,
                dataset_repo_id=self.active_job.dataset_repo_id,
                created_at=self.active_job.created_at.isoformat(),
                final_loss=self.active_job.progress.loss,
                checkpoint_path="",
                loss_history=self.active_job.progress.loss_history,
                output_dir=str(self.active_job.output_dir),
            )

        # Look in outputs directory
        output_dir = self.outputs_path / policy_id
        if output_dir.exists():
            return self._parse_policy_directory(output_dir)

        return None

    def get_policy_config(self, policy_id: str) -> Optional[PolicyConfig]:
        """Parse the policy's config.json to extract input/output features.

        Returns:
            PolicyConfig with cameras, arms, and dimensions extracted,
            or None if policy or config not found.
        """
        policy = self.get_policy(policy_id)
        if not policy or not policy.checkpoint_path:
            return None

        config_path = Path(policy.checkpoint_path) / "config.json"
        if not config_path.exists():
            logger.warning(f"[TrainingService] No config.json found at {config_path}")
            return None

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"[TrainingService] Failed to read config.json: {e}")
            return None

        # Extract cameras from input_features
        cameras = []
        input_features = config.get("input_features", {})
        for key in input_features:
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                cameras.append(camera_name)

        # Extract state dimension
        state_shape = input_features.get("observation.state", {}).get("shape", [14])
        state_dim = state_shape[0] if state_shape else 14

        # Extract action dimension
        output_features = config.get("output_features", {})
        action_shape = output_features.get("action", {}).get("shape", [14])
        action_dim = action_shape[0] if action_shape else 14

        # Infer which arms were used (action_dim reflects actual motor count,
        # unlike state_dim which may include vel/tau for extended-state policies)
        arms = self._infer_arms_from_policy(policy, state_dim, action_dim)

        policy_type = config.get("type", "unknown")

        logger.info(f"[TrainingService] Policy config: cameras={cameras}, arms={arms}, state_dim={state_dim}")

        return PolicyConfig(
            cameras=cameras,
            arms=arms,
            state_dim=state_dim,
            action_dim=action_dim,
            policy_type=policy_type
        )

    def _infer_arms_from_policy(self, policy: PolicyInfo, state_dim: int, action_dim: int = 0) -> List[str]:
        """Infer which arms were used by checking dataset info.json or dimension heuristics."""
        # First try to get from dataset metadata
        if policy.dataset_repo_id:
            info_path = self.datasets_path / policy.dataset_repo_id / "meta" / "info.json"
            if info_path.exists():
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])
                    arms = set()
                    for name in state_names:
                        if name.startswith("left_"):
                            arms.add("left")
                        elif name.startswith("right_"):
                            arms.add("right")
                    if arms:
                        return list(arms)
                except Exception as e:
                    logger.warning(f"[TrainingService] Failed to read dataset info.json: {e}")

        # Fallback: dimension-based heuristic
        # Use action_dim (actual motor count) over state_dim, since state_dim
        # may be inflated by extended state (pos + vel + tau = 3x motor count)
        check_dim = action_dim if action_dim > 0 else state_dim
        if check_dim <= 7:
            return ["left"]
        else:
            return ["left", "right"]

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy and its output directory."""
        # Don't delete if it's currently training
        if self.active_job and self.active_job.output_dir and self.active_job.output_dir.name == policy_id:
            raise ValueError("Cannot delete a policy that is currently training")

        output_dir = self.outputs_path / policy_id
        if not output_dir.exists():
            raise ValueError(f"Policy {policy_id} not found")

        # Security check
        if ".." in policy_id:
            raise ValueError("Invalid policy_id")

        try:
            shutil.rmtree(output_dir)
            logger.info(f"Deleted policy {policy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {e}")
            raise

    def rename_policy(self, policy_id: str, new_name: str) -> bool:
        """Rename a policy by updating its metadata."""
        output_dir = self.outputs_path / policy_id
        if not output_dir.exists():
            raise ValueError(f"Policy {policy_id} not found")

        metadata_path = output_dir / "policy_metadata.json"

        # Load existing metadata or create new
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        metadata["name"] = new_name

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Renamed policy {policy_id} to '{new_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to rename policy {policy_id}: {e}")
            raise

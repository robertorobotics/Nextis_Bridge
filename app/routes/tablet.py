import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.deployment.types import RuntimeState
from app.core.hardware.types import ConnectionStatus
from app.dependencies import get_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tablet"])


class CameraInfo(BaseModel):
    id: str
    name: str
    streaming: bool


class TabletStatus(BaseModel):
    connected: bool = False
    teleop_active: bool = False
    recording_active: bool = False
    deployment_active: bool = False
    connected_arms: int = 0
    connected_cameras: list[CameraInfo] = []
    episode_count: int = 0
    current_dataset: str | None = None
    active_policy: str | None = None


@router.get("/tablet/status", response_model=TabletStatus)
def tablet_status():
    """Aggregated status for the tablet touch-panel UI (polled every ~1.5s)."""
    try:
        state = get_state()
    except Exception:
        return TabletStatus()

    try:
        # -- Arms --
        registry = state.arm_registry
        if registry:
            arm_statuses = registry.arm_status.values()
            connected_arms = sum(
                1 for s in arm_statuses if s == ConnectionStatus.CONNECTED
            )
            connected = connected_arms > 0
        else:
            connected_arms = 0
            connected = False

        # -- Teleop & recording --
        teleop = state.teleop_service
        if teleop:
            teleop_active = bool(teleop.is_running)
            recording_active = bool(getattr(teleop, "recording_active", False))
            episode_count = int(getattr(teleop, "episode_count", 0))
            dataset_cfg = getattr(teleop, "dataset_config", None)
            current_dataset = (
                dataset_cfg.get("repo_id") if dataset_cfg and teleop.session_active else None
            )
        else:
            teleop_active = False
            recording_active = False
            episode_count = 0
            current_dataset = None

        # -- Deployment --
        runtime = state.deployment_runtime
        if runtime:
            deployment_active = runtime._state in (
                RuntimeState.RUNNING,
                RuntimeState.HUMAN_ACTIVE,
            )
            if deployment_active and runtime._checkpoint_path:
                active_policy = Path(runtime._checkpoint_path).name
            else:
                active_policy = None
        else:
            deployment_active = False
            active_policy = None

        # -- Cameras --
        cam_svc = state.camera_service
        connected_cameras: list[CameraInfo] = []
        if cam_svc:
            cam_status = cam_svc.get_status()
            cam_config = cam_svc.get_camera_config()
            for key, info in cam_status.items():
                if info.get("status") == "connected":
                    cfg = cam_config.get(key, {})
                    connected_cameras.append(
                        CameraInfo(
                            id=key,
                            name=cfg.get("name", key),
                            streaming=True,
                        )
                    )

        return TabletStatus(
            connected=connected,
            teleop_active=teleop_active,
            recording_active=recording_active,
            deployment_active=deployment_active,
            connected_arms=connected_arms,
            connected_cameras=connected_cameras,
            episode_count=episode_count,
            current_dataset=current_dataset,
            active_policy=active_policy,
        )
    except Exception:
        logger.exception("Error building tablet status")
        return TabletStatus()

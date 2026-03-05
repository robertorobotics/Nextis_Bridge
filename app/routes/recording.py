import logging
import sys

from fastapi import APIRouter, Request

from app.dependencies import get_state

_recording_logger = logging.getLogger("recording_debug")

router = APIRouter(tags=["recording"])


@router.get("/recording/options")
def get_recording_options():
    """Returns available cameras and arms for recording selection."""
    system = get_state()
    cameras = []
    arms = []

    # 1. Cameras: always read from CameraService config (independent of robot mock)
    if system.camera_service:
        try:
            cam_configs = system.camera_service.get_camera_config()
            # cam_configs is Dict[str, Dict] — iterate .items() not just keys
            for cam_id, cam_cfg in cam_configs.items():
                cameras.append({
                    "id": cam_id,
                    "name": cam_id.replace("_", " ").title()
                })
        except Exception:
            pass

    # 2. Arms: read from ArmRegistry (the real source of truth)
    if system.arm_registry:
        try:
            for arm in system.arm_registry.get_followers():
                joints = len(arm.get("config", {}).get("home_position", {})) or 7
                arms.append({
                    "id": arm["id"],
                    "name": arm.get("name", arm["id"]),
                    "joints": joints,
                    "status": arm.get("status", "disconnected"),
                })
        except Exception:
            pass

    # 3. Legacy fallback: real robot (not mock) with left_arm/right_arm
    if not arms and system.robot and getattr(system.robot, 'is_connected', False) and not getattr(system.robot, 'is_mock', False):
        if hasattr(system.robot, 'left_arm') and system.robot.left_arm:
            arms.append({"id": "left", "name": "Left Arm", "joints": 7})
        if hasattr(system.robot, 'right_arm') and system.robot.right_arm:
            arms.append({"id": "right", "name": "Right Arm", "joints": 7})
        if not arms:
            arms.append({"id": "default", "name": "Robot Arm", "joints": 7})

    # 4. Pairings: return follower-based pairing info for recording selection
    pairings = []
    if system.arm_registry:
        try:
            for p in system.arm_registry.get_pairings():
                pairings.append({
                    "id": p["follower_id"],  # follower_id is the pairing key
                    "name": p.get("name", f"{p['leader_id']} → {p['follower_id']}"),
                    "leader_id": p["leader_id"],
                    "follower_id": p["follower_id"],
                })
        except Exception:
            pass

    # 5. Detect extended state support (Damiao followers with MIT response cache)
    supports_extended_state = False
    if system.arm_registry:
        for arm_dict in system.arm_registry.get_followers():
            instance = system.arm_registry.get_arm_instance(arm_dict["id"])
            if instance and hasattr(instance, 'bus') and hasattr(instance.bus, 'read_cached_velocities'):
                supports_extended_state = True
                break

    return {"cameras": cameras, "arms": arms, "pairings": pairings,
            "supports_extended_state": supports_extended_state}

@router.post("/recording/session/start")
async def start_recording_session(request: Request):
    system = get_state()
    print("\n>>> API: /recording/session/start called")
    _recording_logger.info("API: /recording/session/start called")
    sys.stdout.flush()

    data = await request.json()
    repo_id = data.get("repo_id")
    task = data.get("task")
    selected_cameras = data.get("selected_cameras")        # list of camera IDs or None (all)
    selected_pairing_ids = data.get("selected_pairing_ids")  # list of follower arm IDs or None (all)
    selected_arms = data.get("selected_arms")              # legacy: list of arm prefixes or None (all)
    record_extended_state = data.get("record_extended_state", False)  # include vel + torque in observation
    # Streaming encoding overrides (None = use config defaults)
    streaming_encoding = data.get("streaming_encoding")      # bool or None
    vcodec = data.get("vcodec")                              # str or None
    encoder_queue_maxsize = data.get("encoder_queue_maxsize") # int or None
    encoder_threads = data.get("encoder_threads")            # int or None
    print(f"    repo_id={repo_id}, task={task}, cameras={selected_cameras}, pairings={selected_pairing_ids}, arms={selected_arms}")
    _recording_logger.info(f"  repo_id={repo_id}, task={task}, cameras={selected_cameras}, pairings={selected_pairing_ids}, arms={selected_arms}")

    if not repo_id or not task:
        print("    ERROR: Missing repo_id or task")
        _recording_logger.error("Missing repo_id or task")
        return {"status": "error", "message": "Missing repo_id or task"}

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        _recording_logger.info("Calling teleop_service.start_recording_session...")
        system.teleop_service.start_recording_session(
            repo_id, task,
            selected_cameras=selected_cameras,
            selected_pairing_ids=selected_pairing_ids,
            selected_arms=selected_arms,
            record_extended_state=record_extended_state,
            streaming_encoding=streaming_encoding,
            vcodec=vcodec,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Session started (episode_count={episode_count})")
        _recording_logger.info(f"SUCCESS: Session started (episode_count={episode_count})")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Recording Session Started", "episode_count": episode_count}

@router.post("/recording/session/stop")
def stop_recording_session():
    system = get_state()
    print("\n>>> API: /recording/session/stop called")
    _recording_logger.info("API: /recording/session/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_recording_session()
        print("    SUCCESS: Session stopped")
        _recording_logger.info("SUCCESS: Session stopped")
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")

    return {"status": "success", "message": "Recording Session Finalized"}

@router.post("/recording/episode/start")
def start_episode():
    system = get_state()
    print("\n>>> API: /recording/episode/start called")
    _recording_logger.info("API: /recording/episode/start called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.start_episode()
        print("    SUCCESS: Episode started")
        _recording_logger.info("SUCCESS: Episode started")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Episode Started"}

@router.post("/recording/episode/stop")
def stop_episode():
    system = get_state()
    print("\n>>> API: /recording/episode/stop called")
    _recording_logger.info("API: /recording/episode/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_episode()
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Episode stopped (total: {episode_count})")
        _recording_logger.info(f"SUCCESS: Episode stopped (total: {episode_count})")
        return {"status": "success", "message": "Episode Saved", "episode_count": episode_count}
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@router.delete("/recording/episode/last")
def delete_last_episode():
    system = get_state()
    if not system.teleop_service:
         return {"status": "error", "message": "Teleop Service not active"}

    if not system.teleop_service.session_active:
         return {"status": "error", "message": "No recording session active"}

    if not system.teleop_service.dataset:
         return {"status": "error", "message": "No dataset loaded"}

    try:
        repo_id = system.teleop_service.dataset.repo_id
        current_count = system.teleop_service.episode_count

        if current_count <= 0:
            return {"status": "error", "message": "No episodes to delete"}

        # Delete the last episode (index = count - 1)
        last_index = current_count - 1

        print(f"[DELETE_LAST] Starting delete for episode {last_index}")
        print(f"[DELETE_LAST] BEFORE: episode_count={current_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        # CRITICAL: Flush pending episode data to disk BEFORE deletion
        # Without this, the metadata_buffer may have unflushed episode data
        # that won't be found on disk by delete_episode(), causing ghost episodes
        system.teleop_service.sync_to_disk()

        result = system.dataset_service.delete_episode(repo_id, last_index)
        print(f"[DELETE_LAST] delete_episode returned: {result}")

        # Remove orphaned frames from deleted episode so dataset_from/to indices stay accurate
        try:
            consolidate_result = system.dataset_service.consolidate_dataset(repo_id)
            print(f"[DELETE_LAST] consolidate returned: {consolidate_result}")
        except Exception as ce:
            print(f"[DELETE_LAST] WARNING: consolidation failed (non-fatal): {ce}")

        # Refresh metadata from disk AFTER deletion to reload clean state
        system.teleop_service.refresh_metadata_from_disk()

        print(f"[DELETE_LAST] AFTER: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        return {"status": "success", "message": "Last Episode Deleted", "episode_count": system.teleop_service.episode_count}
    except Exception as e:
        import traceback
        print(f"[DELETE_LAST] ERROR: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@router.get("/recording/status")
def get_recording_status():
    system = get_state()
    if not system.teleop_service:
         return {"active": False, "episode_count": 0}

    # TeleopService.get_data() already returns recording info, but explicit endpoint helps too
    return system.teleop_service.get_data().get("recording", {})

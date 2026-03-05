import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from app.core.config import (  # noqa: E402
    DATASETS_DIR,
    FRAME_QUEUE_MAXSIZE,
    STREAMING_ENCODER_QUEUE_MAXSIZE,
    STREAMING_ENCODER_THREADS,
    STREAMING_ENCODING_ENABLED,
    STREAMING_VCODEC,
)

_DEFAULT_DATASETS_PATH = DATASETS_DIR

# Conditional lerobot imports
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.datasets.video_utils import StreamingVideoEncoder, VideoEncodingManager
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep
except ImportError:
    StreamingVideoEncoder = None
    def precise_sleep(dt):
        time.sleep(max(0, dt))


def _get_effective_robot(svc):
    """Return the active robot instance (arm_registry or legacy).

    In arm_registry mode, svc.robot is None; svc._active_robot holds the
    first pairing's follower instance.
    """
    return getattr(svc, '_active_robot', None) or svc.robot


def _build_allowed_motor_keys(
    selected_pairing_ids: list[str] | None,
) -> tuple[set[str], set[str]]:
    """Build leader-exclusion and follower-allowlist key sets.

    SAFETY: Leader motor keys are ALWAYS excluded from recordings.
    Follower motor keys are only included for selected pairings.

    Returns:
        (leader_keys, allowed_follower_keys) — sets of motor key base names
        (e.g. "base", "link1", "shoulder_pan").  If allowed_follower_keys is
        None, all non-leader keys are allowed.
    """
    from app.state import state

    leader_keys: set[str] = set()
    allowed_keys: set[str] | None = None

    if not state.arm_registry:
        return leader_keys, allowed_keys

    # Build leader exclusion set from ALL leaders (regardless of selection)
    for arm_dict in state.arm_registry.get_leaders():
        inst = state.arm_registry.get_arm_instance(arm_dict["id"])
        if inst and hasattr(inst, 'observation_features'):
            for k in inst.observation_features:
                base = k.split(".")[0] if "." in k else k
                leader_keys.add(base)

    # Build follower allowlist from selected pairings
    if selected_pairing_ids is not None:
        allowed_keys = set()
        for fid in selected_pairing_ids:
            pairing = state.arm_registry.get_pairing_by_follower(fid)
            if not pairing:
                continue
            inst = state.arm_registry.get_arm_instance(pairing.follower_id)
            if inst and hasattr(inst, 'observation_features'):
                for k in inst.observation_features:
                    base = k.split(".")[0] if "." in k else k
                    allowed_keys.add(base)

    return leader_keys, allowed_keys


def _is_camera_feature(feat) -> bool:
    """Check if a feature value represents a camera (tuple shape like (H, W, C))."""
    return isinstance(feat, tuple) and len(feat) in (1, 3)


def filter_observation_features(
    obs_features: dict,
    selected_cameras: list[str] | None = None,
    selected_pairing_ids: list[str] | None = None,
    selected_arms: list[str] | None = None,
) -> dict:
    """Filter observation features to only include selected follower arms and cameras.

    SAFETY: Leader arm data is NEVER included.  Only follower arms from
    selected pairings are recorded.  Tool/trigger data is always included.

    Uses pairing-based filtering when selected_pairing_ids is provided,
    falling back to legacy prefix filtering (selected_arms) otherwise.

    Args:
        obs_features: Full robot observation features dict
        selected_cameras: Camera IDs to include (None = all)
        selected_pairing_ids: Follower arm IDs identifying pairings (None = all)
        selected_arms: Legacy arm prefixes ("left", "right") (None = all)
    """
    leader_keys, allowed_keys = _build_allowed_motor_keys(selected_pairing_ids)
    use_pairing_filter = bool(leader_keys) or allowed_keys is not None

    filtered = {}
    for key, feat in obs_features.items():
        # Always include tool/trigger features
        if key.startswith("tool.") or key.startswith("trigger."):
            filtered[key] = feat
            continue

        # Camera features
        if _is_camera_feature(feat):
            if selected_cameras is None or key in selected_cameras:
                filtered[key] = feat
            continue

        # Motor features — pairing-based filter takes priority
        base = key.split(".")[0] if "." in key else key
        if use_pairing_filter:
            if base in leader_keys:
                continue  # SAFETY: never include leader data
            if allowed_keys is None or base in allowed_keys:
                filtered[key] = feat
        else:
            # Legacy prefix filter
            if selected_arms is None:
                filtered[key] = feat
            elif key.startswith("left_") and "left" in selected_arms:
                filtered[key] = feat
            elif key.startswith("right_") and "right" in selected_arms:
                filtered[key] = feat
            elif not key.startswith("left_") and not key.startswith("right_"):
                filtered[key] = feat

    return filtered


def filter_action_features(
    action_features: dict,
    selected_pairing_ids: list[str] | None = None,
    selected_arms: list[str] | None = None,
) -> dict:
    """Filter action features to only include selected follower arms.

    SAFETY: Leader arm data is NEVER included.  Tool/trigger data is
    always included.

    Args:
        action_features: Full robot action features dict
        selected_pairing_ids: Follower arm IDs identifying pairings (None = all)
        selected_arms: Legacy arm prefixes ("left", "right") (None = all)
    """
    leader_keys, allowed_keys = _build_allowed_motor_keys(selected_pairing_ids)
    use_pairing_filter = bool(leader_keys) or allowed_keys is not None

    filtered = {}
    for key, feat in action_features.items():
        # Always include tool/trigger features
        if key.startswith("tool.") or key.startswith("trigger."):
            filtered[key] = feat
            continue

        base = key.split(".")[0] if "." in key else key
        if use_pairing_filter:
            if base in leader_keys:
                continue
            if allowed_keys is None or base in allowed_keys:
                filtered[key] = feat
        else:
            if selected_arms is None:
                filtered[key] = feat
            elif key.startswith("left_") and "left" in selected_arms:
                filtered[key] = feat
            elif key.startswith("right_") and "right" in selected_arms:
                filtered[key] = feat
            elif not key.startswith("left_") and not key.startswith("right_"):
                filtered[key] = feat

    return filtered


def frame_writer_loop(svc):
    """Background thread for writing frames to dataset without blocking teleop loop."""
    print("[FRAME WRITER] Thread Started!")
    written_count = 0

    while not svc._frame_writer_stop.is_set():
        try:
            if svc._frame_queue and svc.dataset is not None:
                try:
                    frame = svc._frame_queue.popleft()
                    svc.dataset.add_frame(frame)
                    written_count += 1
                    if written_count == 1:
                        print("[FRAME WRITER] FIRST FRAME added to dataset buffer!")
                        # Log buffer size to confirm it's working
                        if hasattr(svc.dataset, 'episode_buffer'):
                            buf_size = svc.dataset.episode_buffer.get('size', 0)
                            print(f"[FRAME WRITER] Episode buffer size: {buf_size}")
                    elif written_count % 30 == 0:
                        print(f"[FRAME WRITER] Written {written_count} frames")
                except IndexError:
                    time.sleep(0.005)  # Queue empty, short sleep
            else:
                time.sleep(0.01)
        except Exception as e:
            import traceback
            print(f"[FRAME WRITER] ERROR adding frame: {e}")
            print(traceback.format_exc())
            time.sleep(0.01)

    # Drain remaining frames before stopping
    while svc._frame_queue and svc.dataset is not None:
        try:
            frame = svc._frame_queue.popleft()
            svc.dataset.add_frame(frame)
            written_count += 1
        except IndexError:
            break

    print(f"[FRAME WRITER] Thread Stopped (written={written_count})")


def start_frame_writer(svc):
    """Starts the background frame writer thread."""
    if svc._frame_writer_thread is not None and svc._frame_writer_thread.is_alive():
        return
    svc._frame_writer_stop.clear()
    svc._frame_writer_thread = threading.Thread(target=frame_writer_loop, args=(svc,), daemon=True, name="FrameWriter")
    svc._frame_writer_thread.start()
    logger.info("Frame Writer Thread Started")


def stop_frame_writer(svc):
    """Stops the background frame writer thread."""
    svc._frame_writer_stop.set()
    if svc._frame_writer_thread is not None:
        svc._frame_writer_thread.join(timeout=2.0)
        svc._frame_writer_thread = None
    logger.info("Frame Writer Thread Stopped")


def recording_capture_loop(svc):
    """Background thread that captures observations at recording fps.

    OPTIMIZED: Uses cached teleop data for motors (no slow hardware reads).
    Only cameras use async_read which is fast (Zero-Order Hold pattern).
    This allows reliable 30fps capture without blocking.
    """
    print(f"[REC CAPTURE] Thread started at {svc.recording_fps}fps")
    target_dt = 1.0 / svc.recording_fps

    # Pre-build motor key filter sets (rebuilt if pairing selection changes)
    cached_pairing_ids = getattr(svc, '_selected_pairing_ids', None)
    leader_keys, allowed_keys = _build_allowed_motor_keys(cached_pairing_ids)

    # Wall-clock tracking for actual fps measurement
    episode_start_time = None

    while not svc._recording_stop_event.is_set():
        robot = _get_effective_robot(svc)
        if svc.recording_active and robot and svc.dataset is not None:
            start = time.perf_counter()

            # Rebuild key filter if pairing selection changed
            current_ids = getattr(svc, '_selected_pairing_ids', None)
            if current_ids != cached_pairing_ids:
                cached_pairing_ids = current_ids
                leader_keys, allowed_keys = _build_allowed_motor_keys(cached_pairing_ids)

            # Track when recording actually starts
            if episode_start_time is None:
                episode_start_time = time.perf_counter()
                print("[REC CAPTURE] Episode recording started at wall-clock t=0")
                print(f"[REC CAPTURE] Target: {svc.recording_fps}fps ({target_dt*1000:.1f}ms per frame)")

            try:
                # FAST STRATEGY: Use cached data from teleop loop (no hardware reads!)
                # The teleop loop runs at 60Hz and caches motor positions in _latest_leader_action
                action = {}
                obs = {}

                # SOURCE 1a: LEADER positions → action (the human intent)
                # Filter using pairing-based key sets (leader exclusion + follower allowlist)
                use_pairing_filter = bool(leader_keys) or allowed_keys is not None
                with svc._action_lock:
                    if svc._latest_leader_action:
                        for key, val in svc._latest_leader_action.items():
                            base = key.split(".")[0] if "." in key else key
                            if use_pairing_filter:
                                if base in leader_keys:
                                    continue
                                if allowed_keys is not None and base not in allowed_keys:
                                    continue
                            else:
                                if svc._selected_arms is not None:
                                    if key.startswith("left_") and "left" not in svc._selected_arms:
                                        continue
                                    elif key.startswith("right_") and "right" not in svc._selected_arms:
                                        continue
                            action[key] = val

                # SOURCE 1b: FOLLOWER positions → observation (the robot reality)
                with svc._follower_obs_lock:
                    if svc._latest_follower_obs:
                        for key, val in svc._latest_follower_obs.items():
                            base = key.split(".")[0] if "." in key else key
                            if use_pairing_filter:
                                if base in leader_keys:
                                    continue
                                if allowed_keys is not None and base not in allowed_keys:
                                    continue
                            else:
                                if svc._selected_arms is not None:
                                    if key.startswith("left_") and "left" not in svc._selected_arms:
                                        continue
                                    elif key.startswith("right_") and "right" not in svc._selected_arms:
                                        continue
                            obs[key] = val

                # Fallback: if follower cache not yet populated, use leader as obs
                # (first few frames before Damiao MIT response arrives)
                if not any('.pos' in k for k in obs) and action:
                    for k, v in action.items():
                        if k not in obs:
                            obs[k] = v
                    if svc._recording_frame_counter == 0:
                        print("[REC CAPTURE] WARNING: Using leader positions as obs fallback (follower cache empty)")

                # SOURCE 1b: Extended motor state (velocity + torque) from follower robot
                # These come from the MIT response cache, NOT from the leader action cache
                if getattr(svc, '_record_extended_state', False):
                    active_robot = None
                    if svc._pairing_contexts:
                        for ctx in svc._pairing_contexts:
                            if ctx.active_robot and hasattr(ctx.active_robot, 'bus') and hasattr(ctx.active_robot.bus, 'read_cached_velocities'):
                                active_robot = ctx.active_robot
                                break
                    elif hasattr(svc, '_active_robot') and svc._active_robot:
                        active_robot = svc._active_robot
                    elif svc.robot:
                        active_robot = svc.robot

                    if active_robot and hasattr(active_robot, 'bus') and hasattr(active_robot.bus, 'read_cached_velocities'):
                        try:
                            velocities = active_robot.bus.read_cached_velocities()
                            torques = active_robot.bus.read_torques()
                            for name, vel in velocities.items():
                                key = f"{name}.vel"
                                base = name
                                if use_pairing_filter:
                                    if base in leader_keys:
                                        continue
                                    if allowed_keys is not None and base not in allowed_keys:
                                        continue
                                obs[key] = vel
                            for name, tau in torques.items():
                                key = f"{name}.tau"
                                base = name
                                if use_pairing_filter:
                                    if base in leader_keys:
                                        continue
                                    if allowed_keys is not None and base not in allowed_keys:
                                        continue
                                obs[key] = tau
                        except Exception as e:
                            if svc._recording_frame_counter == 0:
                                print(f"[REC CAPTURE] Extended state read error: {e}")

                # SOURCE 2: Capture camera images with async_read (FAST - ZOH pattern)
                # Only capture selected cameras
                # Priority: CameraService (standalone) > robot.cameras (legacy)
                cameras_dict = None
                if svc.camera_service and svc.camera_service.cameras:
                    cameras_dict = svc.camera_service.cameras
                elif hasattr(svc, '_active_robot') and svc._active_robot and hasattr(svc._active_robot, 'cameras') and svc._active_robot.cameras:
                    cameras_dict = svc._active_robot.cameras
                elif hasattr(svc.robot, 'cameras') and svc.robot.cameras:
                    cameras_dict = svc.robot.cameras

                if cameras_dict:
                    cameras_to_capture = svc._selected_cameras if svc._selected_cameras else list(cameras_dict.keys())

                    for cam_key in cameras_to_capture:
                        if cam_key not in cameras_dict:
                            continue
                        cam = cameras_dict[cam_key]
                        try:
                            # async_read(blocking=False) returns last cached frame instantly
                            if hasattr(cam, 'async_read'):
                                frame = cam.async_read(blocking=False)
                                if frame is not None:
                                    obs[cam_key] = frame
                                    if svc._recording_frame_counter == 0:
                                        print(f"[REC CAPTURE] Camera {cam_key}: shape={frame.shape}")
                            # Also capture depth if enabled
                            if hasattr(cam, 'async_read_depth') and hasattr(cam.config, 'use_depth') and cam.config.use_depth:
                                depth_frame = cam.async_read_depth(blocking=False)
                                if depth_frame is not None:
                                    # Expand depth from (H, W) to (H, W, 1) for dataset compatibility
                                    if depth_frame.ndim == 2:
                                        depth_frame = depth_frame[..., np.newaxis]
                                    obs[f"{cam_key}_depth"] = depth_frame
                                    if svc._recording_frame_counter == 0:
                                        print(f"[REC CAPTURE] Depth {cam_key}_depth: shape={depth_frame.shape}")
                        except Exception as cam_err:
                            if svc._recording_frame_counter == 0:
                                print(f"[REC CAPTURE] Camera {cam_key} error: {cam_err}")

                # SOURCE 3: Tool/trigger state
                from app.core.hardware.tool_state import get_tool_observations
                from app.state import state
                tool_obs = get_tool_observations(state.tool_registry, state.trigger_listener)
                obs.update(tool_obs)

                # Check data availability
                has_motor_data = any('.pos' in k for k in obs.keys())
                has_camera_data = any(hasattr(obs.get(k), 'shape') for k in obs.keys())

                if svc._recording_frame_counter == 0:
                    print(f"[REC CAPTURE] action keys ({len(action)}): {list(action.keys())}")
                    print(f"[REC CAPTURE] obs keys ({len(obs)}): {list(obs.keys())}")
                    # Show the delta between leader and follower to verify gravity offset
                    for key in action:
                        if key in obs:
                            delta = action[key] - obs[key]
                            if abs(delta) > 0.01:
                                print(f"[REC CAPTURE] leader-follower delta: {key} = {delta:+.4f} rad")

                # Need at least motor data OR camera data
                if not has_motor_data and not has_camera_data:
                    if svc._recording_frame_counter == 0:
                        print(f"[REC CAPTURE] WARNING: No data! Teleop running: {svc.is_running}")
                    time.sleep(0.005)
                    continue

                # Build frame using LeRobot helpers
                try:
                    obs_frame = build_dataset_frame(svc.dataset.features, obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(svc.dataset.features, action, prefix=ACTION)

                    frame = {
                        **obs_frame,
                        **action_frame,
                        "task": svc.dataset_config.get("task", ""),
                    }

                    if svc._recording_frame_counter == 0:
                        print(f"[REC CAPTURE] Built frame with keys: {list(frame.keys())}")

                    # Queue for async writing (with backpressure)
                    if len(svc._frame_queue) >= FRAME_QUEUE_MAXSIZE:
                        if svc._recording_frame_counter % 30 == 0:
                            logger.warning(
                                f"[REC CAPTURE] Frame queue full ({len(svc._frame_queue)}/{FRAME_QUEUE_MAXSIZE}), "
                                f"dropping frame — encoder may be falling behind"
                            )
                    else:
                        svc._frame_queue.append(frame)
                        svc._recording_frame_counter += 1

                    if svc._recording_frame_counter == 1:
                        print("[REC CAPTURE] FIRST FRAME captured and queued!")
                    elif svc._recording_frame_counter % 30 == 0:
                        wall_elapsed = time.perf_counter() - episode_start_time
                        actual_fps = svc._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                        queue_size = len(svc._frame_queue)
                        print(f"[REC CAPTURE] {svc._recording_frame_counter} frames ({actual_fps:.1f}fps), queue: {queue_size}")

                except Exception as frame_err:
                    if svc._recording_frame_counter == 0:
                        import traceback
                        print(f"[REC CAPTURE] Frame build error: {frame_err}")
                        print(traceback.format_exc())

            except Exception as e:
                import traceback
                if svc._recording_frame_counter == 0 or svc._recording_frame_counter % 30 == 0:
                    print(f"[REC CAPTURE] Error: {e}")
                    print(traceback.format_exc())

            # Maintain target fps with precise timing
            elapsed = time.perf_counter() - start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                precise_sleep(sleep_time)
        else:
            # Not recording - log why (only once per state change)
            idle_state = (svc.recording_active, robot is not None, svc.dataset is not None)
            if not hasattr(svc, '_last_idle_reason') or svc._last_idle_reason != idle_state:
                svc._last_idle_reason = idle_state
                if not svc.recording_active:
                    pass  # Normal idle state, don't spam logs
                else:
                    print(f"[REC CAPTURE] IDLE - recording_active:{svc.recording_active}, robot:{robot is not None}, dataset:{svc.dataset is not None}")

            # Reset episode timing when not actively recording
            if episode_start_time is not None:
                wall_elapsed = time.perf_counter() - episode_start_time
                actual_fps = svc._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                print(f"[REC CAPTURE] Episode ended: {svc._recording_frame_counter} frames in {wall_elapsed:.1f}s = {actual_fps:.1f}fps")
                episode_start_time = None
            time.sleep(0.01)  # Idle when not recording

    print(f"[REC CAPTURE] Thread stopped ({svc._recording_frame_counter} total frames)")


def start_recording_capture(svc):
    """Starts the recording capture thread."""
    if svc._recording_capture_thread is not None and svc._recording_capture_thread.is_alive():
        return
    svc._recording_stop_event.clear()
    svc._recording_capture_thread = threading.Thread(
        target=recording_capture_loop,
        args=(svc,),
        daemon=True,
        name="RecCapture"
    )
    svc._recording_capture_thread.start()
    logger.info("Recording Capture Thread Started")


def stop_recording_capture(svc):
    """Stops the recording capture thread."""
    svc._recording_stop_event.set()
    if svc._recording_capture_thread is not None:
        svc._recording_capture_thread.join(timeout=2.0)
        svc._recording_capture_thread = None
    logger.info("Recording Capture Thread Stopped")


def start_recording_session(
    svc,
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str = None,
    selected_cameras: list[str] | None = None,
    selected_pairing_ids: list[str] | None = None,
    selected_arms: list[str] | None = None,
    record_extended_state: bool = False,
    streaming_encoding: bool | None = None,
    vcodec: str | None = None,
    encoder_queue_maxsize: int | None = None,
    encoder_threads: int | None = None,
):
    """Initializes a new LeRobotDataset for recording.

    Args:
        svc: TeleoperationService instance
        repo_id: Dataset repository ID
        task: Task description
        fps: Recording frames per second
        root: Custom dataset root path
        selected_cameras: Camera IDs to record (None = all)
        selected_pairing_ids: Follower arm IDs identifying pairings to record (None = all)
        selected_arms: Legacy arm prefixes ("left", "right") — used when
            selected_pairing_ids is not provided
        streaming_encoding: Enable real-time MP4 encoding (None = use config default)
        vcodec: Video codec (None = use config default)
        encoder_queue_maxsize: Per-camera encoder queue size (None = use config default)
        encoder_threads: Threads per encoder (None = use config default)
    """
    from app.state import state

    print("=" * 60)
    print(f"[START_SESSION] Called with repo_id='{repo_id}', task='{task}'")
    print(f"  selected_cameras: {selected_cameras}")
    print(f"  selected_pairing_ids: {selected_pairing_ids}")
    print(f"  selected_arms: {selected_arms}")
    print(f"  session_active: {svc.session_active}")
    robot = _get_effective_robot(svc)
    print(f"  robot: {robot is not None} ({type(robot).__name__ if robot else 'None'})")
    print("=" * 60)

    if svc.session_active:
        print("[START_SESSION] ERROR: Session already active!")
        raise Exception("Session already active")

    # Store selections for use during recording
    svc._selected_cameras = selected_cameras
    svc._selected_pairing_ids = selected_pairing_ids
    svc._selected_arms = selected_arms

    # --- Pre-start validation ---
    if selected_pairing_ids and state.arm_registry:
        for fid in selected_pairing_ids:
            pairing = state.arm_registry.get_pairing_by_follower(fid)
            if not pairing:
                raise ValueError(f"No pairing found for follower '{fid}'")
            follower = state.arm_registry.get_arm_instance(pairing.follower_id)
            if not follower or not getattr(follower, 'is_connected', False):
                raise ValueError(f"Follower '{fid}' not connected — cannot record")

    if selected_cameras and state.camera_service:
        for cam_id in selected_cameras:
            cam = state.camera_service.cameras.get(cam_id)
            if not cam:
                logger.warning(f"Camera '{cam_id}' not connected — may miss frames")

    # Resolve streaming encoding defaults from config
    use_streaming = streaming_encoding if streaming_encoding is not None else STREAMING_ENCODING_ENABLED
    effective_vcodec = vcodec or STREAMING_VCODEC
    effective_queue_maxsize = encoder_queue_maxsize if encoder_queue_maxsize is not None else STREAMING_ENCODER_QUEUE_MAXSIZE
    effective_encoder_threads = encoder_threads if encoder_threads is not None else STREAMING_ENCODER_THREADS

    # Guard: streaming requires StreamingVideoEncoder to be available
    if use_streaming and StreamingVideoEncoder is None:
        logger.warning("StreamingVideoEncoder not available (lerobot too old?), falling back to batch encoding")
        use_streaming = False

    print(f"[START_SESSION] Streaming encoding: {use_streaming} (vcodec={effective_vcodec})")

    # Set default root to app datasets directory
    if root is None:
        base_dir = _DEFAULT_DATASETS_PATH
    else:
        base_dir = Path(root)

    # Target Path
    dataset_dir = base_dir / repo_id

    print(f"[START_SESSION] Dataset dir: {dataset_dir}")

    try:
        # --- Toggle extended state BEFORE reading features ---
        # This ensures observation_features includes .vel and .tau keys in the schema
        svc._record_extended_state = record_extended_state
        if record_extended_state:
            if svc._pairing_contexts and selected_pairing_ids:
                for ctx in svc._pairing_contexts:
                    fid = ctx.pairing_id.split("\u2192")[-1].strip() if "\u2192" in ctx.pairing_id else ctx.pairing_id
                    if fid in selected_pairing_ids and ctx.active_robot:
                        if hasattr(ctx.active_robot, 'config') and hasattr(ctx.active_robot.config, 'record_extended_state'):
                            ctx.active_robot.config.record_extended_state = True
            elif svc._pairing_contexts:
                for ctx in svc._pairing_contexts:
                    if ctx.active_robot and hasattr(ctx.active_robot, 'config') and hasattr(ctx.active_robot.config, 'record_extended_state'):
                        ctx.active_robot.config.record_extended_state = True
            elif robot:
                if hasattr(robot, 'config') and hasattr(robot.config, 'record_extended_state'):
                    robot.config.record_extended_state = True

        # --- Build obs/action features from pairing contexts or legacy robot ---
        raw_obs_features: dict = {}
        raw_action_features: dict = {}

        if svc._pairing_contexts and selected_pairing_ids:
            # Arm-registry mode: combine features from selected follower instances
            for ctx in svc._pairing_contexts:
                # Extract follower_id from pairing_id (format: "leader→follower")
                fid = ctx.pairing_id.split("→")[-1].strip() if "→" in ctx.pairing_id else ctx.pairing_id
                if fid in selected_pairing_ids and ctx.active_robot:
                    raw_obs_features.update(ctx.active_robot.observation_features)
                    raw_action_features.update(ctx.active_robot.action_features)
            if not raw_obs_features:
                raise RuntimeError(
                    f"No matching pairing contexts for selected_pairing_ids={selected_pairing_ids}. "
                    f"Available: {[c.pairing_id for c in svc._pairing_contexts]}"
                )
        elif svc._pairing_contexts:
            # Arm-registry mode, no pairing selection: include all followers
            for ctx in svc._pairing_contexts:
                if ctx.active_robot:
                    raw_obs_features.update(ctx.active_robot.observation_features)
                    raw_action_features.update(ctx.active_robot.action_features)
        elif robot:
            # Legacy mode
            if not hasattr(robot, "observation_features") or not hasattr(robot, "action_features"):
                raise RuntimeError("Robot does not have feature definitions (observation_features/action_features).")
            raw_obs_features = dict(robot.observation_features)
            raw_action_features = dict(robot.action_features)
        else:
            raise Exception("No robot or pairing contexts available — cannot record")

        # --- Inject CameraService cameras into observation features ---
        # CameraService cameras are independent of robot config (which has cameras={}),
        # but must be included in the dataset schema for video recording to work.
        if state.camera_service and state.camera_service.cameras:
            cameras_to_include = selected_cameras or list(state.camera_service.cameras.keys())
            for cam_key in cameras_to_include:
                cam = state.camera_service.cameras.get(cam_key)
                if cam and hasattr(cam, 'async_read'):
                    test_frame = cam.async_read(blocking=False)
                    if test_frame is not None:
                        raw_obs_features[cam_key] = test_frame.shape  # e.g. (480, 640, 3)
                        print(f"[START_SESSION] Added camera feature: {cam_key} shape={test_frame.shape}")
                    else:
                        logger.warning(f"Camera '{cam_key}' has no cached frame — skipping feature")

        # Use LeRobot Helpers to construct correct feature dicts
        from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features

        # Filter observation features (leader exclusion + pairing isolation)
        filtered_obs_features = filter_observation_features(
            raw_obs_features,
            selected_cameras,
            selected_pairing_ids,
            selected_arms,
        )

        # Filter action features
        filtered_action_features = filter_action_features(
            raw_action_features,
            selected_pairing_ids,
            selected_arms,
        )

        # Add tool/trigger features to observation schema
        from app.core.hardware.tool_state import get_tool_action_features
        tool_features = get_tool_action_features(state.tool_registry)
        if tool_features:
            filtered_obs_features.update(tool_features)

        print(f"[START_SESSION] Action features ({len(filtered_action_features)}): {list(filtered_action_features.keys())}")
        print(f"[START_SESSION] Obs features ({len(filtered_obs_features)}): {list(filtered_obs_features.keys())}")
        print(f"[START_SESSION] Action source: LEADER positions")
        print(f"[START_SESSION] Obs source: FOLLOWER positions + vel + tau")

        features = combine_feature_dicts(
            hw_to_dataset_features(filtered_obs_features, prefix=OBS_STR, use_video=True),
            hw_to_dataset_features(filtered_action_features, prefix=ACTION, use_video=True)
        )

        # Determine robot_type for dataset metadata
        robot_type = "unknown"
        if robot and hasattr(robot, 'robot_type'):
            robot_type = robot.robot_type
        elif svc._pairing_contexts:
            for ctx in svc._pairing_contexts:
                if ctx.active_robot and hasattr(ctx.active_robot, 'robot_type'):
                    robot_type = ctx.active_robot.robot_type
                    break

        # 2. Open or Create Dataset (In-Process)
        # Check for VALID dataset (must have meta/info.json)
        is_valid_dataset = (dataset_dir / "meta/info.json").exists()

        if dataset_dir.exists() and not is_valid_dataset:
             logger.warning(f"Found existing folder '{dataset_dir}' but it is not a valid dataset (missing info.json). Backing up...")
             import datetime
             import shutil
             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
             backup_dir = base_dir / f"{repo_id}_backup_{timestamp}"
             shutil.move(str(dataset_dir), str(backup_dir))
             logger.info(f"Moved invalid folder to {backup_dir}")

        if dataset_dir.exists():
            logger.info(f"Valid Dataset exists at {dataset_dir}. Resuming...")
            svc.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_dir,
                local_files_only=True,
                streaming_encoding=use_streaming,
                vcodec=effective_vcodec,
                encoder_queue_maxsize=effective_queue_maxsize,
                encoder_threads=effective_encoder_threads,
            )
        else:
            logger.info("Creating new Dataset...")
            svc.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=dataset_dir,
                robot_type=robot_type,
                features=features,
                use_videos=True,
                streaming_encoding=use_streaming,
                vcodec=effective_vcodec,
                encoder_queue_maxsize=effective_queue_maxsize,
                encoder_threads=effective_encoder_threads,
            )

        svc.dataset.meta.metadata_buffer_size = 1
        svc._streaming_encoding = use_streaming
        print("[START_SESSION] Dataset created/loaded successfully")

        # CRITICAL: Set episode_count from actual dataset state
        # For new dataset: total_episodes = 0
        # For existing dataset: total_episodes = actual count
        svc.episode_count = svc.dataset.meta.total_episodes
        print(f"[START_SESSION] Episode count set to {svc.episode_count} (from dataset)")

        # 3. Start Video Encoding / Image Writer
        if use_streaming:
            # Streaming mode: encoder is internal to LeRobotDataset, no
            # VideoEncodingManager or PNG image writer needed
            print("[START_SESSION] Streaming encoding active — skipping VideoEncodingManager and image writer")
        else:
            # Batch mode: use VideoEncodingManager + PNG image writer
            if not svc.video_manager:
                svc.video_manager = VideoEncodingManager(svc.dataset)
                svc.video_manager.__enter__()
            print("[START_SESSION] Video manager started (batch mode)")

            svc.dataset.start_image_writer(num_processes=0, num_threads=4)
            print("[START_SESSION] Image writer started (batch mode)")

        svc.dataset_config = {"repo_id": repo_id, "task": task}
        svc.session_active = True

        # Update recording fps to match dataset
        svc.recording_fps = fps
        svc._recording_frame_counter = 0
        print(f"[START_SESSION] Recording at {svc.recording_fps}fps (teleop={svc.frequency}Hz)")

        # Start async frame writer thread
        start_frame_writer(svc)

        # Start recording capture thread (separate from teleop loop for smooth control)
        start_recording_capture(svc)

        print("[START_SESSION] SUCCESS! Session is now active")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"[START_SESSION] ERROR: {e}")
        print(traceback.format_exc())
        svc.session_active = False
        svc.dataset = None
        raise


def stop_recording_session(svc):
    """Finalizes the LeRobotDataset."""
    print("=" * 60)
    print("[STOP_SESSION] Called!")
    print(f"  session_active: {svc.session_active}")
    print(f"  dataset: {svc.dataset is not None}")
    print(f"  episode_saving: {svc._episode_saving}")
    print("=" * 60)

    if not svc.session_active:
        print("[STOP_SESSION] Not active, returning")
        return

    # AUTO-SAVE: If an episode is actively recording, save it first
    if svc.recording_active:
        print("[STOP_SESSION] Episode still recording - auto-saving before finalize...")
        try:
            stop_episode(svc)
            print("[STOP_SESSION] Auto-save completed successfully")
        except Exception as e:
            print(f"[STOP_SESSION] WARNING: Auto-save failed: {e}")
    elif (svc.dataset and hasattr(svc.dataset, 'episode_buffer')
          and svc.dataset.episode_buffer
          and svc.dataset.episode_buffer.get('size', 0) > 0):
        # Episode buffer has unsaved frames (e.g. E-STOP cleared recording_active
        # but frames were already buffered via add_frame)
        buffer_size = svc.dataset.episode_buffer.get('size', 0)
        print(f"[STOP_SESSION] Found {buffer_size} unsaved frames in episode buffer — emergency saving...")
        try:
            svc.dataset.save_episode()
            svc.episode_count += 1
            print(f"[STOP_SESSION] Emergency save completed (episode {svc.episode_count})")
        except Exception as e:
            print(f"[STOP_SESSION] WARNING: Emergency save failed: {e}")

    # Reset extended state flag on robots (session-scoped, don't persist)
    if getattr(svc, '_record_extended_state', False):
        if svc._pairing_contexts:
            for ctx in svc._pairing_contexts:
                if ctx.active_robot and hasattr(ctx.active_robot, 'config') and hasattr(ctx.active_robot.config, 'record_extended_state'):
                    ctx.active_robot.config.record_extended_state = False
        svc._record_extended_state = False

    print("[STOP_SESSION] Stopping Recording Session...")
    svc.session_active = False

    # Stop recording capture thread first (it produces frames)
    print("[STOP_SESSION] Stopping recording capture thread...")
    stop_recording_capture(svc)

    # Stop async frame writer (drains remaining frames)
    print(f"[STOP_SESSION] Stopping frame writer (queue size: {len(svc._frame_queue)})")
    stop_frame_writer(svc)

    # CRITICAL: Wait for any pending episode save to complete
    # This prevents finalize() from closing writers while save_episode() is still writing
    print("[STOP_SESSION] Acquiring episode save lock (waiting for pending save)...")
    with svc._episode_save_lock:
        print("[STOP_SESSION] Episode save lock acquired, safe to finalize")

        try:
            if svc.dataset:
                print("[STOP_SESSION] Finalizing Dataset...")

                # Check if writer exists before finalize
                has_writer = hasattr(svc.dataset, 'writer') and svc.dataset.writer is not None
                has_meta_writer = (hasattr(svc.dataset, 'meta') and
                                   hasattr(svc.dataset.meta, 'writer') and
                                   svc.dataset.meta.writer is not None)
                print(f"[STOP_SESSION] Before finalize - data writer: {has_writer}, meta writer: {has_meta_writer}")

                # Call finalize to close parquet writers
                svc.dataset.finalize()

                # Verify writers are closed
                has_writer_after = hasattr(svc.dataset, 'writer') and svc.dataset.writer is not None
                has_meta_writer_after = (hasattr(svc.dataset, 'meta') and
                                         hasattr(svc.dataset.meta, 'writer') and
                                         svc.dataset.meta.writer is not None)
                print(f"[STOP_SESSION] After finalize - data writer: {has_writer_after}, meta writer: {has_meta_writer_after}")

                if has_writer_after or has_meta_writer_after:
                    print("[STOP_SESSION] WARNING: Writers not fully closed, forcing close...")
                    # Force close if still open
                    if hasattr(svc.dataset, '_close_writer'):
                        svc.dataset._close_writer()
                    if hasattr(svc.dataset, 'meta') and hasattr(svc.dataset.meta, '_close_writer'):
                        svc.dataset.meta._close_writer()

                print("[STOP_SESSION] Dataset finalized")

                if getattr(svc, '_streaming_encoding', False):
                    print("[STOP_SESSION] Streaming mode — no VideoEncodingManager to close")
                elif svc.video_manager:
                    svc.video_manager.__exit__(None, None, None)
                    svc.video_manager = None
                    print("[STOP_SESSION] Video manager closed (batch mode)")

                svc._streaming_encoding = False
                svc.dataset = None
                print("[STOP_SESSION] SUCCESS! Session Stopped and Saved!")
        except Exception as e:
            import traceback
            print(f"[STOP_SESSION] ERROR: {e}")
            print(traceback.format_exc())
            # Ensure cleanup even on error
            svc.dataset = None


# Episode lifecycle functions moved to app.core.teleop.episode
from app.core.teleop.episode import (  # noqa: E402, F401
    delete_last_episode,
    refresh_metadata_from_disk,
    start_episode,
    stop_episode,
    sync_to_disk,
)

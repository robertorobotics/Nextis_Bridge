import time

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from app.dependencies import get_state

router = APIRouter(tags=["cameras"])


def generate_frames(camera_key: str, max_width: int | None = None, quality: int = 85, target_fps: int = 30):
    import numpy as np
    import torch

    system = get_state()
    sleep_interval = 1.0 / target_fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    last_good_frame = None  # Serve last good frame during brief gaps instead of placeholder
    first_frame = True

    while True:
        frame = None

        # PRIORITY 1: CameraService managed cameras (independent of robot connection)
        if system.camera_service and camera_key in system.camera_service.cameras:
            cam = system.camera_service.cameras[camera_key]
            try:
                if first_frame:
                    # Blocking read for first frame — ensures we serve a real camera image
                    # immediately instead of a placeholder (reduces time-to-first-frame)
                    frame = cam.async_read(blocking=True, timeout_ms=1000)
                else:
                    frame = cam.async_read(blocking=False)  # ZOH: cached frame
            except Exception:
                pass

        # PRIORITY 1b: Robot cameras (legacy path, if cameras were attached directly)
        if frame is None and system.robot and hasattr(system.robot, 'cameras') and system.robot.cameras and camera_key in system.robot.cameras:
            cam = system.robot.cameras[camera_key]
            try:
                frame = cam.async_read(blocking=False)
            except Exception:
                pass

        # Use last good frame during brief gaps (avoid flashing placeholder)
        if frame is not None:
            # Convert PyTorch tensor to numpy if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype("uint8")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            last_good_frame = frame
            first_frame = False
        elif last_good_frame is not None:
            # Serve last good frame instead of placeholder
            frame = last_good_frame
        else:
            # No frame ever received — show placeholder
            pw, ph = 640, 480
            if system.camera_service:
                pw, ph = system.camera_service.get_camera_resolution(camera_key)
            blank_image = np.zeros((ph, pw, 3), np.uint8)
            cv2.putText(blank_image, f"Waiting for {camera_key}...", (50, ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = blank_image

        # Downscale for UI preview if max_width is set
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(sleep_interval)


@router.get("/video_feed/{camera_key}")
def video_feed(
    camera_key: str,
    max_width: int | None = Query(default=None, ge=160, le=3840),
    quality: int = Query(default=85, ge=1, le=100),
    target_fps: int = Query(default=30, ge=1, le=60),
):
    return StreamingResponse(
        generate_frames(camera_key, max_width=max_width, quality=quality, target_fps=target_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/cameras/{camera_key}/snapshot")
def camera_snapshot(camera_key: str):
    """Return a single full-resolution JPEG frame (for AI/training capture)."""
    import numpy as np
    import torch

    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")

    # Try connected camera first (fast path)
    frame = None
    cam = system.camera_service.cameras.get(camera_key)
    if cam and getattr(cam, 'is_connected', False):
        try:
            frame = cam.async_read(blocking=False)
        except Exception:
            pass

    # Fallback to one-shot capture
    if frame is None:
        frame = system.camera_service.capture_snapshot(camera_key)

    if frame is None:
        raise HTTPException(status_code=404, detail=f"No frame available for '{camera_key}'")

    # Convert tensor if needed
    if isinstance(frame, torch.Tensor):
        frame = frame.permute(1, 2, 0).cpu().numpy() * 255
        frame = frame.astype("uint8")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ret:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.post("/cameras/{camera_key}/connect")
def connect_camera(camera_key: str):
    """Connect a single camera by its config key.
    Uses def (not async) so FastAPI runs it in a thread pool — avoids blocking the event loop
    during the 2-3s camera warmup (time.sleep + frame reads).
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    result = system.camera_service.connect_camera(camera_key)
    return result

@router.post("/cameras/{camera_key}/disconnect")
def disconnect_camera(camera_key: str):
    """Disconnect a single camera by its config key.
    Uses def (not async) so FastAPI runs it in a thread pool.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    result = system.camera_service.disconnect_camera(camera_key)
    return result

@router.post("/cameras/{camera_key}/reconnect")
def reconnect_camera(camera_key: str):
    """Disconnect then reconnect a camera with retry logic.
    Uses def (not async) so FastAPI runs it in a thread pool.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    # connect_camera already handles disconnect-existing + retry with backoff
    result = system.camera_service.connect_camera(camera_key)
    return result


@router.get("/cameras/{camera_key}/exposure")
def get_camera_exposure(camera_key: str):
    """Get current exposure/gain settings and supported ranges for a RealSense camera.
    Returns sensor name, current values, and min/max/step/default for each option.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    result = system.camera_service.get_camera_exposure_info(camera_key)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.post("/cameras/{camera_key}/exposure")
async def set_camera_exposure(camera_key: str, request: Request):
    """Set exposure/gain/brightness for a connected RealSense camera.
    Body: {"exposure": 200, "gain": 16, "brightness": 0}
    Omit exposure to re-enable auto-exposure.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    data = await request.json()
    result = system.camera_service.set_camera_exposure(
        camera_key,
        exposure=data.get("exposure"),
        gain=data.get("gain"),
        brightness=data.get("brightness"),
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.post("/cameras/reconnect-all")
def reconnect_all_cameras():
    """Disconnect all cameras then reconnect sequentially with retry logic.
    Uses def (not async) — can take 30+ seconds for multiple cameras.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    system.camera_service.disconnect_all()
    time.sleep(1.0)
    result = system.camera_service.connect_all_cameras()
    return result


@router.get("/cameras/status")
def get_camera_status():
    """Get connection status of all configured cameras."""
    system = get_state()
    if not system.camera_service:
        return {}
    return system.camera_service.get_status()

@router.get("/cameras/scan")
def scan_cameras():
    system = get_state()
    if not system.camera_service:
        return {"status": "error", "message": "Camera service not initialized"}

    # If robot is connected, we combine active cameras with a fresh scan for available ones.
    if system.robot and system.robot.is_connected:
        current_config = system.camera_service.get_camera_config()

        # 1. Get Active Cameras
        active_opencv = []
        active_realsense = []

        active_opencv_indices = set()
        active_realsense_serials = set()

        for key, conf in current_config.items():
            if conf.get("type") == "opencv":
                idx = conf.get("index_or_path")
                active_opencv.append({
                    "index": idx,
                    "name": f"Active Camera ({key})",
                    "port": idx,
                    "is_active": True
                })
                active_opencv_indices.add(str(idx))
            elif conf.get("type") == "intelrealsense":
                serial = conf.get("serial_number_or_name")
                active_realsense.append({
                    "name": f"Active Camera ({key})",
                    "serial_number": serial,
                    "is_active": True
                })
                active_realsense_serials.add(str(serial))

        # 2. Scan for Available Cameras (force=True since user explicitly requested scan)
        available = system.camera_service.scan_cameras(active_ids=list(active_opencv_indices), force=True)

        # 3. Merge (avoid duplicates)
        final_opencv = active_opencv[:]
        for cam in available.get("opencv", []):
            cam_idx = str(cam.get("port") or cam.get("id") or cam.get("index"))
            if cam_idx not in active_opencv_indices:
                cam["is_active"] = False
                final_opencv.append(cam)

        final_realsense = active_realsense[:]
        for cam in available.get("realsense", []):
            cam_serial = str(cam.get("serial_number"))
            if cam_serial not in active_realsense_serials:
                cam["is_active"] = False
                final_realsense.append(cam)
    else:
        # Robot not connected - just do a fresh scan
        available = system.camera_service.scan_cameras()
        final_opencv = available.get("opencv", [])
        final_realsense = available.get("realsense", [])
        for cam in final_opencv:
            cam["is_active"] = False
        for cam in final_realsense:
            cam["is_active"] = False

    # Standaradize keys for Frontend
    for cam in final_opencv:
         if "id" not in cam:
             cam["id"] = cam.get("port") or cam.get("index") or cam.get("index_or_path")
         if "index_or_path" not in cam:
             cam["index_or_path"] = cam.get("id")

    for cam in final_realsense:
         if "id" not in cam:
             cam["id"] = cam.get("serial_number") or cam.get("serial_number_or_name")
         if "serial_number_or_name" not in cam:
             cam["serial_number_or_name"] = cam.get("id")

    return {
        "opencv": final_opencv,
        "realsense": final_realsense,
        "note": "Merged active and available cameras."
    }

@router.get("/cameras/capabilities/{device_type}/{device_id:path}")
def get_camera_capabilities(device_type: str, device_id: str):
    """Probe supported resolutions for a camera device.
    Non-destructive: if camera is already connected, returns actual resolution only.
    Results cached for 60s per device.
    Uses def (not async) so FastAPI runs it in a thread pool — probing can take seconds.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")

    if device_type not in ("opencv", "intelrealsense"):
        raise HTTPException(status_code=400, detail=f"Unsupported device type: {device_type}")

    try:
        result = system.camera_service.get_capabilities(device_type, device_id)
        return result
    except Exception as e:
        return {"resolutions": [], "native": None, "error": str(e)}


@router.get("/cameras/config")
def get_camera_config():
    system = get_state()
    if not system.camera_service:
        return []

    # Return List for Frontend (CameraModal.tsx expects array)
    config = system.camera_service.get_camera_config()
    export_list = []
    for key, val in config.items():
        # Inject key as 'id'
        item = val.copy()
        item["id"] = key # 'camera_1' etc
        export_list.append(item)
    return export_list

@router.post("/cameras/config")
async def update_camera_config(request: Request, background_tasks: BackgroundTasks):
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")

    data = await request.json()

    # Get existing config to compare
    existing_config = system.camera_service.get_camera_config()

    # Convert array format from frontend to dict format for storage
    # Frontend sends: [{id: "cam1", video_device_id: ..., type: ..., use_depth: ...}, ...]
    # Backend expects: {"cam1": {video_device_id: ..., type: ..., use_depth: ...}, ...}
    if isinstance(data, list):
        config_dict = {}
        for item in data:
            cam_id = item.get("id", "unknown")
            # Remove 'id' from the stored config (it's the key)
            config_entry = {k: v for k, v in item.items() if k != "id"}
            vid = config_entry.get("video_device_id", "")

            # Ensure type is set based on video_device_id if not provided
            if "type" not in config_entry:
                if str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4):
                    config_entry["type"] = "opencv"
                else:
                    config_entry["type"] = "intelrealsense"

            # Synthesize index_or_path for opencv cameras
            if config_entry.get("type") == "opencv" and "index_or_path" not in config_entry:
                config_entry["index_or_path"] = vid

            # Synthesize serial_number_or_name for realsense cameras
            if config_entry.get("type") == "intelrealsense" and "serial_number_or_name" not in config_entry:
                config_entry["serial_number_or_name"] = vid

            config_dict[cam_id] = config_entry
        data = config_dict

    system.camera_service.update_camera_config(data)

    # Determine what changed per camera
    cameras_needing_reconnect = []
    needs_full_reload = False

    if set(data.keys()) != set(existing_config.keys()):
        # Cameras added or removed — full reload needed
        needs_full_reload = True
    else:
        for cam_id, new_cfg in data.items():
            old_cfg = existing_config.get(cam_id, {})
            # Device assignment changed (different physical camera) — full reload
            if (new_cfg.get("video_device_id") != old_cfg.get("video_device_id") or
                new_cfg.get("type") != old_cfg.get("type")):
                needs_full_reload = True
                break
            # Resolution or FPS changed — only reconnect THIS camera
            if (new_cfg.get("width") != old_cfg.get("width") or
                new_cfg.get("height") != old_cfg.get("height") or
                new_cfg.get("fps") != old_cfg.get("fps")):
                cameras_needing_reconnect.append(cam_id)

    if needs_full_reload:
        background_tasks.add_task(system.reload)
        return {"status": "success", "message": "Camera config updated. System reloading..."}
    elif cameras_needing_reconnect:
        def reconnect_changed():
            import time as _time
            for cid in cameras_needing_reconnect:
                try:
                    system.camera_service.disconnect_camera(cid)
                    _time.sleep(0.5)
                    system.camera_service.connect_camera(cid)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to reconnect {cid} after resolution change: {exc}")
        background_tasks.add_task(reconnect_changed)
        return {"status": "success", "message": f"Resolution updated. Reconnecting: {', '.join(cameras_needing_reconnect)}"}
    else:
        return {"status": "success", "message": "Camera config updated (no reload needed)."}

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

from app.core.config import CONFIG_PATH, load_config, save_config

logger = logging.getLogger(__name__)


class CameraService:
    _MAX_CONNECT_RETRIES = 3
    _BACKOFF_DELAYS = [1.0, 3.0, 7.0]   # seconds between retries
    _USB_SETTLE_DELAY = 0.5              # seconds for USB stack to release handles
    _HEALTH_CHECK_INTERVAL = 10.0        # seconds between health checks
    _MAX_AUTO_RECONNECT = 2              # max consecutive auto-reconnect attempts per camera

    def __init__(self):
        self._cameras: Dict[str, Any] = {}          # Connected camera instances (Camera objects)
        self._camera_status: Dict[str, str] = {}    # "connected" | "disconnected" | "error"
        self._camera_errors: Dict[str, str] = {}    # Last error message per camera
        self._camera_actual_res: Dict[str, Dict[str, Any]] = {}  # Actual connected resolution
        self._capabilities_cache: Dict[str, Dict[str, Any]] = {}  # "{type}:{device_id}" -> {data, timestamp}
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()        # Serializes connect attempts (USB contention)

        # Health monitor
        self._health_stop_event = threading.Event()
        self._health_thread: threading.Thread | None = None
        self._reconnect_attempts: Dict[str, int] = {}  # camera_key -> consecutive failure count
        self._health_failure_counts: Dict[str, int] = {}  # camera_key -> consecutive health check failures
        self._connect_timestamps: Dict[str, float] = {}  # camera_key -> time.time() of last connect
        self._HEALTH_GRACE_PERIOD = 30.0  # seconds after connect before health checks apply
        self._HEALTH_CONSECUTIVE_FAILURES = 3  # consecutive async_read failures before unhealthy

    @property
    def cameras(self) -> Dict[str, Any]:
        """Returns dict of currently connected camera instances."""
        with self._lock:
            return dict(self._cameras)

    def connect_camera(self, camera_key: str) -> Dict[str, Any]:
        """
        Connect a single camera by its config key (e.g. 'camera_1').
        Retries up to 3 times with exponential backoff on transient failures.
        Serialized via _connect_lock to prevent USB contention.
        """
        with self._connect_lock:
            return self._connect_camera_inner(camera_key)

    def _connect_camera_inner(self, camera_key: str) -> Dict[str, Any]:
        """Inner connection logic with retry. Caller MUST hold _connect_lock."""
        config = self.get_camera_config()
        if camera_key not in config:
            return {"status": "error", "message": f"Camera '{camera_key}' not found in config."}

        # Disconnect existing instance if any
        if camera_key in self._cameras:
            try:
                self._cameras[camera_key].disconnect()
            except Exception:
                pass
            with self._lock:
                del self._cameras[camera_key]

        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type", "opencv")

        if cam_type not in ("opencv", "intelrealsense"):
            self._camera_status[camera_key] = "error"
            self._camera_errors[camera_key] = f"Unsupported camera type: {cam_type}"
            return {"status": "error", "message": f"Unsupported camera type: {cam_type}"}

        last_error = None
        for attempt in range(1 + self._MAX_CONNECT_RETRIES):
            try:
                if cam_type == "opencv":
                    camera = self._connect_opencv_camera(camera_key, cam_cfg)
                else:
                    camera = self._connect_realsense_camera(camera_key, cam_cfg)

                with self._lock:
                    self._cameras[camera_key] = camera
                self._camera_status[camera_key] = "connected"
                self._camera_errors[camera_key] = ""
                self._reconnect_attempts.pop(camera_key, None)
                self._health_failure_counts.pop(camera_key, None)
                self._connect_timestamps[camera_key] = time.time()
                # Invalidate capabilities cache so frontend re-probes at new resolution
                cam_cfg_for_cache = config[camera_key]
                for prefix in ("opencv", "intelrealsense"):
                    for field in ("index_or_path", "serial_number_or_name", "video_device_id"):
                        val = cam_cfg_for_cache.get(field)
                        if val is not None:
                            self._capabilities_cache.pop(f"{prefix}:{val}", None)
                if attempt > 0:
                    logger.info(f"Camera '{camera_key}' ({cam_type}) connected on retry {attempt}.")
                else:
                    logger.info(f"Camera '{camera_key}' ({cam_type}) connected successfully.")
                return {"status": "connected", "camera_key": camera_key}

            except (ConnectionError, RuntimeError) as e:
                last_error = e
                if attempt < self._MAX_CONNECT_RETRIES:
                    delay = self._BACKOFF_DELAYS[attempt]
                    logger.warning(
                        f"Camera '{camera_key}' connect attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    gc.collect()
                    time.sleep(self._USB_SETTLE_DELAY)
                    time.sleep(delay)
            except Exception as e:
                # Non-retryable error
                last_error = e
                break

        # All retries exhausted
        error_msg = self._enrich_error_message(str(last_error), cam_type)
        self._camera_status[camera_key] = "error"
        self._camera_errors[camera_key] = error_msg
        logger.error(
            f"Failed to connect camera '{camera_key}' after "
            f"{min(attempt + 1, self._MAX_CONNECT_RETRIES + 1)} attempt(s): {error_msg}"
        )
        return {"status": "error", "message": error_msg}

    @staticmethod
    def _enrich_error_message(error: str, cam_type: str) -> str:
        """Append actionable suggestions to common camera error messages."""
        error_lower = error.lower()

        if cam_type == "intelrealsense":
            if "resolve requests" in error_lower or "couldn't resolve" in error_lower:
                return (
                    f"{error} | Suggestion: Try reducing resolution to 640x480 "
                    "or disconnecting other USB3 devices to free bandwidth."
                )
        if cam_type == "opencv":
            if "can't open camera" in error_lower or "failed to open" in error_lower:
                return (
                    f"{error} | Suggestion: Camera device may be in use by another "
                    "process. Run `lsof /dev/video0` to check."
                )

        return error

    def _connect_opencv_camera(self, camera_key: str, cam_cfg: dict) -> OpenCVCamera:
        """Connect an OpenCV camera, with fps/resolution auto-detect and discovery fallbacks."""
        from lerobot.cameras.opencv import OpenCVCameraConfig

        idx = cam_cfg.get("index_or_path")
        fps = cam_cfg.get("fps", 30)
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)

        # Attempt 1: Try with configured fps/width/height
        try:
            c_conf = OpenCVCameraConfig(
                index_or_path=idx,
                fps=fps,
                width=width,
                height=height,
            )
            camera = OpenCVCamera(c_conf)
            camera.connect(warmup=True)
            self._camera_actual_res[camera_key] = {
                "width": camera.width, "height": camera.height, "fps": camera.fps,
            }
            return camera
        except RuntimeError as fps_err:
            # FPS or resolution validation failed — camera opened but can't match settings
            logger.warning(
                f"Camera '{camera_key}': configured {width}x{height}@{fps}fps failed "
                f"({fps_err}), retrying with auto-detect..."
            )
            try:
                del camera
            except NameError:
                pass
            gc.collect()
        except ConnectionError as conn_err:
            # Camera can't be opened at all — skip to discovery
            logger.warning(f"Camera '{camera_key}': path '{idx}' can't be opened ({conn_err}), trying discovery...")
            # Jump straight to discovery (auto-detect won't help if device can't open)
            return self._opencv_discovery_fallback(camera_key, idx, fps, width, height)

        # Attempt 2: Auto-detect fps/resolution (let camera pick native capabilities)
        try:
            c_conf = OpenCVCameraConfig(
                index_or_path=idx,
                fps=None,
                width=None,
                height=None,
            )
            camera = OpenCVCamera(c_conf)
            camera.connect(warmup=True)
            self._camera_actual_res[camera_key] = {
                "width": camera.width, "height": camera.height, "fps": camera.fps,
            }
            logger.info(
                f"Camera '{camera_key}': auto-detect connected at "
                f"{camera.width}x{camera.height}@{camera.fps}fps"
            )
            return camera
        except (ConnectionError, RuntimeError) as second_err:
            logger.warning(f"Camera '{camera_key}': auto-detect also failed ({second_err}), trying discovery...")
            try:
                del camera
            except NameError:
                pass
            gc.collect()

        # Attempt 3: Auto-discovery fallback
        return self._opencv_discovery_fallback(camera_key, idx, fps, width, height)

    def _opencv_discovery_fallback(self, camera_key: str, idx, fps, width, height) -> OpenCVCamera:
        """Find available OpenCV cameras via discovery and connect to the first one."""
        from lerobot.cameras.opencv import OpenCVCameraConfig

        from app.core.cameras.discovery import discover_cameras
        discovered = discover_cameras(opencv_only=True)
        opencv_devices = discovered.get("opencv", [])

        if not opencv_devices:
            raise ConnectionError(
                f"Configured path '{idx}' failed and no USB webcams were discovered. "
                "Check that a webcam is plugged in."
            )

        # Filter out devices already connected by this service
        connected_paths = set()
        for key, cam in self._cameras.items():
            if hasattr(cam, 'index_or_path'):
                connected_paths.add(str(cam.index_or_path))

        available = [d for d in opencv_devices if d.get("index_or_path") not in connected_paths]

        if not available:
            raise ConnectionError(
                f"Configured path '{idx}' failed and all discovered webcams are already in use."
            )

        # Use the first available device
        new_path = available[0]["index_or_path"]
        logger.info(f"Camera '{camera_key}': auto-discovered device at '{new_path}', connecting...")

        c_conf = OpenCVCameraConfig(
            index_or_path=new_path,
            fps=fps,
            width=width,
            height=height,
        )
        camera = OpenCVCamera(c_conf)
        camera.connect(warmup=True)
        self._camera_actual_res[camera_key] = {
            "width": camera.width, "height": camera.height, "fps": camera.fps,
        }

        # Update settings.yaml with the correct path so next time it works directly
        self._update_camera_path(camera_key, new_path)

        return camera

    def _connect_realsense_camera(self, camera_key: str, cam_cfg: dict) -> RealSenseCamera:
        """Connect a RealSense camera by serial number, with resolution auto-detect fallback."""
        from lerobot.cameras.realsense import RealSenseCameraConfig

        serial = cam_cfg.get("serial_number_or_name")
        fps = cam_cfg.get("fps", 30)
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)
        use_depth = cam_cfg.get("use_depth", False)
        exposure = cam_cfg.get("exposure")
        gain = cam_cfg.get("gain")
        brightness = cam_cfg.get("brightness")

        # Attempt 1: Try with configured resolution
        try:
            c_conf = RealSenseCameraConfig(
                serial_number_or_name=serial,
                fps=fps,
                width=width,
                height=height,
                use_depth=use_depth,
                exposure=exposure,
                gain=gain,
                brightness=brightness,
            )
            camera = RealSenseCamera(c_conf)
            camera.connect(warmup=True)
            self._camera_actual_res[camera_key] = {
                "width": camera.width, "height": camera.height, "fps": camera.fps,
            }
            self._prime_realsense(camera, camera_key)
            return camera

        except (ConnectionError, RuntimeError) as first_err:
            cause = first_err.__cause__ if first_err.__cause__ else first_err
            logger.warning(
                f"Camera '{camera_key}': configured {width}x{height}@{fps}fps failed "
                f"(pyrealsense2: {cause}), retrying with auto-detect resolution..."
            )
            # Release failed camera's USB handles before retry
            try:
                del camera
            except NameError:
                pass
            gc.collect()

        # Attempt 2: Auto-detect resolution (let camera pick native defaults)
        c_conf = RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=None,
            width=None,
            height=None,
            use_depth=use_depth,
            exposure=exposure,
            gain=gain,
            brightness=brightness,
        )
        camera = RealSenseCamera(c_conf)
        camera.connect(warmup=True)
        self._camera_actual_res[camera_key] = {
            "width": camera.width, "height": camera.height, "fps": camera.fps,
        }

        logger.info(
            f"Camera '{camera_key}': auto-detect connected at "
            f"{camera.width}x{camera.height}@{camera.fps}fps"
        )
        self._prime_realsense(camera, camera_key)
        return camera

    @staticmethod
    def _prime_realsense(camera: RealSenseCamera, camera_key: str):
        """Start the background read thread and prime the frame cache for a RealSense camera.

        RealSense connect() does NOT start the background thread (unlike OpenCV).
        We start it explicitly so async_read(blocking=False) has frames immediately.
        """
        # Ensure background thread is running
        if not (camera.thread and camera.thread.is_alive()):
            camera._start_read_thread()

        # Prime: wait for the first frame to populate latest_frame
        try:
            camera.async_read(blocking=True, timeout_ms=5000)
        except Exception as e:
            logger.warning(f"Camera '{camera_key}': initial frame prime failed ({e}), stream may take a moment")

        # Verify frame cache is populated
        with camera.frame_lock:
            if camera.latest_frame is None:
                logger.warning(f"Camera '{camera_key}': latest_frame still None after prime")

    def set_camera_exposure(
        self,
        camera_key: str,
        exposure: int | None = None,
        gain: int | None = None,
        brightness: int | None = None,
    ) -> Dict[str, Any]:
        """Adjust exposure/gain/brightness on a connected RealSense camera.
        Persists settings to settings.yaml so they survive reconnect.
        """
        with self._lock:
            camera = self._cameras.get(camera_key)

        if camera is None:
            return {"status": "error", "message": f"Camera '{camera_key}' not connected"}

        if not isinstance(camera, RealSenseCamera):
            return {"status": "error", "message": "Exposure control only supported for RealSense cameras"}

        result = camera.set_exposure(exposure=exposure, gain=gain, brightness=brightness)

        if result.get("status") == "ok":
            # Persist to settings.yaml
            try:
                full_config = load_config()
                cameras_cfg = full_config.get("robot", {}).get("cameras", {})
                if camera_key in cameras_cfg:
                    if exposure is not None:
                        cameras_cfg[camera_key]["exposure"] = exposure
                    if gain is not None:
                        cameras_cfg[camera_key]["gain"] = gain
                    if brightness is not None:
                        cameras_cfg[camera_key]["brightness"] = brightness
                    save_config(full_config)
                    logger.info(f"Persisted exposure settings for '{camera_key}' to settings.yaml")
            except Exception as e:
                logger.warning(f"Failed to persist exposure settings for '{camera_key}': {e}")

        return result

    def get_camera_exposure_info(self, camera_key: str) -> Dict[str, Any]:
        """Get current exposure settings and supported ranges for a connected RealSense camera."""
        with self._lock:
            camera = self._cameras.get(camera_key)

        if camera is None:
            return {"status": "error", "message": f"Camera '{camera_key}' not connected"}

        if not isinstance(camera, RealSenseCamera):
            return {"status": "error", "message": "Exposure info only available for RealSense cameras"}

        return camera.get_exposure_info()

    def _update_camera_path(self, camera_key: str, new_path: str):
        """Update the device path for an OpenCV camera in settings.yaml."""
        try:
            full_config = load_config()
            cameras_cfg = full_config.get("robot", {}).get("cameras", {})
            if camera_key in cameras_cfg:
                cameras_cfg[camera_key]["index_or_path"] = new_path
                cameras_cfg[camera_key]["video_device_id"] = new_path
                save_config(full_config)
                logger.info(f"Updated settings.yaml: {camera_key} path → {new_path}")
        except Exception as e:
            logger.warning(f"Failed to update settings.yaml for {camera_key}: {e}")

    def disconnect_camera(self, camera_key: str) -> Dict[str, Any]:
        """Disconnect a single camera by key (public API)."""
        return self._disconnect_camera_internal(camera_key)

    def _disconnect_camera_internal(self, camera_key: str) -> Dict[str, Any]:
        """Inner disconnect logic. Callable by health monitor without extra locking."""
        with self._lock:
            camera = self._cameras.pop(camera_key, None)

        if camera:
            try:
                camera.disconnect()
                logger.info(f"Camera '{camera_key}' disconnected.")
            except Exception as e:
                logger.warning(f"Error disconnecting camera '{camera_key}': {e}")

        self._camera_status[camera_key] = "disconnected"
        self._camera_errors[camera_key] = ""
        self._camera_actual_res.pop(camera_key, None)
        self._health_failure_counts.pop(camera_key, None)
        self._connect_timestamps.pop(camera_key, None)
        return {"status": "disconnected", "camera_key": camera_key}

    def disconnect_all(self):
        """Disconnect all managed cameras. Called during shutdown."""
        keys = list(self._cameras.keys())
        for key in keys:
            self.disconnect_camera(key)

    def connect_all_cameras(self) -> Dict[str, Any]:
        """
        Connect all configured cameras sequentially.
        Order: OpenCV cameras first, then RealSense sorted by serial number.
        Inserts a 2s gap between each camera to avoid USB contention.
        """
        config = self.get_camera_config()
        if not config:
            return {"status": "ok", "message": "No cameras configured.", "results": {}}

        # Partition into OpenCV and RealSense, sort RealSense by serial
        opencv_keys = []
        realsense_keys = []
        for key, cfg in config.items():
            cam_type = cfg.get("type", "opencv")
            if cam_type == "intelrealsense":
                realsense_keys.append(key)
            else:
                opencv_keys.append(key)

        realsense_keys.sort(key=lambda k: config[k].get("serial_number_or_name", ""))
        ordered_keys = opencv_keys + realsense_keys
        results = {}

        for i, camera_key in enumerate(ordered_keys):
            logger.info(f"connect_all_cameras: [{i+1}/{len(ordered_keys)}] connecting '{camera_key}'...")
            result = self.connect_camera(camera_key)
            results[camera_key] = result

            # Inter-camera gap (skip after the last one)
            if i < len(ordered_keys) - 1:
                time.sleep(2.0)

        connected = sum(1 for r in results.values() if r.get("status") == "connected")
        failed = len(results) - connected
        return {
            "status": "ok" if failed == 0 else "partial",
            "message": f"{connected}/{len(results)} cameras connected.",
            "results": results,
        }

    # ── Health Monitor ────────────────────────────────────────────────────

    def start_health_monitor(self):
        """Start the background health monitor thread."""
        if self._health_thread is not None and self._health_thread.is_alive():
            logger.warning("Camera health monitor already running.")
            return
        self._health_stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="CameraHealthMonitor",
        )
        self._health_thread.start()
        logger.info("Camera health monitor started.")

    def stop_health_monitor(self):
        """Signal the health monitor thread to stop and wait for it."""
        self._health_stop_event.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=15.0)
            if self._health_thread.is_alive():
                logger.warning("Camera health monitor did not stop within timeout.")
            self._health_thread = None
        self._reconnect_attempts.clear()
        logger.info("Camera health monitor stopped.")

    def _health_monitor_loop(self):
        """Periodically check all connected cameras and auto-reconnect stale ones."""
        logger.info("Camera health monitor thread running.")
        while not self._health_stop_event.is_set():
            if self._health_stop_event.wait(timeout=self._HEALTH_CHECK_INTERVAL):
                break  # stop requested
            try:
                self._health_check_cycle()
            except Exception as e:
                logger.error(f"Health monitor cycle error: {e}")
        logger.info("Camera health monitor thread exiting.")

    def _health_check_cycle(self):
        """Single health check pass over all connected cameras."""
        now = time.time()
        for camera_key in list(self._cameras.keys()):
            status = self._camera_status.get(camera_key, "disconnected")
            if status != "connected":
                continue

            cam = self._cameras.get(camera_key)
            if cam is None:
                continue

            # Grace period: skip cameras that just connected (allow stabilization)
            connect_time = self._connect_timestamps.get(camera_key, 0)
            if (now - connect_time) < self._HEALTH_GRACE_PERIOD:
                continue

            healthy = self._is_camera_healthy(camera_key, cam)
            if healthy:
                self._reconnect_attempts.pop(camera_key, None)
                self._health_failure_counts.pop(camera_key, None)
                continue

            # Camera is unhealthy — attempt auto-reconnect
            attempts = self._reconnect_attempts.get(camera_key, 0)
            if attempts >= self._MAX_AUTO_RECONNECT:
                self._camera_status[camera_key] = "error"
                self._camera_errors[camera_key] = (
                    "Auto-reconnect failed. Manual reconnect required."
                )
                logger.error(
                    f"Camera '{camera_key}': auto-reconnect exhausted "
                    f"({self._MAX_AUTO_RECONNECT} attempts). Manual reconnect required."
                )
                continue

            # Try to acquire _connect_lock without blocking
            acquired = self._connect_lock.acquire(blocking=False)
            if not acquired:
                logger.debug(
                    f"Camera '{camera_key}': skipping auto-reconnect, "
                    "manual operation in progress."
                )
                continue

            try:
                self._reconnect_attempts[camera_key] = attempts + 1
                logger.warning(
                    f"Camera '{camera_key}': unhealthy, auto-reconnect "
                    f"attempt {attempts + 1}/{self._MAX_AUTO_RECONNECT}..."
                )
                self._disconnect_camera_internal(camera_key)
                gc.collect()
                time.sleep(1.0)
                self._connect_camera_inner(camera_key)
            except Exception as e:
                logger.error(f"Auto-reconnect failed for '{camera_key}': {e}")
                self._camera_status[camera_key] = "error"
                self._camera_errors[camera_key] = f"Auto-reconnect failed: {e}"
            finally:
                self._connect_lock.release()

    def _is_camera_healthy(self, camera_key: str, cam) -> bool:
        """Check if a camera is still operational.

        Thread death is immediately unhealthy. Frame read failures require
        consecutive failures (self._HEALTH_CONSECUTIVE_FAILURES) before
        declaring unhealthy, to tolerate brief USB congestion.
        """
        if not getattr(cam, 'is_connected', False):
            logger.warning(f"Camera '{camera_key}': is_connected=False")
            self._health_failure_counts.pop(camera_key, None)
            return False

        # Thread death is immediately unhealthy (no counter needed)
        if hasattr(cam, 'thread') and cam.thread is not None:
            if not cam.thread.is_alive():
                logger.warning(f"Camera '{camera_key}': background thread dead")
                self._health_failure_counts.pop(camera_key, None)
                return False

        # Frame read — use consecutive failure counter
        try:
            frame = cam.async_read(blocking=False)
            if frame is None:
                count = self._health_failure_counts.get(camera_key, 0) + 1
                self._health_failure_counts[camera_key] = count
                if count >= self._HEALTH_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"Camera '{camera_key}': async_read returned None "
                        f"{count} consecutive times — marking unhealthy"
                    )
                    return False
                logger.debug(
                    f"Camera '{camera_key}': async_read returned None "
                    f"({count}/{self._HEALTH_CONSECUTIVE_FAILURES})"
                )
                return True  # Not yet unhealthy
        except Exception as e:
            count = self._health_failure_counts.get(camera_key, 0) + 1
            self._health_failure_counts[camera_key] = count
            if count >= self._HEALTH_CONSECUTIVE_FAILURES:
                logger.warning(f"Camera '{camera_key}': async_read failed {count} times: {e}")
                return False
            return True  # Not yet unhealthy

        # Success — reset failure counter
        self._health_failure_counts.pop(camera_key, None)
        return True

    # ── Capabilities Probing ─────────────────────────────────────────────

    _COMMON_RESOLUTIONS = [
        (3840, 2160, "4K"),
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (640, 480, "480p"),
        (320, 240, "QVGA"),
    ]

    def get_capabilities(self, device_type: str, device_id: str) -> Dict[str, Any]:
        """
        Probe supported resolutions for a camera device.
        If the camera is already connected, returns its actual resolution without
        re-opening the device (non-destructive). Results cached for 60s.
        """
        cache_key = f"{device_type}:{device_id}"
        cached = self._capabilities_cache.get(cache_key)
        if cached and (time.time() - cached["timestamp"]) < 60.0:
            return cached["data"]

        # Check if camera is already connected — return actual resolution only
        config = self.get_camera_config()
        for cam_key, cam_cfg in config.items():
            matches = False
            if device_type == "opencv" and cam_cfg.get("type") == "opencv":
                matches = str(cam_cfg.get("index_or_path")) == str(device_id) or \
                          str(cam_cfg.get("video_device_id")) == str(device_id)
            elif device_type == "intelrealsense" and cam_cfg.get("type") == "intelrealsense":
                matches = str(cam_cfg.get("serial_number_or_name")) == str(device_id)

            if matches and cam_key in self._cameras:
                actual = self._camera_actual_res.get(cam_key, {})
                if actual:
                    result = {
                        "resolutions": [{
                            "width": actual["width"],
                            "height": actual["height"],
                            "fps": [actual.get("fps", 30)],
                        }],
                        "native": {"width": actual["width"], "height": actual["height"]},
                        "connected": True,
                    }
                    self._capabilities_cache[cache_key] = {"data": result, "timestamp": time.time()}
                    return result

        # Probe device
        if device_type == "opencv":
            result = self._probe_opencv_capabilities(device_id)
        elif device_type == "intelrealsense":
            result = self._probe_realsense_capabilities(device_id)
        else:
            result = {"resolutions": [], "native": None, "connected": False}

        self._capabilities_cache[cache_key] = {"data": result, "timestamp": time.time()}
        return result

    def _probe_opencv_capabilities(self, device_id: str) -> Dict[str, Any]:
        """Open cv2.VideoCapture, try common resolutions, report which ones work."""
        import cv2

        try:
            idx = device_id if device_id.startswith('/') else int(device_id)
        except ValueError:
            idx = device_id

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return {"resolutions": [], "native": None, "connected": False}

        try:
            # Get native resolution (camera default)
            native_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            native_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            native = {"width": native_w, "height": native_h}

            supported = []
            for w, h, label in self._COMMON_RESOLUTIONS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if actual_w == w and actual_h == h:
                    # Probe FPS at this resolution
                    fps_list = []
                    for target_fps in [15, 30, 60]:
                        cap.set(cv2.CAP_PROP_FPS, target_fps)
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        if abs(actual_fps - target_fps) < 2:
                            fps_list.append(target_fps)
                    if not fps_list:
                        fps_list = [30]
                    supported.append({"width": w, "height": h, "fps": fps_list, "label": label})

            # Add native if not in common list
            if not any(r["width"] == native_w and r["height"] == native_h for r in supported):
                supported.append({"width": native_w, "height": native_h, "fps": [30], "label": "Native"})

            return {"resolutions": supported, "native": native, "connected": False}
        except Exception as e:
            logger.warning(f"OpenCV capabilities probe failed: {e}")
            return {"resolutions": [], "native": None, "connected": False}
        finally:
            cap.release()

    def _probe_realsense_capabilities(self, serial: str) -> Dict[str, Any]:
        """Query RealSense stream profiles for supported color resolutions."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            return {"resolutions": [], "native": None, "connected": False}

        ctx = rs.context()
        devices = ctx.query_devices()
        device = None
        for dev in devices:
            try:
                if dev.get_info(rs.camera_info.serial_number) == str(serial):
                    device = dev
                    break
            except Exception:
                continue

        if device is None:
            return {"resolutions": [], "native": None, "connected": False}

        res_fps_map: Dict[tuple, set] = {}
        native = None

        try:
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()
                for profile in profiles:
                    if not profile.is_video_stream_profile():
                        continue
                    vp = profile.as_video_stream_profile()
                    if vp.stream_type() != rs.stream.color:
                        continue
                    fmt = vp.format()
                    if fmt != rs.format.rgb8 and fmt != rs.format.bgr8 and fmt != rs.format.yuyv:
                        continue
                    w, h, fps = vp.width(), vp.height(), vp.fps()
                    key = (w, h)
                    if key not in res_fps_map:
                        res_fps_map[key] = set()
                    res_fps_map[key].add(fps)

                    if profile.is_default() and native is None:
                        native = {"width": w, "height": h}
        except Exception as e:
            logger.warning(f"RealSense capabilities probe failed: {e}")
            return {"resolutions": [], "native": None, "connected": False}

        supported = []
        for (w, h), fps_set in sorted(res_fps_map.items(), key=lambda x: x[0][0] * x[0][1], reverse=True):
            supported.append({
                "width": w,
                "height": h,
                "fps": sorted(fps_set),
            })

        return {"resolutions": supported, "native": native, "connected": False}

    # ── Status & Resolution ────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """
        Returns connection status for all configured cameras.
        Detects if a connected camera's background thread has died.
        """
        config = self.get_camera_config()
        status = {}
        for camera_key in config:
            cam = self._cameras.get(camera_key)
            res = self._camera_actual_res.get(camera_key, {})
            if cam and getattr(cam, 'is_connected', False):
                # Check if background thread is still alive
                thread_alive = hasattr(cam, 'thread') and cam.thread is not None and cam.thread.is_alive()
                if thread_alive:
                    status[camera_key] = {
                        "status": "connected", "error": "",
                        "actual_width": res.get("width"),
                        "actual_height": res.get("height"),
                        "actual_fps": res.get("fps"),
                    }
                else:
                    status[camera_key] = {
                        "status": "error", "error": "Background read thread stopped",
                        "actual_width": res.get("width"),
                        "actual_height": res.get("height"),
                        "actual_fps": res.get("fps"),
                    }
                    self._camera_status[camera_key] = "error"
                    self._camera_errors[camera_key] = "Background read thread stopped"
            else:
                status[camera_key] = {
                    "status": self._camera_status.get(camera_key, "disconnected"),
                    "error": self._camera_errors.get(camera_key, ""),
                    "actual_width": None,
                    "actual_height": None,
                    "actual_fps": None,
                }
        # Health monitor info
        status["_health_monitor"] = {
            "running": self._health_thread is not None and self._health_thread.is_alive(),
            "reconnect_attempts": dict(self._reconnect_attempts),
        }
        return status

    def get_camera_resolution(self, camera_key: str) -> tuple:
        """Return (width, height) — actual if connected, configured otherwise, 640x480 fallback."""
        res = self._camera_actual_res.get(camera_key)
        if res and res.get("width") and res.get("height"):
            return res["width"], res["height"]
        config = self.get_camera_config()
        if camera_key in config:
            return config[camera_key].get("width", 640), config[camera_key].get("height", 480)
        return 640, 480

    # ── Legacy methods (unchanged) ────────────────────────────────────────

    def scan_cameras(self, active_ids: List[str] = None, force: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scans for available OpenCV and RealSense cameras.
        Returns a dict with 'opencv' and 'realsense' lists.
        Filters out devices that cannot be opened or read.
        Results are cached for 30s unless force=True.
        """
        if active_ids is None:
            active_ids = []

        results = {
            "opencv": [],
            "realsense": []
        }

        from app.core.cameras.discovery import discover_cameras

        try:
            logger.info("Scanning for cameras using unified discovery...")
            discovered = discover_cameras(skip_devices=active_ids, force=force)
            results["opencv"] = discovered.get("opencv", [])
            results["realsense"] = discovered.get("realsense", [])
        except Exception as e:
            logger.error(f"Error during unified camera scan: {e}")

        return results

    def capture_snapshot(self, camera_key: str):
        """
        Capture a single frame from a camera.
        Prefers using a connected managed camera (fast path).
        Falls back to one-shot open/read/close for OpenCV cameras.
        """
        # Fast path: use managed connected camera
        cam = self._cameras.get(camera_key)
        if cam and getattr(cam, 'is_connected', False):
            try:
                frame = cam.async_read(blocking=False)
                if frame is not None:
                    return frame
            except Exception:
                pass

        # Slow path: one-shot capture (OpenCV only, RealSense needs persistent connection)
        config = self.get_camera_config()
        if camera_key not in config:
            return None

        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type", "opencv")

        try:
            import cv2
            if cam_type == "opencv":
                idx = cam_cfg.get("index_or_path")
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        return frame
        except Exception as e:
            logger.error(f"Snapshot failed for {camera_key}: {e}")

        return None

    def get_camera_config(self) -> Dict[str, Any]:
        """Returns the current camera configuration from settings.yaml."""
        config = load_config()
        raw_cameras = config.get("robot", {}).get("cameras", {})

        # Normalize List to Dict if needed
        if isinstance(raw_cameras, list):
            normalized = {}
            for c in raw_cameras:
                c_id = c.get("id", "unknown")
                vid = c.get("video_device_id")
                c_type = c.get("type", "opencv" if (str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4)) else "intelrealsense")
                normalized[c_id] = {
                    "type": c_type,
                    "index_or_path": vid,
                    "serial_number_or_name": vid,
                    **c
                }
            return normalized

        # Normalize dict format too (ensure index_or_path/serial_number_or_name exist)
        if isinstance(raw_cameras, dict):
            for cam_id, cam_cfg in raw_cameras.items():
                vid = cam_cfg.get("video_device_id")
                cam_type = cam_cfg.get("type", "opencv")

                if cam_type == "opencv" and "index_or_path" not in cam_cfg:
                    cam_cfg["index_or_path"] = vid

                if cam_type == "intelrealsense" and "serial_number_or_name" not in cam_cfg:
                    cam_cfg["serial_number_or_name"] = vid

        return raw_cameras

    def update_camera_config(self, new_cameras_config: Dict[str, Any]):
        """
        Updates the camera configuration in settings.yaml.
        new_cameras_config: Dict mapping camera_key (e.g. 'camera_1') to config dict.
        """
        config = load_config()
        if "robot" not in config:
            config["robot"] = {}

        config["robot"]["cameras"] = new_cameras_config
        save_config(config)
        logger.info("Camera configuration updated.")

    def test_camera(self, camera_key: str) -> Dict[str, Any]:
        """Tests if a configured camera can be opened and read."""
        config = self.get_camera_config()
        if camera_key not in config:
            return {"status": "error", "message": f"Camera {camera_key} not found in config."}

        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type")

        try:
            camera = None
            if cam_type == "opencv":
                from lerobot.cameras.opencv import OpenCVCameraConfig
                c_conf = OpenCVCameraConfig(
                    index_or_path=cam_cfg.get("index_or_path"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480)
                )
                camera = OpenCVCamera(c_conf)

            elif cam_type == "intelrealsense":
                from lerobot.cameras.realsense import RealSenseCameraConfig
                c_conf = RealSenseCameraConfig(
                    serial_number_or_name=cam_cfg.get("serial_number_or_name"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480)
                )
                camera = RealSenseCamera(c_conf)

            if camera:
                camera.connect()
                frame = camera.read()
                camera.disconnect()

                if frame is not None:
                    return {"status": "success", "message": "Camera connected and frame read successfully."}
                else:
                    return {"status": "error", "message": "Camera connected but returned empty frame."}
            else:
                return {"status": "error", "message": f"Unsupported camera type: {cam_type}"}

        except Exception as e:
            return {"status": "error", "message": f"Failed to connect: {str(e)}"}

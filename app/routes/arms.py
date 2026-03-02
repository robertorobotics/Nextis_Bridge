import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.hardware.types import ArmRole, ConnectionStatus
from app.dependencies import get_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["arms"])


# --- Arm Registry Endpoints ---
# NOTE: Static routes (/arms/pairings, /arms/scan-ports) MUST come before
# parameterized routes (/arms/{arm_id}) to avoid routing conflicts in FastAPI.

@router.get("/arms")
async def get_all_arms():
    """Get all registered arms with their status."""
    system = get_state()
    if not system.arm_registry:
        return {"arms": [], "summary": {}}
    return {
        "arms": system.arm_registry.get_all_arms(),
        "summary": system.arm_registry.get_status_summary()
    }

@router.post("/arms")
async def add_arm(request: Request):
    """Add a new arm to the registry."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.add_arm(data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

# Static routes - must be defined before /{arm_id} routes
@router.get("/arms/pairings")
async def get_pairings():
    """Get all leader-follower pairings."""
    system = get_state()
    if not system.arm_registry:
        return {"pairings": []}
    return {"pairings": system.arm_registry.get_pairings()}

@router.post("/arms/pairings")
async def create_pairing(request: Request):
    """Create a new leader-follower pairing."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.create_pairing(
        leader_id=data.get("leader_id"),
        follower_id=data.get("follower_id"),
        name=data.get("name")
    )
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.delete("/arms/pairings")
async def remove_pairing(request: Request):
    """Remove a leader-follower pairing."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.remove_pairing(
        leader_id=data.get("leader_id"),
        follower_id=data.get("follower_id")
    )
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.get("/arms/scan-ports")
async def scan_ports():
    """Scan for available serial ports."""
    system = get_state()
    if not system.arm_registry:
        return {"ports": []}
    return {"ports": system.arm_registry.scan_ports()}

# Parameterized routes - must come after static routes
@router.get("/arms/{arm_id}")
async def get_arm(arm_id: str):
    """Get details of a specific arm."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=404, content={"error": "Arm registry not initialized"})
    arm = system.arm_registry.get_arm(arm_id)
    if not arm:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})
    return arm

@router.put("/arms/{arm_id}")
async def update_arm(arm_id: str, request: Request):
    """Update an existing arm."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.update_arm(arm_id, **data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.delete("/arms/{arm_id}")
async def remove_arm(arm_id: str):
    """Remove an arm from the registry."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.remove_arm(arm_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.post("/arms/{arm_id}/connect")
def connect_arm(arm_id: str):
    """Connect a specific arm. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial port connection and motor configuration."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.connect_arm(arm_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    # Start safety watchdog if a follower arm just connected
    if system.safety_watchdog and not system.safety_watchdog.is_running:
        for aid, status in system.arm_registry.arm_status.items():
            arm_def = system.arm_registry.arms.get(aid)
            if (status == ConnectionStatus.CONNECTED
                    and arm_def and arm_def.role == ArmRole.FOLLOWER):
                system.safety_watchdog.start()
                logger.info("Safety watchdog started (follower arm connected)")
                break

    return result

@router.post("/arms/{arm_id}/disconnect")
def disconnect_arm(arm_id: str):
    """Disconnect a specific arm. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial port disconnection."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.disconnect_arm(arm_id)

    # Stop safety watchdog if no follower arms remain connected
    if system.safety_watchdog and system.safety_watchdog.is_running:
        has_connected_follower = False
        for aid, status in system.arm_registry.arm_status.items():
            arm_def = system.arm_registry.arms.get(aid)
            if (status == ConnectionStatus.CONNECTED
                    and arm_def and arm_def.role == ArmRole.FOLLOWER):
                has_connected_follower = True
                break
        if not has_connected_follower:
            system.safety_watchdog.stop()
            logger.info("Safety watchdog stopped (no follower arms connected)")

    return result

@router.post("/arms/{arm_id}/set-home")
def set_home_position(arm_id: str):
    """Capture current motor positions as home position for this arm."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    instance = system.arm_registry.arm_instances.get(arm_id)
    if not instance or not getattr(instance, 'is_connected', False):
        return JSONResponse(status_code=400, content={"error": f"Arm '{arm_id}' not connected"})

    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
    bus = getattr(instance, 'bus', None)
    if not bus or not isinstance(bus, DamiaoMotorsBus):
        return JSONResponse(status_code=400, content={"error": "Not a Damiao arm"})

    try:
        positions = bus.sync_read("Present_Position")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read positions: {e}"})
    if not positions:
        return JSONResponse(status_code=400, content={"error": "No position data available"})

    result = system.arm_registry.update_arm(arm_id, config={"home_position": positions})
    if result.get("success"):
        return {"success": True, "home_position": {k: round(v, 4) for k, v in positions.items()}}
    return JSONResponse(status_code=400, content=result)

@router.delete("/arms/{arm_id}/set-home")
async def clear_home_position(arm_id: str):
    """Clear saved home position for this arm."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    arm = system.arm_registry.arms.get(arm_id)
    if not arm:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})
    arm.config.pop("home_position", None)
    system.arm_registry.save_config()
    return {"success": True}

@router.get("/arms/{arm_id}/motors/diagnostics")
def get_motor_diagnostics(arm_id: str):
    """Read live motor telemetry (position, temperature, voltage, current, errors) for a connected arm."""
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    arm_def = system.arm_registry.arms.get(arm_id)
    if not arm_def:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})

    instance = system.arm_registry.arm_instances.get(arm_id)
    if not instance or not getattr(instance, 'is_connected', False):
        return JSONResponse(status_code=400, content={"error": f"Arm '{arm_id}' not connected"})

    bus = getattr(instance, 'bus', None)
    if not bus:
        return JSONResponse(status_code=400, content={"error": "No motor bus available"})

    motor_type = arm_def.motor_type.value
    motors_info = []

    # Determine which registers to read based on motor type
    try:
        from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        is_dynamixel = isinstance(bus, DynamixelMotorsBus)
        is_feetech = isinstance(bus, FeetechMotorsBus)
    except ImportError:
        is_dynamixel = False
        is_feetech = False

    # Check for Damiao (CAN-based, different API)
    is_damiao = False
    try:
        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
        is_damiao = isinstance(bus, DamiaoMotorsBus)
    except ImportError:
        pass

    if is_damiao:
        # Damiao: read from CAN state cache (no sync_read for these)
        for name, motor in bus.motors.items():
            motor_data = {
                "name": name,
                "id": motor.id if hasattr(motor, 'id') else getattr(motor, 'slave_id', 0),
                "model": getattr(motor, 'model', 'damiao'),
                "position": None,
                "velocity": None,
                "current": None,
                "temperature": None,
                "voltage": None,
                "load": None,
                "error": 0,
                "error_names": [],
            }
            # Damiao stores last state in _last_positions, _last_velocities, _last_torques
            if hasattr(bus, '_last_positions'):
                motor_data["position"] = round(bus._last_positions.get(name, 0), 3)
            if hasattr(bus, '_last_velocities'):
                motor_data["velocity"] = round(bus._last_velocities.get(name, 0), 3)
            if hasattr(bus, '_last_torques'):
                motor_data["load"] = round(bus._last_torques.get(name, 0), 3)
            motors_info.append(motor_data)
    else:
        # Dynamixel / Feetech: use sync_read for each register
        motor_names = list(bus.motors.keys())

        # Build motor list with IDs
        motor_map = {}
        for name in motor_names:
            m = bus.motors[name]
            motor_map[name] = {
                "name": name,
                "id": m.id,
                "model": getattr(m, 'model', motor_type),
                "position": None,
                "velocity": None,
                "current": None,
                "temperature": None,
                "voltage": None,
                "load": None,
                "error": 0,
                "error_names": [],
            }

        def safe_read(data_name):
            try:
                return bus.sync_read(data_name, normalize=False)
            except Exception:
                return {}

        positions = safe_read("Present_Position")
        for name, val in positions.items():
            if name in motor_map:
                motor_map[name]["position"] = int(val) if val is not None else None

        velocities = safe_read("Present_Velocity")
        for name, val in velocities.items():
            if name in motor_map:
                motor_map[name]["velocity"] = int(val) if val is not None else None

        temperatures = safe_read("Present_Temperature")
        for name, val in temperatures.items():
            if name in motor_map:
                motor_map[name]["temperature"] = int(val) if val is not None else None

        # Current (Dynamixel) or Present_Current (Feetech)
        currents = safe_read("Present_Current")
        for name, val in currents.items():
            if name in motor_map:
                motor_map[name]["current"] = int(val) if val is not None else None

        # Voltage — different register names per type
        if is_dynamixel:
            voltages = safe_read("Present_Input_Voltage")
        else:
            voltages = safe_read("Present_Voltage")
        for name, val in voltages.items():
            if name in motor_map:
                motor_map[name]["voltage"] = round(int(val) * 0.1, 1) if val is not None else None

        # Load (Feetech only)
        if is_feetech:
            loads = safe_read("Present_Load")
            for name, val in loads.items():
                if name in motor_map:
                    motor_map[name]["load"] = int(val) if val is not None else None

        # Hardware errors (Dynamixel only)
        if is_dynamixel:
            errors = safe_read("Hardware_Error_Status")
            for name, val in errors.items():
                if name in motor_map and val:
                    err = int(val)
                    motor_map[name]["error"] = err
                    names = []
                    if err & 0x01:
                        names.append("Voltage")
                    if err & 0x04:
                        names.append("Overheat")
                    if err & 0x08:
                        names.append("Encoder")
                    if err & 0x10:
                        names.append("Shock")
                    if err & 0x20:
                        names.append("Overload")
                    motor_map[name]["error_names"] = names

        motors_info = list(motor_map.values())

    return {
        "arm_id": arm_id,
        "motor_type": motor_type,
        "motors": motors_info,
    }


@router.get("/arms/{leader_id}/compatible-followers")
async def get_compatible_followers(leader_id: str):
    """Get followers compatible with a leader arm."""
    system = get_state()
    if not system.arm_registry:
        return {"followers": []}
    return {"followers": system.arm_registry.get_compatible_followers(leader_id)}


# --- Damiao Velocity Limiter Endpoints ---

@router.get("/robot/velocity-limit")
async def get_velocity_limit():
    """Get current global velocity limit for Damiao robots.

    Returns velocity_limit as float (0.0-1.0) where 1.0 = 100% max velocity.
    """
    system = get_state()
    # Check if we have a Damiao robot
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        return {"velocity_limit": system.robot.velocity_limit, "has_velocity_limit": True}

    # Check teleop service's active robot (set during teleop start from arm registry)
    if system.teleop_service:
        active = getattr(system.teleop_service, '_active_robot', None)
        if active and hasattr(active, 'velocity_limit'):
            return {"velocity_limit": active.velocity_limit, "has_velocity_limit": True}
        robot = getattr(system.teleop_service, 'robot', None)
        if robot and hasattr(robot, 'velocity_limit'):
            return {"velocity_limit": robot.velocity_limit, "has_velocity_limit": True}

    # Check arm registry for any connected Damiao follower
    if system.arm_registry:
        for arm_id, instance in system.arm_registry.arm_instances.items():
            if hasattr(instance, 'velocity_limit'):
                return {"velocity_limit": instance.velocity_limit, "has_velocity_limit": True}

    return {"velocity_limit": 1.0, "has_velocity_limit": False}

@router.post("/robot/velocity-limit")
async def set_velocity_limit(request: Request):
    """Set global velocity limit for Damiao robots (0.0-1.0).

    SAFETY: This limits the maximum velocity of ALL motor commands.
    Default for Damiao is 0.2 (20%) for safety with high-torque motors.
    """
    system = get_state()
    data = await request.json()
    limit = float(data.get("limit", 1.0))
    limit = max(0.0, min(1.0, limit))

    updated = False

    # Update main robot if it's a Damiao
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        system.robot.velocity_limit = limit
        updated = True
        logger.info(f"Set velocity_limit to {limit:.2f} on main robot")

    # Update teleop service's active robot and legacy robot
    if system.teleop_service:
        active = getattr(system.teleop_service, '_active_robot', None)
        if active and hasattr(active, 'velocity_limit'):
            active.velocity_limit = limit
            updated = True
            logger.info(f"Set velocity_limit to {limit:.2f} on teleop active robot")
        robot = getattr(system.teleop_service, 'robot', None)
        if robot and hasattr(robot, 'velocity_limit') and robot is not active:
            robot.velocity_limit = limit
            updated = True
            logger.info(f"Set velocity_limit to {limit:.2f} on teleop robot")

    # Update any Damiao followers in arm registry
    if system.arm_registry:
        for arm_id, instance in system.arm_registry.arm_instances.items():
            if hasattr(instance, 'velocity_limit'):
                instance.velocity_limit = limit
                updated = True
                logger.info(f"Set velocity_limit to {limit:.2f} on arm_registry/{arm_id}")

    if updated:
        return {"status": "ok", "velocity_limit": limit}
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "No robot with velocity limit found (Damiao required)"}
        )

@router.get("/robot/damiao/status")
async def get_damiao_status():
    """Get status of Damiao robot (if connected).

    Returns connection status, velocity limit, and torque readings.
    """
    system = get_state()
    result = {
        "connected": False,
        "velocity_limit": 1.0,
        "motor_type": None,
    }

    # Check main robot
    robot = None
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        robot = system.robot
    elif system.teleop_service and hasattr(system.teleop_service, 'robot'):
        r = system.teleop_service.robot
        if hasattr(r, 'velocity_limit'):
            robot = r

    if robot:
        result["connected"] = robot.is_connected
        result["velocity_limit"] = robot.velocity_limit
        result["motor_type"] = "damiao"

        # Try to get torque readings for safety monitoring
        if hasattr(robot, 'get_torques'):
            try:
                result["torques"] = robot.get_torques()
                result["torque_limits"] = robot.get_torque_limits()
            except Exception as e:
                logger.warning(f"Failed to read Damiao torques: {e}")

    return result

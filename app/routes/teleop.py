from fastapi import APIRouter, Request

from app.core.hardware.types import ConnectionStatus
from app.dependencies import get_state

router = APIRouter(tags=["teleop"])


@router.post("/teleop/start")
async def start_teleop(request: Request):
    system = get_state()
    try:
        data = await request.json()
    except Exception:
        data = {}

    force = data.get("force", False)
    active_arms = data.get("active_arms", None)

    # Auto-select all connected arms when none specified (e.g. tablet)
    if active_arms is None and system.arm_registry:
        active_arms = [
            arm_id
            for arm_id, status in system.arm_registry.arm_status.items()
            if status == ConnectionStatus.CONNECTED
        ]
        if not active_arms:
            return {"status": "error", "message": "No arms connected"}

    # Lazy Initialize Teleop Service if needed
    if not system.teleop_service:
        if system.robot:
             from app.core.teleop_service import TeleoperationService
             system.teleop_service = TeleoperationService(system.robot, system.leader, system.lock, leader_assists=system.leader_assists, arm_registry=system.arm_registry)
        else:
             return {"status": "error", "message": "Teleop Service not initialized: Robot not connected"}

    if system.teleop_service:
        try:
            system.teleop_service.start(force=force, active_arms=active_arms)
        except Exception as e:
            print(f"Teleop Start Failed: {e}")
            return {"status": "error", "message": str(e)}

    return {"status": "started"}

@router.post("/teleop/stop")
async def stop_teleop():
    system = get_state()
    if system.teleop_service:
        system.teleop_service.stop()
    return {"status": "stopped"}

@router.get("/teleop/status")
def get_teleop_status():
    system = get_state()
    running = False
    if system.teleop_service:
        running = system.teleop_service.is_running
    return {"running": running}

@router.get("/teleop/data")
def get_teleop_data():
    system = get_state()
    if system.teleop_service:
        return {"data": system.teleop_service.get_data()}
    return {"data": []}

@router.post("/teleop/tune")
async def tune_teleop(request: Request):
    system = get_state()
    data = await request.json()
    # data: {k_gravity, k_assist, k_haptic, v_threshold}

    if system.teleop_service and system.teleop_service.leader_assists:
        count = 0
        for arm_id, service in system.teleop_service.leader_assists.items():
            service.update_gains(
                k_gravity=data.get("k_gravity"),
                k_assist=data.get("k_assist"),
                k_haptic=data.get("k_haptic"),
                v_threshold=data.get("v_threshold"),
                k_damping=data.get("k_damping") # New Damping Parameter
            )
            count += 1
        return {"status": "success", "message": f"Updated gains for {count} arms"}
    return {"status": "error", "message": "Teleop service not active"}

@router.post("/teleop/assist/set")
async def set_teleop_assist(request: Request):
    system = get_state()
    data = await request.json()
    enabled = data.get("enabled", True)
    if system.teleop_service:
        system.teleop_service.set_assist_enabled(enabled)
        return {"status": "success", "enabled": enabled}
    return {"status": "error", "message": "Teleop Service not running"}

@router.get("/teleop/force-feedback")
async def get_force_feedback():
    system = get_state()
    if system.teleop_service:
        return {"status": "success", **system.teleop_service.get_force_feedback_state()}
    return {"status": "success", "gripper": False, "joint": False}

@router.post("/teleop/force-feedback")
async def set_force_feedback(request: Request):
    system = get_state()
    data = await request.json()
    if system.teleop_service:
        system.teleop_service.set_force_feedback(
            gripper=data.get("gripper"),
            joint=data.get("joint"),
        )
        return {"status": "success", **system.teleop_service.get_force_feedback_state()}
    return {"status": "error", "message": "Teleop service not active"}

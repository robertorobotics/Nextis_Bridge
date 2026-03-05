"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useTabletStatus } from "@/hooks/use-tablet-status";
import { tabletApi } from "@/lib/tablet-api";
import CameraFeed from "@/components/ui/CameraFeed";

// ─── Design tokens ───────────────────────────────────────────────
const C = {
  bg: "#000000",
  surface: "rgba(28, 28, 30, 0.72)",
  fill: "rgba(120, 120, 128, 0.16)",
  separator: "rgba(84, 84, 88, 0.34)",
  border: "rgba(84, 84, 88, 0.18)",
  label: "#ffffff",
  secondary: "rgba(235, 235, 245, 0.6)",
  tertiary: "rgba(235, 235, 245, 0.3)",
  quaternary: "rgba(235, 235, 245, 0.16)",
  green: "#30d158",
  red: "#ff453a",
  blue: "#0a84ff",
} as const;

const GLASS: React.CSSProperties = {
  background: C.surface,
  backdropFilter: "blur(40px) saturate(180%)",
  WebkitBackdropFilter: "blur(40px) saturate(180%)",
  border: `0.5px solid ${C.border}`,
  borderRadius: 14,
};

const PRESS_TRANSITION =
  "transform 0.2s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.15s ease";

// ─── Types ───────────────────────────────────────────────────────
interface Toast {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}

// ─── Root page ───────────────────────────────────────────────────
export default function TabletPage() {
  const { status, isOnline } = useTabletStatus();
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [busyKeys, setBusyKeys] = useState<Record<string, boolean>>({});
  const toastId = useRef(0);

  // ── Toast system ──
  const showToast = useCallback(
    (message: string, type: Toast["type"] = "info") => {
      const id = toastId.current++;
      setToasts((prev) => [...prev, { id, message, type }]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 2200);
    },
    []
  );

  // ── Busy guard ──
  const isBusy = useCallback(
    (key: string) => !!busyKeys[key],
    [busyKeys]
  );
  const runAction = useCallback(
    async (key: string, fn: () => Promise<void>) => {
      if (busyKeys[key]) return;
      setBusyKeys((prev) => ({ ...prev, [key]: true }));
      try {
        await fn();
      } finally {
        setBusyKeys((prev) => ({ ...prev, [key]: false }));
      }
    },
    [busyKeys]
  );

  // ── Actions ──
  const toggleTeleop = useCallback(async () => {
    await runAction("teleop", async () => {
      try {
        if (status?.teleop_active) {
          await tabletApi.stopTeleop();
          showToast("Teleoperation stopped", "info");
        } else {
          await tabletApi.startTeleop();
          showToast("Teleoperation started", "success");
        }
      } catch (e: unknown) {
        showToast(
          e instanceof Error ? e.message : "Teleop failed",
          "error"
        );
      }
    });
  }, [status?.teleop_active, runAction, showToast]);

  const toggleRecord = useCallback(async () => {
    if (!status?.current_dataset) {
      showToast("Start a recording session from the main app first", "error");
      return;
    }
    await runAction("record", async () => {
      try {
        if (status?.recording_active) {
          await tabletApi.stopEpisode();
          showToast("Episode saved", "success");
        } else {
          await tabletApi.startEpisode();
          showToast("Recording started", "info");
        }
      } catch (e: unknown) {
        showToast(
          e instanceof Error ? e.message : "Recording failed",
          "error"
        );
      }
    });
  }, [
    status?.current_dataset,
    status?.recording_active,
    runAction,
    showToast,
  ]);

  const discardEpisode = useCallback(async () => {
    await runAction("discard", async () => {
      try {
        if (status?.recording_active) {
          await tabletApi.stopEpisode();
        }
        await tabletApi.discardLastEpisode();
        showToast("Episode discarded", "info");
      } catch (e: unknown) {
        showToast(
          e instanceof Error ? e.message : "Discard failed",
          "error"
        );
      }
    });
  }, [status?.recording_active, runAction, showToast]);

  const stopDeploy = useCallback(async () => {
    await runAction("deploy", async () => {
      try {
        await tabletApi.stopDeploy();
        showToast("Deployment stopped", "info");
      } catch (e: unknown) {
        showToast(
          e instanceof Error ? e.message : "Stop deploy failed",
          "error"
        );
      }
    });
  }, [runAction, showToast]);

  const handleEstop = useCallback(async () => {
    try {
      await tabletApi.emergencyStop();
      showToast("EMERGENCY STOP SENT", "error");
    } catch {
      alert("EMERGENCY STOP FAILED! USE PHYSICAL KILL SWITCH!");
    }
  }, [showToast]);

  // ── Derived state ──
  const cameras = status?.connected_cameras ?? [];
  const recording = status?.recording_active ?? false;

  return (
    <div
      className="fixed inset-0 flex flex-col select-none overflow-hidden"
      style={{
        background: C.bg,
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', system-ui, sans-serif",
        color: C.label,
        // Red border when recording
        boxShadow: recording
          ? `inset 0 0 0 3px ${C.red}, inset 0 0 20px rgba(255, 69, 58, 0.15)`
          : "none",
        transition: "box-shadow 0.4s ease",
      }}
    >
      {/* ── Header ── */}
      <Header isOnline={isOnline} />

      {/* ── Split layout ── */}
      <div className="flex flex-1 min-h-0 gap-0">
        {/* Left — Cameras */}
        <div
          className="flex flex-col gap-3 p-4 pr-2"
          style={{ width: "55%" }}
        >
          <CamerasPanel cameras={cameras} />
        </div>

        {/* Right — Controls */}
        <div
          className="flex flex-col gap-0 p-4 pl-2 overflow-y-auto"
          style={{ width: "45%" }}
        >
          <ControlsPanel
            status={status}
            isOnline={isOnline}
            isBusy={isBusy}
            onToggleTeleop={toggleTeleop}
            onToggleRecord={toggleRecord}
            onDiscardEpisode={discardEpisode}
            onStopDeploy={stopDeploy}
            onEstop={handleEstop}
          />
        </div>
      </div>

      {/* ── Toasts ── */}
      <ToastContainer toasts={toasts} />
    </div>
  );
}

// ─── Header ──────────────────────────────────────────────────────
function Header({ isOnline }: { isOnline: boolean }) {
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      );
    };
    tick();
    const id = setInterval(tick, 10_000);
    return () => clearInterval(id);
  }, []);

  return (
    <header
      className="flex items-center justify-between px-5 flex-shrink-0"
      style={{
        height: 52,
        borderBottom: `0.5px solid ${C.separator}`,
        paddingTop: "env(safe-area-inset-top, 0px)",
      }}
    >
      {/* Left — Brand */}
      <div className="flex items-baseline gap-2">
        <span style={{ fontSize: 21, fontWeight: 600 }}>Nextis</span>
        <span style={{ fontSize: 13, fontWeight: 400, color: C.tertiary }}>
          Cell Control
        </span>
      </div>

      {/* Right — Status + time */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: isOnline ? C.green : C.red,
              boxShadow: isOnline
                ? `0 0 6px ${C.green}`
                : `0 0 6px ${C.red}`,
              transition: "background 0.3s, box-shadow 0.3s",
            }}
          />
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: isOnline ? C.secondary : C.red,
            }}
          >
            {isOnline ? "Connected" : "Offline"}
          </span>
        </div>
        <span
          style={{
            fontSize: 13,
            fontWeight: 400,
            color: C.tertiary,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {time}
        </span>
      </div>
    </header>
  );
}

// ─── Cameras panel ───────────────────────────────────────────────
function CamerasPanel({
  cameras,
}: {
  cameras: { id: string; name: string }[];
}) {
  if (cameras.length === 0) {
    return (
      <div
        className="flex-1 flex items-center justify-center rounded-2xl"
        style={{ border: `0.5px solid ${C.border}`, background: C.surface }}
      >
        <span style={{ color: C.quaternary, fontSize: 15, fontWeight: 400 }}>
          No cameras connected
        </span>
      </div>
    );
  }

  const gridClass =
    cameras.length === 1
      ? "grid-cols-1 grid-rows-1"
      : cameras.length === 2
        ? "grid-cols-1 grid-rows-2"
        : "grid-cols-2 grid-rows-2";

  return (
    <div className={`grid gap-3 flex-1 min-h-0 ${gridClass}`}>
      {cameras.map((cam) => (
        <div key={cam.id} className="min-h-0">
          <CameraFeed
            cameraId={cam.id}
            mode="cover"
            showOverlay
            autoReconnect
            label={cam.name}
            className="!rounded-[14px]"
          />
        </div>
      ))}
    </div>
  );
}

// ─── Controls panel ──────────────────────────────────────────────
import type { TabletStatus } from "@/lib/tablet-api";

function ControlsPanel({
  status,
  isOnline,
  isBusy,
  onToggleTeleop,
  onToggleRecord,
  onDiscardEpisode,
  onStopDeploy,
  onEstop,
}: {
  status: TabletStatus | null;
  isOnline: boolean;
  isBusy: (key: string) => boolean;
  onToggleTeleop: () => void;
  onToggleRecord: () => void;
  onDiscardEpisode: () => void;
  onStopDeploy: () => void;
  onEstop: () => void;
}) {
  const arms = status?.connected_arms ?? 0;
  const camCount = status?.connected_cameras?.length ?? 0;
  const episodes = status?.episode_count ?? 0;
  const teleopActive = status?.teleop_active ?? false;
  const recordingActive = status?.recording_active ?? false;
  const deployActive = status?.deployment_active ?? false;
  const currentDataset = status?.current_dataset ?? null;
  const activePolicy = status?.active_policy ?? null;
  const disabled = !isOnline;

  return (
    <div className="flex flex-col gap-3 flex-1">
      {/* Metrics bar */}
      <MetricsBar arms={arms} cameras={camCount} episodes={episodes} />

      {/* Operation */}
      <SectionLabel>Operation</SectionLabel>
      <OperationRow
        title="Teleoperation"
        subtitle="Leader to follower at 60 Hz"
        active={teleopActive}
        activeLabel="Running"
        inactiveLabel="Start"
        activeColor={C.green}
        busy={isBusy("teleop")}
        disabled={disabled || arms === 0}
        onPress={onToggleTeleop}
      />

      {/* Data Collection */}
      <SectionLabel>Data Collection</SectionLabel>
      <OperationRow
        title="Record Episode"
        subtitle={
          currentDataset
            ? recordingActive
              ? `Recording episode ${episodes + 1}`
              : currentDataset
            : "No session active"
        }
        active={recordingActive}
        activeLabel="Recording"
        inactiveLabel="Start"
        activeColor={C.red}
        busy={isBusy("record")}
        disabled={disabled || !currentDataset}
        onPress={onToggleRecord}
        timer={recordingActive}
      />
      {recordingActive && (
        <GlassButton
          label="Discard Episode"
          color={C.red}
          busy={isBusy("discard")}
          disabled={disabled}
          onPress={onDiscardEpisode}
        />
      )}

      {/* Autonomy */}
      <SectionLabel>Autonomy</SectionLabel>
      <OperationRow
        title="Deploy Policy"
        subtitle={
          deployActive && activePolicy
            ? activePolicy
            : "Deploy from main app"
        }
        active={deployActive}
        activeLabel="Deployed"
        inactiveLabel="Deploy"
        activeColor={C.blue}
        busy={isBusy("deploy")}
        disabled={disabled || !deployActive}
        onPress={onStopDeploy}
      />

      {/* Spacer pushes E-stop to bottom */}
      <div className="flex-1 min-h-3" />

      {/* Emergency stop */}
      <button
        onClick={onEstop}
        className="active:scale-[0.975] active:opacity-[0.88] w-full flex-shrink-0"
        style={{
          background: C.red,
          borderRadius: 14,
          padding: "18px 0",
          border: "none",
          cursor: "pointer",
          transition: PRESS_TRANSITION,
        }}
      >
        <span
          style={{
            fontSize: 16,
            fontWeight: 600,
            color: "#ffffff",
            letterSpacing: "0.02em",
          }}
        >
          Emergency Stop
        </span>
      </button>
    </div>
  );
}

// ─── Metrics bar ─────────────────────────────────────────────────
function MetricsBar({
  arms,
  cameras,
  episodes,
}: {
  arms: number;
  cameras: number;
  episodes: number;
}) {
  return (
    <div
      className="flex items-center"
      style={{ ...GLASS, padding: "14px 0" }}
    >
      <Metric value={arms} label="ARMS" />
      <div
        style={{
          width: 0.5,
          alignSelf: "stretch",
          background: C.separator,
          margin: "0",
        }}
      />
      <Metric value={cameras} label="CAMERAS" />
      <div
        style={{
          width: 0.5,
          alignSelf: "stretch",
          background: C.separator,
          margin: "0",
        }}
      />
      <Metric value={episodes} label="EPISODES" />
    </div>
  );
}

function Metric({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex-1 flex flex-col items-center gap-0.5">
      <span
        style={{
          fontSize: 30,
          fontWeight: 200,
          fontVariantNumeric: "tabular-nums",
          lineHeight: 1,
        }}
      >
        {value}
      </span>
      <span
        style={{
          fontSize: 11,
          fontWeight: 500,
          color: C.tertiary,
          letterSpacing: "0.06em",
        }}
      >
        {label}
      </span>
    </div>
  );
}

// ─── Section label ───────────────────────────────────────────────
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontSize: 13,
        fontWeight: 500,
        color: C.tertiary,
        paddingLeft: 4,
        paddingTop: 8,
      }}
    >
      {children}
    </span>
  );
}

// ─── Operation row ───────────────────────────────────────────────
function OperationRow({
  title,
  subtitle,
  active,
  activeLabel,
  inactiveLabel,
  activeColor,
  busy,
  disabled,
  onPress,
  timer,
}: {
  title: string;
  subtitle: string;
  active: boolean;
  activeLabel: string;
  inactiveLabel: string;
  activeColor: string;
  busy: boolean;
  disabled: boolean;
  onPress: () => void;
  timer?: boolean;
}) {
  return (
    <button
      onClick={onPress}
      disabled={disabled || busy}
      className="w-full text-left active:scale-[0.975] active:opacity-[0.85] disabled:opacity-50 disabled:active:scale-100 disabled:active:opacity-50"
      style={{
        ...GLASS,
        padding: "14px 18px",
        cursor: disabled || busy ? "default" : "pointer",
        transition: PRESS_TRANSITION,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      {/* Left — title + subtitle */}
      <div className="flex flex-col gap-0.5 min-w-0">
        <span style={{ fontSize: 17, fontWeight: 600 }}>{title}</span>
        <span
          className="truncate"
          style={{ fontSize: 13, fontWeight: 400, color: C.tertiary }}
        >
          {subtitle}
        </span>
      </div>

      {/* Right — status */}
      <div className="flex items-center gap-2 flex-shrink-0 ml-3">
        {busy ? (
          <span style={{ fontSize: 15, color: C.tertiary }}>···</span>
        ) : (
          <>
            {active && timer && <RecordingTimer />}
            <span
              style={{
                fontSize: 15,
                fontWeight: 500,
                color: active ? activeColor : C.tertiary,
              }}
            >
              {active ? activeLabel : inactiveLabel}
            </span>
            <StatusDot active={active} color={activeColor} />
          </>
        )}
      </div>
    </button>
  );
}

// ─── Glass button (centered text) ────────────────────────────────
function GlassButton({
  label,
  color,
  busy,
  disabled,
  onPress,
}: {
  label: string;
  color: string;
  busy: boolean;
  disabled: boolean;
  onPress: () => void;
}) {
  return (
    <button
      onClick={onPress}
      disabled={disabled || busy}
      className="w-full active:scale-[0.975] active:opacity-[0.85] disabled:opacity-50"
      style={{
        ...GLASS,
        padding: "14px 18px",
        cursor: disabled || busy ? "default" : "pointer",
        transition: PRESS_TRANSITION,
        textAlign: "center",
      }}
    >
      <span style={{ fontSize: 15, fontWeight: 500, color }}>
        {busy ? "···" : label}
      </span>
    </button>
  );
}

// ─── Status dot ──────────────────────────────────────────────────
function StatusDot({
  active,
  color,
}: {
  active: boolean;
  color: string;
}) {
  return (
    <div
      style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: active ? color : C.quaternary,
        boxShadow: active ? `0 0 8px ${color}` : "none",
        transition: "background 0.3s, box-shadow 0.3s",
      }}
    />
  );
}

// ─── Recording timer ─────────────────────────────────────────────
function RecordingTimer() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    setElapsed(0);
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const secs = String(elapsed % 60).padStart(2, "0");

  return (
    <span
      style={{
        fontSize: 15,
        fontWeight: 500,
        color: C.red,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {mins}:{secs}
    </span>
  );
}

// ─── Toast container ─────────────────────────────────────────────
function ToastContainer({ toasts }: { toasts: Toast[] }) {
  if (toasts.length === 0) return null;

  return (
    <div
      className="fixed flex flex-col items-center gap-2 pointer-events-none"
      style={{
        bottom: "calc(env(safe-area-inset-bottom, 0px) + 24px)",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 100,
      }}
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="pointer-events-auto animate-in fade-in slide-in-from-bottom-2"
          style={{
            background: "#1c1c1e",
            border: `0.5px solid ${C.border}`,
            borderRadius: 100,
            padding: "10px 20px",
            fontSize: 14,
            fontWeight: 500,
            color:
              toast.type === "success"
                ? C.green
                : toast.type === "error"
                  ? C.red
                  : C.secondary,
            whiteSpace: "nowrap",
            animation: "toastIn 0.25s ease-out",
          }}
        >
          {toast.message}
        </div>
      ))}
      <style>{`
        @keyframes toastIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

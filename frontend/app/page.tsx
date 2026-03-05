"use client";

import { useState } from "react";
import { User, LogOut } from "lucide-react";
import TaskGraph from "../components/TaskGraph";
import CalibrationModal from "../components/modals/calibration";
import ArmManagerModal from "../components/modals/arms";
import MotorMonitorModal from "../components/MotorMonitorModal";
import TeleopModal from "../components/modals/teleop";
import RecordingModal from "../components/modals/recording";
import RecordingActiveView from "../components/modals/recording/RecordingActiveView";
import DatasetViewerModal from "../components/modals/datasets";
import UploadModal from "../components/UploadModal";
import AuthModal from "../components/AuthModal";
import TrainModal from "../components/modals/training";
import HILModal from "../components/modals/hil";
import RLTrainingModal from "../components/modals/rl";
import DeployModal from "../components/modals/deploy";
import { useAuth } from "../lib/AuthContext";
import { useTheme } from "../lib/ThemeContext";
import { ThemeToggle } from "../components/ThemeToggle";
import ControlDock from "../components/ControlDock";
import ChatWindow from "../components/ChatWindow";

export default function Dashboard() {
  const { user, loading: authLoading, signOut } = useAuth();
  const { resolvedTheme } = useTheme();
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  // Modal States
  const [isCalibrationOpen, setIsCalibrationOpen] = useState(false);
  const [isTeleopOpen, setIsTeleopOpen] = useState(false);
  const [isRecordingOpen, setIsRecordingOpen] = useState(false);
  const [isDatasetViewerOpen, setIsDatasetViewerOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isTrainOpen, setIsTrainOpen] = useState(false);
  const [isHILOpen, setIsHILOpen] = useState(false);
  const [isRLTrainingOpen, setIsRLTrainingOpen] = useState(false);
  const [isArmManagerOpen, setIsArmManagerOpen] = useState(false);
  const [isMotorMonitorOpen, setIsMotorMonitorOpen] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isDeployOpen, setIsDeployOpen] = useState(false);

  // Plan state — shared between ChatWindow (sets it) and TaskGraph (displays it)
  const [currentPlan, setCurrentPlan] = useState<any>(null);

  // Global maximized state - only one window can be maximized at a time
  const [maximizedWindow, setMaximizedWindow] = useState<string | null>(null);

  // Recording phase state machine: idle → setup (RecordingModal) → active (overlay)
  const [recordingPhase, setRecordingPhase] = useState<"idle" | "active">("idle");
  const [activeDatasetName, setActiveDatasetName] = useState("");

  return (
    <div className="h-screen w-screen overflow-hidden bg-neutral-50 dark:bg-zinc-950 font-sans selection:bg-black selection:text-white dark:selection:bg-white dark:selection:text-black">
      {/* Modals */}
      <CalibrationModal
        isOpen={isCalibrationOpen}
        onClose={() => setIsCalibrationOpen(false)}
        language="en"
      />
      <TeleopModal
        isOpen={isTeleopOpen}
        onClose={() => {
          setIsTeleopOpen(false);
          if (maximizedWindow === "teleop") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <RecordingModal
        isOpen={isRecordingOpen}
        onClose={() => {
          setIsRecordingOpen(false);
          if (maximizedWindow === "recording") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
        onSessionStarted={(datasetName) => {
          setIsRecordingOpen(false);
          if (maximizedWindow === "recording") setMaximizedWindow(null);
          setActiveDatasetName(datasetName);
          setRecordingPhase("active");
        }}
      />
      <DatasetViewerModal
        isOpen={isDatasetViewerOpen}
        onClose={() => {
          setIsDatasetViewerOpen(false);
          if (maximizedWindow === "datasetViewer") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadComplete={(dataset) => {
          console.log("Upload complete:", dataset);
          setIsUploadOpen(false);
        }}
      />
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
      />
      <TrainModal
        isOpen={isTrainOpen}
        onClose={() => {
          setIsTrainOpen(false);
          if (maximizedWindow === "train") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <HILModal
        isOpen={isHILOpen}
        onClose={() => {
          setIsHILOpen(false);
          if (maximizedWindow === "hil") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <RLTrainingModal
        isOpen={isRLTrainingOpen}
        onClose={() => {
          setIsRLTrainingOpen(false);
          if (maximizedWindow === "rl-training") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <DeployModal
        isOpen={isDeployOpen}
        onClose={() => {
          setIsDeployOpen(false);
          if (maximizedWindow === "deploy") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <ArmManagerModal
        isOpen={isArmManagerOpen}
        onClose={() => setIsArmManagerOpen(false)}
      />
      <MotorMonitorModal
        isOpen={isMotorMonitorOpen}
        onClose={() => {
          setIsMotorMonitorOpen(false);
          if (maximizedWindow === "motor-monitor") setMaximizedWindow(null);
        }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />

      {/* Active Recording Overlay */}
      {recordingPhase === "active" && (
        <RecordingActiveView
          datasetName={activeDatasetName}
          onSessionEnded={() => {
            setRecordingPhase("idle");
            setActiveDatasetName("");
          }}
        />
      )}

      {/* 1. LAYER: BACKGROUND (Task Graph) */}
      <div className="absolute inset-0 z-0">
        <TaskGraph plan={currentPlan} darkMode={resolvedTheme === "dark"} />
      </div>

      {/* 2. LAYER: UI OVERLAY (Header & Dock) */}
      <div className="absolute top-0 left-0 right-0 z-30 p-4 lg:p-6 pointer-events-none flex flex-col items-start gap-3 xl:flex-row xl:justify-between xl:items-start">
        {/* Branding + User */}
        <div className="pointer-events-auto flex items-center gap-2">
          <div className="flex items-center gap-3 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-4 py-2 rounded-2xl shadow-sm hover:bg-white dark:hover:bg-zinc-900 transition-colors cursor-default">
            <div className="w-8 h-8 bg-black dark:bg-white rounded-lg flex items-center justify-center shadow-lg shadow-black/20 dark:shadow-white/10">
              <span className="text-white dark:text-black font-bold tracking-tighter">
                N
              </span>
            </div>
            <span className="font-semibold tracking-tight text-neutral-900 dark:text-zinc-100">
              Nextis
            </span>
          </div>

          <div className="bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 rounded-2xl shadow-sm">
            <ThemeToggle />
          </div>

          {!authLoading &&
            (user ? (
              <div className="flex items-center gap-2 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 pl-3 pr-2 py-1.5 rounded-2xl shadow-sm">
                <span className="text-xs text-neutral-600 dark:text-zinc-400 max-w-[120px] truncate">
                  {user.email}
                </span>
                <button
                  onClick={() => signOut()}
                  className="p-1.5 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthOpen(true)}
                className="flex items-center gap-2 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-4 py-2 rounded-2xl shadow-sm hover:bg-white dark:hover:bg-zinc-900 transition-colors"
              >
                <User className="w-4 h-4 text-neutral-600 dark:text-zinc-400" />
                <span className="text-xs font-medium text-neutral-700 dark:text-zinc-300">
                  Sign In
                </span>
              </button>
            ))}
        </div>

        {/* Center Control Dock */}
        <ControlDock
          modalStates={{
            isRecordingOpen,
            isDatasetViewerOpen,
            isUploadOpen,
            isTrainOpen,
            isHILOpen,
            isRLTrainingOpen,
            isDeployOpen,
            isChatOpen,
            isArmManagerOpen,
            isMotorMonitorOpen,
            isCalibrationOpen,
            isTeleopOpen,
          }}
          setIsRecordingOpen={setIsRecordingOpen}
          setIsDatasetViewerOpen={setIsDatasetViewerOpen}
          setIsUploadOpen={setIsUploadOpen}
          setIsTrainOpen={setIsTrainOpen}
          setIsHILOpen={setIsHILOpen}
          setIsRLTrainingOpen={setIsRLTrainingOpen}
          setIsDeployOpen={setIsDeployOpen}
          setIsChatOpen={setIsChatOpen}
          setIsArmManagerOpen={setIsArmManagerOpen}
          setIsMotorMonitorOpen={setIsMotorMonitorOpen}
          setIsCalibrationOpen={setIsCalibrationOpen}
          setIsTeleopOpen={setIsTeleopOpen}
        />
      </div>

      {/* 3. LAYER: FLOATING CHAT WINDOW */}
      <ChatWindow
        isChatOpen={isChatOpen}
        setIsChatOpen={setIsChatOpen}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
        onPlanReceived={setCurrentPlan}
      />
    </div>
  );
}

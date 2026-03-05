import React, { useState, useEffect, useMemo } from "react";
import {
  Play,
  Loader2,
  AlertCircle,
  Shield,
  Brain,
  Sparkles,
  Eye,
  User,
  Activity,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { armsApi, policiesApi, rlApi, deployApi } from "../../../lib/api";
import type {
  PolicyInfo,
  Arm,
  Pairing,
  RewardClassifier,
  SARMModel,
} from "../../../lib/api/types";

interface DeploySetupProps {
  policies: PolicyInfo[];
  arms: Arm[];
  onDeployStarted: () => void;
  speak: (text: string) => void;
}

type RewardSource = "sarm" | "gvl" | "classifier";

export default function DeploySetup({ policies, arms, onDeployStarted, speak }: DeploySetupProps) {
  // Arm selection
  const [pairings, setPairings] = useState<Pairing[]>([]);
  const [selectedFollowers, setSelectedFollowers] = useState<Set<string>>(new Set());

  // Policy selection
  const [selectedPolicyId, setSelectedPolicyId] = useState("");
  const [policyPreview, setPolicyPreview] = useState<PolicyInfo | null>(null);

  // Safety
  const [speedScale, setSpeedScale] = useState(0.3);

  // Capabilities
  const [hilEnabled, setHilEnabled] = useState(false);
  const [rlEnabled, setRlEnabled] = useState(false);
  const [interventionDataset, setInterventionDataset] = useState("hil_interventions/v1");
  const [taskDescription, setTaskDescription] = useState("");

  // RL sub-settings
  const [rewardSource, setRewardSource] = useState<RewardSource>("sarm");
  const [classifiers, setClassifiers] = useState<RewardClassifier[]>([]);
  const [sarmModels, setSarmModels] = useState<SARMModel[]>([]);
  const [selectedClassifier, setSelectedClassifier] = useState("");
  const [selectedSarmModel, setSelectedSarmModel] = useState("");
  const [maxEpisodes, setMaxEpisodes] = useState(100);

  // Temporal ensemble (ACT only)
  const [temporalEnsemble, setTemporalEnsemble] = useState(false);
  const [temporalEnsembleCoeff, setTemporalEnsembleCoeff] = useState(0.01);

  // UI state
  const [error, setError] = useState("");
  const [isStarting, setIsStarting] = useState(false);

  const followers = useMemo(() => arms.filter((a) => a.role === "follower"), [arms]);
  const leaders = useMemo(() => arms.filter((a) => a.role === "leader"), [arms]);

  // Fetch pairings on mount
  useEffect(() => {
    armsApi.listPairings().then((d) => setPairings(d.pairings || [])).catch(console.error);
  }, []);

  // Fetch RL models when RL is enabled
  useEffect(() => {
    if (rlEnabled) {
      rlApi.listClassifiers().then((d) => setClassifiers(d.classifiers || [])).catch(console.error);
      rlApi.listSarmModels().then((d) => setSarmModels(d.models || [])).catch(console.error);
    }
  }, [rlEnabled]);

  // Fetch policy preview on selection
  useEffect(() => {
    if (selectedPolicyId) {
      policiesApi.get(selectedPolicyId).then(setPolicyPreview).catch(console.error);
    } else {
      setPolicyPreview(null);
    }
  }, [selectedPolicyId]);

  const isActPolicy = policyPreview?.policy_type?.toLowerCase() === "act";

  // Reset temporal ensemble when switching away from ACT
  useEffect(() => {
    if (!isActPolicy) setTemporalEnsemble(false);
  }, [isActPolicy]);

  // Auto-selected leaders based on pairings
  const autoSelectedLeaders = useMemo(() => {
    const leaderIds = new Set<string>();
    for (const p of pairings) {
      if (selectedFollowers.has(p.follower_id)) {
        leaderIds.add(p.leader_id);
      }
    }
    return leaderIds;
  }, [pairings, selectedFollowers]);

  const toggleFollower = (id: string) => {
    setSelectedFollowers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Compute deployment mode
  const mode = rlEnabled ? "hil_serl" : hilEnabled ? "hil" : "inference";
  const modeLabel = rlEnabled ? "HIL-SERL" : hilEnabled ? "HIL" : "Inference";
  const modeColor = rlEnabled ? "orange" : hilEnabled ? "blue" : "emerald";

  const canDeploy =
    selectedFollowers.size > 0 &&
    selectedPolicyId &&
    !isStarting &&
    !(rlEnabled && rewardSource === "sarm" && !selectedSarmModel) &&
    !(rlEnabled && rewardSource === "classifier" && !selectedClassifier) &&
    !(rlEnabled && rewardSource === "gvl" && !taskDescription);

  const startDeploy = async () => {
    setError("");
    setIsStarting(true);

    try {
      const activeArms = [
        ...Array.from(selectedFollowers),
        ...Array.from(autoSelectedLeaders),
      ];

      const config: Record<string, unknown> = {
        policy_id: selectedPolicyId,
        active_arms: activeArms,
        mode,
        safety: { speed_scale: speedScale },
        movement_scale: speedScale,
      };

      if (temporalEnsemble && isActPolicy) {
        config.temporal_ensemble_override = temporalEnsembleCoeff;
      }

      if (hilEnabled) {
        config.intervention_dataset = interventionDataset;
        config.task = taskDescription;
      }

      if (rlEnabled) {
        config.reward_source = rewardSource;
        config.max_episodes = maxEpisodes;
        if (rewardSource === "sarm") config.reward_model = selectedSarmModel;
        else if (rewardSource === "classifier") config.reward_model = selectedClassifier;
      }

      const data = await deployApi.start(config);
      if (data.status === "started") {
        speak(`Deployment started, ${modeLabel} mode`);
        onDeployStarted();
      } else {
        setError("Failed to start deployment");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setIsStarting(false);
    }
  };

  const speedColor =
    speedScale >= 0.8 ? "text-red-500" : speedScale >= 0.5 ? "text-amber-500" : "text-green-500";

  const speedHelp =
    speedScale < 0.3
      ? "Gentle \u2014 ideal for first test"
      : speedScale < 0.6
      ? "Moderate \u2014 verify policy behavior before increasing"
      : speedScale < 0.8
      ? "Fast \u2014 use only with tested policies"
      : "Maximum \u2014 experienced operators only";

  return (
    <div className="flex-1 overflow-y-auto space-y-5 pr-1">
      {/* Error Banner */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl text-sm text-red-700 dark:text-red-300">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* ── A. Arm Selector ── */}
      <section>
        <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">
          Select Arms
        </h3>
        {followers.length === 0 ? (
          <p className="text-xs text-neutral-400 dark:text-zinc-500">
            No follower arms configured. Connect arms first.
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {followers.map((arm) => {
              const selected = selectedFollowers.has(arm.id);
              const connected = arm.status === "connected";
              return (
                <button
                  key={arm.id}
                  onClick={() => toggleFollower(arm.id)}
                  className={`relative p-3 rounded-xl border-2 transition-all text-left ${
                    selected
                      ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30"
                      : "border-neutral-200 dark:border-zinc-700 hover:border-emerald-300 dark:hover:border-emerald-700"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        connected ? "bg-green-500" : "bg-red-400"
                      }`}
                    />
                    <span className="font-semibold text-sm text-neutral-900 dark:text-zinc-100 truncate">
                      {arm.name}
                    </span>
                  </div>
                  <div className="text-xs text-neutral-400 dark:text-zinc-500 mt-0.5">
                    {arm.motor_type}
                  </div>
                  {!arm.calibrated && (
                    <span className="absolute top-1.5 right-1.5 text-[10px] bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-400 px-1.5 py-0.5 rounded font-medium">
                      Uncalibrated
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}
        {/* Auto-selected leaders */}
        {autoSelectedLeaders.size > 0 && (
          <div className="mt-2 text-xs text-neutral-400 dark:text-zinc-500 flex items-center gap-1">
            <User className="w-3 h-3" />
            Leaders auto-selected:{" "}
            {leaders
              .filter((l) => autoSelectedLeaders.has(l.id))
              .map((l) => l.name)
              .join(", ")}
          </div>
        )}
      </section>

      {/* ── B. Policy Selector ── */}
      <section>
        <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">
          Policy
        </h3>
        <select
          value={selectedPolicyId}
          onChange={(e) => setSelectedPolicyId(e.target.value)}
          className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
        >
          <option value="">Choose a trained policy...</option>
          {policies.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.policy_type})
            </option>
          ))}
        </select>
        {policyPreview && (
          <div className="mt-2 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 border border-neutral-100 dark:border-zinc-700 text-xs space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-neutral-500 dark:text-zinc-400">Type</span>
              <span className="font-medium text-neutral-800 dark:text-zinc-200">{policyPreview.policy_type}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-neutral-500 dark:text-zinc-400">Dataset</span>
              <span className="font-medium text-neutral-800 dark:text-zinc-200">{policyPreview.dataset_repo_id}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-neutral-500 dark:text-zinc-400">Steps</span>
              <span className="font-medium text-neutral-800 dark:text-zinc-200">{policyPreview.steps?.toLocaleString()}</span>
            </div>
            {policyPreview.final_loss != null && (
              <div className="flex items-center justify-between">
                <span className="text-neutral-500 dark:text-zinc-400">Final Loss</span>
                <span className="font-medium text-neutral-800 dark:text-zinc-200">{policyPreview.final_loss.toFixed(4)}</span>
              </div>
            )}
            {isActPolicy && (
              <div className="flex items-center justify-between">
                <span className="text-neutral-500 dark:text-zinc-400">Ensemble</span>
                <span className="text-xs font-medium text-violet-600 dark:text-violet-400">Recommended (ACT)</span>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── B2. Policy Settings (ACT only) ── */}
      {isActPolicy && (
        <section className="bg-violet-50 dark:bg-violet-950/20 rounded-xl p-4 border border-violet-200 dark:border-violet-800">
          <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 flex items-center gap-1.5 mb-3">
            <Activity className="w-4 h-4 text-violet-500" /> Policy Settings
          </h3>

          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={temporalEnsemble}
              onChange={(e) => setTemporalEnsemble(e.target.checked)}
              className="mt-0.5 w-4 h-4 rounded border-violet-300 dark:border-violet-600 text-violet-600 focus:ring-violet-500"
            />
            <div>
              <span className="text-sm font-semibold text-neutral-800 dark:text-zinc-200">
                Temporal Ensembling
              </span>
              <p className="text-xs text-neutral-400 dark:text-zinc-500">
                Exponential weighting of overlapping action chunks for smoother control (recommended for ACT)
              </p>
            </div>
          </label>

          {temporalEnsemble && (
            <div className="mt-3 ml-7">
              <label className="block text-xs font-bold text-violet-700 dark:text-violet-300 mb-1">
                Ensemble Coefficient
              </label>
              <input
                type="number"
                value={temporalEnsembleCoeff}
                onChange={(e) => setTemporalEnsembleCoeff(parseFloat(e.target.value) || 0.01)}
                min={0.001}
                max={1.0}
                step={0.001}
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-violet-200 dark:border-violet-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
              <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
                Lower = smoother (recommended: 0.01). Sets action_steps=1 automatically.
              </p>
            </div>
          )}
        </section>
      )}

      {/* ── C. Safety Panel ── */}
      <section className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-100 dark:border-zinc-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 flex items-center gap-1.5">
            <Shield className="w-4 h-4 text-emerald-500" /> Safety
          </h3>
          <span className={`text-sm font-bold ${speedColor}`}>
            {Math.round(speedScale * 100)}%
          </span>
        </div>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={speedScale}
          onChange={(e) => setSpeedScale(parseFloat(e.target.value))}
          className="w-full h-2 rounded-full appearance-none cursor-pointer"
          style={{
            background:
              "linear-gradient(to right, #22c55e 0%, #22c55e 40%, #eab308 50%, #eab308 70%, #ef4444 80%, #ef4444 100%)",
          }}
        />
        <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-2">{speedHelp}</p>

        {/* Safety stage indicators */}
        <div className="flex flex-wrap gap-2 mt-3">
          {["Joint Limits", "Velocity Clamp", "Acceleration Filter", "Torque Monitor"].map((stage) => (
            <span
              key={stage}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 rounded-full"
            >
              <Shield className="w-3 h-3" /> {stage}
            </span>
          ))}
        </div>
      </section>

      {/* ── D. Capabilities ── */}
      <section className="space-y-3">
        <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300">
          Capabilities
        </h3>

        {/* HIL checkbox */}
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={hilEnabled}
            onChange={(e) => {
              setHilEnabled(e.target.checked);
              if (!e.target.checked) setRlEnabled(false);
            }}
            className="mt-0.5 w-4 h-4 rounded border-neutral-300 dark:border-zinc-600 text-blue-600 focus:ring-blue-500"
          />
          <div>
            <span className="text-sm font-semibold text-neutral-800 dark:text-zinc-200">
              Human-in-the-Loop
            </span>
            <p className="text-xs text-neutral-400 dark:text-zinc-500">
              Record interventions, pause after human control for review
            </p>
          </div>
        </label>

        {/* HIL expanded options */}
        {hilEnabled && (
          <div className="ml-7 space-y-3 bg-blue-50 dark:bg-blue-950/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <div>
              <label className="block text-xs font-bold text-blue-700 dark:text-blue-300 mb-1">
                Intervention Dataset
              </label>
              <input
                type="text"
                value={interventionDataset}
                onChange={(e) => setInterventionDataset(e.target.value)}
                placeholder="hil_interventions/v1"
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-blue-200 dark:border-blue-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs font-bold text-blue-700 dark:text-blue-300 mb-1">
                Task Description
              </label>
              <input
                type="text"
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                placeholder="Describe the task..."
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-blue-200 dark:border-blue-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        )}

        {/* Online RL checkbox */}
        <label
          className={`flex items-start gap-3 ${hilEnabled ? "cursor-pointer" : "cursor-not-allowed opacity-50"}`}
        >
          <input
            type="checkbox"
            checked={rlEnabled}
            disabled={!hilEnabled}
            onChange={(e) => setRlEnabled(e.target.checked)}
            className="mt-0.5 w-4 h-4 rounded border-neutral-300 dark:border-zinc-600 text-orange-600 focus:ring-orange-500"
          />
          <div>
            <span className="text-sm font-semibold text-neutral-800 dark:text-zinc-200">
              Online RL (HIL-SERL)
            </span>
            <p className="text-xs text-neutral-400 dark:text-zinc-500">
              Requires HIL. Fine-tune policy online with reward signals.
            </p>
          </div>
        </label>

        {/* RL expanded options */}
        {rlEnabled && (
          <div className="ml-7 space-y-3 bg-orange-50 dark:bg-orange-950/20 rounded-xl p-4 border border-orange-200 dark:border-orange-800">
            {/* Reward source grid */}
            <label className="block text-xs font-bold text-orange-700 dark:text-orange-300 mb-1">
              Reward Source
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setRewardSource("sarm")}
                className={`p-3 rounded-xl border-2 transition-all text-left ${
                  rewardSource === "sarm"
                    ? "border-orange-500 bg-orange-50 dark:bg-orange-900/30"
                    : "border-neutral-200 dark:border-zinc-700 hover:border-orange-300"
                }`}
              >
                <Brain className={`w-5 h-5 mb-1 ${rewardSource === "sarm" ? "text-orange-600 dark:text-orange-400" : "text-neutral-400 dark:text-zinc-500"}`} />
                <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100 flex items-center gap-1">
                  SARM
                  <span className="text-[10px] bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-400 px-1.5 py-0.5 rounded font-medium">Best</span>
                </div>
                <div className="text-xs text-neutral-400 dark:text-zinc-500">Stage-aware</div>
              </button>
              <button
                onClick={() => setRewardSource("gvl")}
                className={`p-3 rounded-xl border-2 transition-all text-left ${
                  rewardSource === "gvl"
                    ? "border-orange-500 bg-orange-50 dark:bg-orange-900/30"
                    : "border-neutral-200 dark:border-zinc-700 hover:border-orange-300"
                }`}
              >
                <Sparkles className={`w-5 h-5 mb-1 ${rewardSource === "gvl" ? "text-orange-600 dark:text-orange-400" : "text-neutral-400 dark:text-zinc-500"}`} />
                <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">GVL</div>
                <div className="text-xs text-neutral-400 dark:text-zinc-500">Zero-shot (Gemini)</div>
              </button>
              <button
                onClick={() => setRewardSource("classifier")}
                className={`p-3 rounded-xl border-2 transition-all text-left ${
                  rewardSource === "classifier"
                    ? "border-orange-500 bg-orange-50 dark:bg-orange-900/30"
                    : "border-neutral-200 dark:border-zinc-700 hover:border-orange-300"
                }`}
              >
                <Eye className={`w-5 h-5 mb-1 ${rewardSource === "classifier" ? "text-orange-600 dark:text-orange-400" : "text-neutral-400 dark:text-zinc-500"}`} />
                <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">Classifier</div>
                <div className="text-xs text-neutral-400 dark:text-zinc-500">Binary (0/1)</div>
              </button>
            </div>

            {/* Reward model selector */}
            {rewardSource === "sarm" && (
              <div>
                <label className="block text-xs font-bold text-orange-700 dark:text-orange-300 mb-1">SARM Model</label>
                <select
                  value={selectedSarmModel}
                  onChange={(e) => setSelectedSarmModel(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-orange-200 dark:border-orange-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                  <option value="">Select model...</option>
                  {sarmModels.map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
              </div>
            )}
            {rewardSource === "classifier" && (
              <div>
                <label className="block text-xs font-bold text-orange-700 dark:text-orange-300 mb-1">Classifier</label>
                <select
                  value={selectedClassifier}
                  onChange={(e) => setSelectedClassifier(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-orange-200 dark:border-orange-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                  <option value="">Select classifier...</option>
                  {classifiers.map((c) => (
                    <option key={c.name} value={c.name}>{c.name} ({(c.accuracy * 100).toFixed(0)}%)</option>
                  ))}
                </select>
              </div>
            )}
            {rewardSource === "gvl" && !taskDescription && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Enter a task description above for GVL to work.
              </p>
            )}

            {/* Max episodes */}
            <div>
              <label className="block text-xs font-bold text-orange-700 dark:text-orange-300 mb-1">Max Episodes</label>
              <input
                type="number"
                value={maxEpisodes}
                onChange={(e) => setMaxEpisodes(parseInt(e.target.value) || 100)}
                min={1}
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-orange-200 dark:border-orange-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>
          </div>
        )}
      </section>

      {/* ── E. Deploy Button ── */}
      <button
        onClick={startDeploy}
        disabled={!canDeploy}
        className={`w-full py-4 text-white rounded-2xl font-bold text-lg hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-xl ${
          modeColor === "orange"
            ? "bg-orange-600 hover:bg-orange-700 shadow-orange-200 dark:shadow-orange-900/30"
            : modeColor === "blue"
            ? "bg-blue-600 hover:bg-blue-700 shadow-blue-200 dark:shadow-blue-900/30"
            : "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-200 dark:shadow-emerald-900/30"
        }`}
      >
        {isStarting ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : (
          <Play className="w-5 h-5" />
        )}
        {isStarting ? "Starting..." : `Deploy \u2014 ${modeLabel}`}
      </button>
    </div>
  );
}

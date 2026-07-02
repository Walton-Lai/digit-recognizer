/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { TrainingProgress, DigitSample } from "../types";
import { Play, RotateCcw, Plus, ListPlus, Sliders, Database, TrendingDown, RefreshCw } from "lucide-react";

interface TrainingControlsProps {
  progress: TrainingProgress;
  onTrain: (epochs: number, samplesPerDigit: number, learningRate: number) => void;
  onAddCustomSample: (label: number) => void;
  customSamplesCount: number;
  totalDatasetSize: number;
  onClearCustomSamples: () => void;
}

export function TrainingControls({
  progress,
  onTrain,
  onAddCustomSample,
  customSamplesCount,
  totalDatasetSize,
  onClearCustomSamples,
}: TrainingControlsProps) {
  const [epochs, setEpochs] = useState(15);
  const [samplesPerDigit, setSamplesPerDigit] = useState(50);
  const [learningRate, setLearningRate] = useState(0.01);
  const [selectedLabel, setSelectedLabel] = useState<number>(5);

  const handleTrainClick = () => {
    if (progress.isTraining) return;
    onTrain(epochs, samplesPerDigit, learningRate);
  };

  // Render SVG Sparkline/Chart for loss history
  const renderHistoryChart = () => {
    const history = progress.history;
    if (history.length === 0) {
      return (
        <div className="h-28 flex items-center justify-center border border-dashed border-[#00FF00]/40 bg-black text-[10px] font-mono uppercase text-white/40 tracking-tight">
          No training logs. Click 'Train Network' to begin.
        </div>
      );
    }

    const padding = 12;
    const width = 280;
    const height = 100;
    
    const maxLoss = Math.max(...history.map((h) => h.loss), 0.5);
    const minLoss = Math.min(...history.map((h) => h.loss), 0);
    const lossRange = maxLoss - minLoss || 1;

    // Generate points for SVG path
    const points = history
      .map((h, idx) => {
        const x = padding + (idx * (width - padding * 2)) / (history.length - 1 || 1);
        const y = height - padding - ((h.loss - minLoss) / lossRange) * (height - padding * 2);
        return `${x},${y}`;
      })
      .join(" ");

    // Generate area points (closing the shape to bottom)
    const areaPoints = history.length > 0
      ? `${padding},${height - padding} ${points} ${width - padding},${height - padding}`
      : "";

    return (
      <div className="bg-black p-3 border border-white/10">
        <div className="flex justify-between items-center text-[9px] font-mono uppercase tracking-tight mb-2 text-white/70">
          <span className="flex items-center gap-1">
            <TrendingDown className="h-3 w-3 text-[#00FF00]" />
            Loss History Epochs
          </span>
          <span>
            INIT: <b className="text-[#00FF00]">{history[0].loss.toFixed(3)}</b> &rarr; CURR: <b className="text-[#00FF00] font-bold">{history[history.length - 1].loss.toFixed(3)}</b>
          </span>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-24 overflow-visible">
          {/* Grid lines */}
          <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="rgba(0, 255, 0, 0.1)" strokeDasharray="2,2" />
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(0, 255, 0, 0.1)" strokeDasharray="2,2" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(0, 255, 0, 0.3)" />

          {/* Fill Area under loss path */}
          {areaPoints && (
            <polygon
              points={areaPoints}
              fill="url(#lossGrad)"
              opacity="0.1"
            />
          )}

          {/* Line Path */}
          <polyline
            fill="none"
            stroke="#00FF00"
            strokeWidth="2"
            points={points}
          />

          {/* Points markers */}
          {history.map((h, idx) => {
            const x = padding + (idx * (width - padding * 2)) / (history.length - 1 || 1);
            const y = height - padding - ((h.loss - minLoss) / lossRange) * (height - padding * 2);
            return (
              <circle
                key={idx}
                cx={x}
                cy={y}
                r="2.5"
                className="fill-[#00FF00] stroke-black stroke-1 hover:r-4 cursor-pointer"
                title={`Epoch ${h.epoch}: Loss ${h.loss.toFixed(4)}`}
              />
            );
          })}

          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00FF00" />
              <stop offset="100%" stopColor="#00FF00" stopOpacity="0" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full items-stretch">
      {/* Hyperparameter Configuration & Training */}
      <div className="bg-[#111111] border-2 border-[#00FF00] p-6 flex flex-col gap-4 shadow-[4px_4px_0px_0px_rgba(0,255,0,0.3)]">
        <div className="flex items-center justify-between border-b border-[#00FF00]/30 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-[#00FF00]" />
            <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-[#00FF00]">Hyperparameters</h3>
          </div>
          <div className="text-[10px] font-mono text-[#00FF00] bg-black border border-[#00FF00]/40 px-2 py-0.5">
            LEARNING_LAB
          </div>
        </div>

        {/* Hyperparameter Inputs */}
        <div className="flex flex-col gap-3">
          {/* Epochs Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs font-mono uppercase text-white/70">
              <span>Training Epochs:</span>
              <span className="text-[#00FF00] font-bold">[{epochs}]</span>
            </div>
            <input
              type="range"
              min={5}
              max={30}
              step={1}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              disabled={progress.isTraining}
              className="w-full accent-[#00FF00] bg-black h-1 cursor-pointer disabled:opacity-50"
            />
            <p className="text-[10px] font-mono uppercase text-white/45 tracking-tight">
              Passes through model layers per batch cycle.
            </p>
          </div>

          {/* Samples per Digit slider */}
          <div className="flex flex-col gap-1.5 mt-1">
            <div className="flex justify-between text-xs font-mono uppercase text-white/70">
              <span>Samples per Digit:</span>
              <span className="text-[#00FF00] font-bold">[{samplesPerDigit}]</span>
            </div>
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              value={samplesPerDigit}
              onChange={(e) => setSamplesPerDigit(Number(e.target.value))}
              disabled={progress.isTraining}
              className="w-full accent-[#00FF00] bg-black h-1 cursor-pointer disabled:opacity-50"
            />
            <p className="text-[10px] font-mono uppercase text-white/45 tracking-tight">
              Samples generated per digit label. Total dataset batch: {samplesPerDigit * 10}.
            </p>
          </div>

          {/* Learning Rate Select */}
          <div className="flex items-center justify-between gap-4 mt-2">
            <span className="text-xs text-white/70 font-mono uppercase tracking-tight">Learning Rate:</span>
            <select
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              disabled={progress.isTraining}
              className="bg-black text-[#00FF00] border border-[#00FF00]/40 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-[#00FF00] disabled:opacity-50 cursor-pointer"
            >
              <option value={0.001}>0.001 (Slow / Precise)</option>
              <option value={0.01}>0.01 (Balanced)</option>
              <option value={0.05}>0.05 (Fast / Aggressive)</option>
              <option value={0.1}>0.1 (Very Fast)</option>
            </select>
          </div>
        </div>

        {/* Trigger Train Button */}
        <button
          onClick={handleTrainClick}
          disabled={progress.isTraining}
          className={`flex items-center justify-center gap-2 py-3 px-4 font-mono font-bold text-xs uppercase border-2 transition-all cursor-pointer mt-2 ${
            progress.isTraining
              ? "bg-[#111111] text-white/30 border-white/10 cursor-not-allowed"
              : "bg-[#00FF00] text-black border-transparent hover:bg-black hover:text-[#00FF00] hover:border-[#00FF00]"
          }`}
        >
          {progress.isTraining ? (
            <>
              <RefreshCw className="h-3.5 w-3.5 animate-spin text-[#00FF00]" />
              Training (Epoch {progress.epoch}/{progress.totalEpochs})...
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" />
              Train Neural Network
            </>
          )}
        </button>

        {/* Live Training Status Card */}
        {progress.isTraining && (
          <div className="bg-black border border-[#00FF00]/40 p-3 animate-pulse">
            <div className="flex justify-between items-center text-[10px] font-mono mb-1.5 text-[#00FF00] uppercase">
              <span>Epoch Progress:</span>
              <span>{Math.round((progress.epoch / progress.totalEpochs) * 100)}%</span>
            </div>
            <div className="w-full bg-slate-900 h-1">
              <div
                style={{ width: `${(progress.epoch / progress.totalEpochs) * 100}%` }}
                className="bg-[#00FF00] h-full"
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono uppercase text-white/60 mt-2">
              <span>Loss: <b className="text-[#00FF00] font-bold">{progress.loss.toFixed(4)}</b></span>
              <span>Acc: <b className="text-[#00FF00] font-bold">{(progress.accuracy * 100).toFixed(1)}%</b></span>
            </div>
          </div>
        )}
      </div>

      {/* Dataset & Custom Training Data Feed */}
      <div className="bg-[#111111] border-2 border-[#00FF00] p-6 flex flex-col gap-4 shadow-[4px_4px_0px_0px_rgba(0,255,0,0.3)]">
        <div className="flex items-center justify-between border-b border-[#00FF00]/30 pb-3">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-[#00FF00]" />
            <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-[#00FF00]">Interactive Dataset</h3>
          </div>
          <div className="text-[10px] font-mono text-[#00FF00] bg-black border border-[#00FF00]/40 px-2 py-0.5">
            ACTIVE_DATASET
          </div>
        </div>

        {/* Dataset Stats */}
        <div className="grid grid-cols-2 gap-3 bg-black p-3.5 border border-white/10">
          <div className="flex flex-col gap-0.5">
            <span className="text-[8px] font-mono uppercase tracking-wider text-white/50">Total Training samples</span>
            <span className="text-base font-bold text-[#00FF00] font-mono">{totalDatasetSize} SAMPLES</span>
          </div>
          <div className="flex flex-col gap-0.5 border-l border-[#00FF00]/20 pl-3">
            <span className="text-[8px] font-mono uppercase tracking-wider text-white/50">Custom Drawings</span>
            <span className="text-base font-bold text-[#00FF00] font-mono">{customSamplesCount} SAMPLES</span>
          </div>
        </div>

        {/* Custom digit feedback adder */}
        <div className="flex flex-col gap-3">
          <p className="text-[10px] font-mono uppercase text-white/60 leading-relaxed tracking-tight">
            Draw custom digits on the canvas above, select the corresponding tag index, and feed it into the active tensor block.
          </p>

          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase text-white/50">Label:</span>
            <div className="flex flex-wrap gap-1 flex-1 justify-center bg-black p-1.5 border border-white/10">
              {Array.from({ length: 10 }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedLabel(i)}
                  className={`w-6 h-6 text-[10px] font-mono transition-all font-bold cursor-pointer border ${
                    selectedLabel === i
                      ? "bg-[#00FF00] text-black border-[#00FF00] font-black"
                      : "bg-black text-[#00FF00]/60 border-white/10 hover:border-[#00FF00]"
                  }`}
                >
                  {i}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => onAddCustomSample(selectedLabel)}
              className="flex-1 flex items-center justify-center gap-2 py-2 px-3 border-2 border-transparent bg-[#00FF00] text-black hover:bg-black hover:text-[#00FF00] hover:border-[#00FF00] text-xs font-mono font-bold uppercase transition-all cursor-pointer"
            >
              <Plus className="h-3.5 w-3.5" />
              Inject Sample
            </button>
            {customSamplesCount > 0 && (
              <button
                onClick={onClearCustomSamples}
                className="flex items-center justify-center gap-2 py-2 px-3 border-2 border-white/20 text-white hover:bg-white hover:text-black transition-all text-xs font-mono font-bold uppercase cursor-pointer"
                title="Reset custom samples"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Render the History Chart */}
        {renderHistoryChart()}
      </div>
    </div>
  );
}

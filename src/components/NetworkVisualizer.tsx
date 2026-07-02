/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Prediction } from "../types";
import { Brain, Sparkles } from "lucide-react";

interface NetworkVisualizerProps {
  pixels: Float32Array;
  predictions: Prediction[];
  modelWeights?: {
    h1Weights?: number[][]; // 784 x 16 (we can sample a subset or use them)
    h1Biases?: number[];
    h2Weights?: number[][]; // 16 x 16
    h2Biases?: number[];
    outWeights?: number[][]; // 16 x 10
    outBiases?: number[];
  };
  // Activations of hidden layers computed during prediction
  activations?: {
    h1?: number[]; // length 16
    h2?: number[]; // length 16
    out?: number[]; // length 10
  };
}

export function NetworkVisualizer({
  pixels,
  predictions,
  modelWeights,
  activations,
}: NetworkVisualizerProps) {
  const [hoveredNode, setHoveredNode] = useState<{
    layer: "input" | "hidden1" | "hidden2" | "output";
    index: number;
    info: string;
  } | null>(null);

  // Fallback / mock activations if model is not loaded/predicting yet
  const activeInputIndices = [];
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] > 0.1) {
      activeInputIndices.push(i);
    }
  }

  // Sample 8 representative inputs from the 784 pixels to display on the SVG
  // Choose coordinates corresponding to standard digit drawing areas (center, loop areas)
  const sampledInputCoords = [
    { x: 14, y: 14, idx: 14 * 28 + 14, name: "Center (14,14)" },
    { x: 7, y: 14, idx: 7 * 28 + 14, name: "Top-Center (7,14)" },
    { x: 21, y: 14, idx: 21 * 28 + 14, name: "Bottom-Center (21,14)" },
    { x: 14, y: 7, idx: 14 * 28 + 7, name: "Left-Center (14,7)" },
    { x: 14, y: 21, idx: 14 * 28 + 21, name: "Right-Center (14,21)" },
    { x: 10, y: 10, idx: 10 * 28 + 10, name: "Top-Left (10,10)" },
    { x: 10, y: 18, idx: 10 * 28 + 18, name: "Top-Right (10,18)" },
    { x: 18, y: 10, idx: 18 * 28 + 10, name: "Bottom-Left (18,10)" },
  ];

  // Hidden Layer Nodes counts to draw
  const h1Count = 8;
  const h2Count = 8;
  const outCount = 10;

  // Derive activation values for drawing
  const h1Activations = activations?.h1?.slice(0, h1Count) || Array(h1Count).fill(0.1);
  const h2Activations = activations?.h2?.slice(0, h2Count) || Array(h2Count).fill(0.1);
  // Output activations based on predictions
  const outActivations = Array(outCount).fill(0);
  predictions.forEach((p) => {
    if (p.label >= 0 && p.label < outCount) {
      outActivations[p.label] = p.probability;
    }
  });

  // Winning output index
  const winningIndex = predictions.length > 0 && predictions[0].probability > 0.1 ? predictions[0].label : -1;

  // Node position configuration for SVG mapping
  const svgWidth = 720;
  const svgHeight = 360;

  const getPos = (layer: "input" | "hidden1" | "hidden2" | "output", index: number) => {
    const xGap = svgWidth / 4;
    
    switch (layer) {
      case "input":
        return {
          x: 40,
          y: 40 + (index * (svgHeight - 80)) / (sampledInputCoords.length - 1),
        };
      case "hidden1":
        return {
          x: 40 + xGap,
          y: 35 + (index * (svgHeight - 70)) / (h1Count - 1),
        };
      case "hidden2":
        return {
          x: 40 + xGap * 2,
          y: 35 + (index * (svgHeight - 70)) / (h2Count - 1),
        };
      case "output":
        return {
          x: svgWidth - 40,
          y: 25 + (index * (svgHeight - 50)) / (outCount - 1),
        };
    }
  };

  return (
    <div className="bg-[#111111] border-2 border-[#00FF00] p-6 w-full shadow-[4px_4px_0px_0px_rgba(0,255,0,0.3)] flex flex-col gap-6">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#00FF00]/30 pb-4">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-[#00FF00] animate-pulse" />
          <div>
            <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-[#00FF00]">Live Neural Graph</h3>
            <p className="text-[10px] font-mono text-white/50 uppercase tracking-tight">
              Real-time feed-forward propagation visualizer.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-black border border-[#00FF00]/40 px-3 py-1.5 self-start md:self-auto">
          <Sparkles className="h-3.5 w-3.5 text-[#00FF00]" />
          <span className="text-[10px] font-mono uppercase text-[#00FF00]">
            {activeInputIndices.length} ACTIVE CHANNELS
          </span>
        </div>
      </div>

      {/* Interactive SVG Stage */}
      <div className="relative w-full overflow-x-auto overflow-y-hidden bg-black border border-white/10 p-4 scrollbar-thin">
        <svg
          width={svgWidth}
          height={svgHeight}
          className="mx-auto block select-none overflow-visible"
        >
          <defs>
            {/* Glow filters for active items */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="heavy-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            {/* Line Gradients */}
            <linearGradient id="active-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00FF00" stopOpacity="0.1" />
              <stop offset="50%" stopColor="#00FF00" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#00FF00" stopOpacity="0.8" />
            </linearGradient>
            <linearGradient id="inactive-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#111111" stopOpacity="0.1" />
              <stop offset="100%" stopColor="#111111" stopOpacity="0.1" />
            </linearGradient>
          </defs>

          {/* LAYER CONNECTIONS (Background weight lines) */}
          
          {/* Input Layer to Hidden Layer 1 */}
          {sampledInputCoords.map((input, inIdx) => {
            const inPos = getPos("input", inIdx);
            const val = pixels[input.idx] || 0;
            const isInputActive = val > 0.15;

            return h1Activations.map((h1Val, h1Idx) => {
              const h1Pos = getPos("hidden1", h1Idx);
              const customWeight = modelWeights?.h1Weights ? modelWeights.h1Weights[input.idx]?.[h1Idx] || 0 : 0.5;
              const weightStrength = Math.abs(customWeight);
              const isActiveConnection = isInputActive && h1Val > 0.2;

              return (
                <line
                  key={`in-h1-${inIdx}-${h1Idx}`}
                  x1={inPos.x}
                  y1={inPos.y}
                  x2={h1Pos.x}
                  y2={h1Pos.y}
                  stroke={isActiveConnection ? "url(#active-grad)" : "#222222"}
                  strokeWidth={isActiveConnection ? 1.5 + weightStrength * 2 : 0.4}
                  strokeOpacity={isActiveConnection ? 0.7 : 0.1}
                  className="transition-all duration-300"
                />
              );
            });
          })}

          {/* Hidden Layer 1 to Hidden Layer 2 */}
          {h1Activations.map((h1Val, h1Idx) => {
            const h1Pos = getPos("hidden1", h1Idx);
            const isH1Active = h1Val > 0.25;

            return h2Activations.map((h2Val, h2Idx) => {
              const h2Pos = getPos("hidden2", h2Idx);
              const customWeight = modelWeights?.h2Weights ? modelWeights.h2Weights[h1Idx]?.[h2Idx] || 0 : 0.4;
              const weightStrength = Math.abs(customWeight);
              const isActiveConnection = isH1Active && h2Val > 0.25;

              return (
                <line
                  key={`h1-h2-${h1Idx}-${h2Idx}`}
                  x1={h1Pos.x}
                  y1={h1Pos.y}
                  x2={h2Pos.x}
                  y2={h2Pos.y}
                  stroke={isActiveConnection ? "#00FF00" : "#222222"}
                  strokeWidth={isActiveConnection ? 1.2 + weightStrength * 1.5 : 0.4}
                  strokeOpacity={isActiveConnection ? 0.6 : 0.08}
                  className="transition-all duration-300"
                />
              );
            });
          })}

          {/* Hidden Layer 2 to Output Layer */}
          {h2Activations.map((h2Val, h2Idx) => {
            const h2Pos = getPos("hidden2", h2Idx);
            const isH2Active = h2Val > 0.25;

            return outActivations.map((outVal, outIdx) => {
              const outPos = getPos("output", outIdx);
              const customWeight = modelWeights?.outWeights ? modelWeights.outWeights[h2Idx]?.[outIdx] || 0 : 0.6;
              const weightStrength = Math.abs(customWeight);
              const isActiveConnection = isH2Active && outIdx === winningIndex;

              return (
                <line
                  key={`h2-out-${h2Idx}-${outIdx}`}
                  x1={h2Pos.x}
                  y1={h2Pos.y}
                  x2={outPos.x}
                  y2={outPos.y}
                  stroke={isActiveConnection ? "#00FF00" : "#222222"}
                  strokeWidth={isActiveConnection ? 2.2 + weightStrength * 2.5 : 0.3}
                  strokeOpacity={isActiveConnection ? 0.8 : 0.06}
                  className="transition-all duration-300"
                  filter={isActiveConnection ? "url(#glow)" : undefined}
                />
              );
            });
          })}

          {/* NODES LAYER (Foreground) */}

          {/* 1. Sampled Input Nodes */}
          {sampledInputCoords.map((input, idx) => {
            const pos = getPos("input", idx);
            const val = pixels[input.idx] || 0;
            const isHovered = hoveredNode?.layer === "input" && hoveredNode?.index === idx;

            return (
              <g
                key={`node-in-${idx}`}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHoveredNode({
                    layer: "input",
                    index: idx,
                    info: `${input.name}: activation: ${val.toFixed(3)} based on pixel opacity.`,
                  })
                }
                onMouseLeave={() => setHoveredNode(null)}
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={isHovered ? 8 : 5}
                  fill={val > 0.1 ? "#00FF00" : "#111111"}
                  stroke={isHovered ? "#00FF00" : "rgba(0, 255, 0, 0.4)"}
                  strokeWidth={1}
                  filter={val > 0.3 ? "url(#glow)" : undefined}
                  opacity={val > 0.1 ? 0.4 + val * 0.6 : 0.4}
                  className="transition-all duration-200"
                />
                {/* Visual coordinate indicator on hover */}
                {isHovered && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={12}
                    fill="none"
                    stroke="#00FF00"
                    strokeWidth={0.5}
                    strokeDasharray="2,2"
                    className="animate-spin"
                  />
                )}
              </g>
            );
          })}

          {/* 2. Hidden Layer 1 Nodes */}
          {h1Activations.map((val, idx) => {
            const pos = getPos("hidden1", idx);
            const bias = modelWeights?.h1Biases?.[idx] || 0;
            const isHovered = hoveredNode?.layer === "hidden1" && hoveredNode?.index === idx;

            return (
              <g
                key={`node-h1-${idx}`}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHoveredNode({
                    layer: "hidden1",
                    index: idx,
                    info: `Hidden Node 1-#${idx}: current activation: ${val.toFixed(3)}, local bias: ${bias.toFixed(3)}.`,
                  })
                }
                onMouseLeave={() => setHoveredNode(null)}
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={isHovered ? 9 : 6}
                  fill={val > 0.25 ? "#00FF00" : "#111111"}
                  stroke={isHovered ? "#FFFFFF" : "rgba(0, 255, 0, 0.4)"}
                  strokeWidth={1}
                  filter={val > 0.4 ? "url(#glow)" : undefined}
                  className="transition-all duration-200"
                />
              </g>
            );
          })}

          {/* 3. Hidden Layer 2 Nodes */}
          {h2Activations.map((val, idx) => {
            const pos = getPos("hidden2", idx);
            const bias = modelWeights?.h2Biases?.[idx] || 0;
            const isHovered = hoveredNode?.layer === "hidden2" && hoveredNode?.index === idx;

            return (
              <g
                key={`node-h2-${idx}`}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHoveredNode({
                    layer: "hidden2",
                    index: idx,
                    info: `Hidden Node 2-#${idx}: current activation: ${val.toFixed(3)}, local bias: ${bias.toFixed(3)}.`,
                  })
                }
                onMouseLeave={() => setHoveredNode(null)}
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={isHovered ? 9 : 6}
                  fill={val > 0.25 ? "#00FF00" : "#111111"}
                  stroke={isHovered ? "#FFFFFF" : "rgba(0, 255, 0, 0.4)"}
                  strokeWidth={1}
                  filter={val > 0.4 ? "url(#glow)" : undefined}
                  className="transition-all duration-200"
                />
              </g>
            );
          })}

          {/* 4. Output Layer Nodes */}
          {outActivations.map((val, idx) => {
            const pos = getPos("output", idx);
            const isWinner = idx === winningIndex;
            const isHovered = hoveredNode?.layer === "output" && hoveredNode?.index === idx;
            const bias = modelWeights?.outBiases?.[idx] || 0;

            return (
              <g
                key={`node-out-${idx}`}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHoveredNode({
                    layer: "output",
                    index: idx,
                    info: `Output Node [Digit ${idx}]: probability: ${(val * 100).toFixed(1)}%, bias: ${bias.toFixed(3)}.`,
                  })
                }
                onMouseLeave={() => setHoveredNode(null)}
              >
                {isWinner && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={16}
                    fill="none"
                    stroke="#00FF00"
                    strokeWidth={1}
                    className="animate-ping"
                    opacity={0.5}
                  />
                )}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={isWinner ? 11 : isHovered ? 9 : 7}
                  fill={isWinner ? "#00FF00" : val > 0.1 ? "rgba(0, 255, 0, 0.6)" : "#111111"}
                  stroke={isWinner ? "#FFFFFF" : isHovered ? "#00FF00" : "rgba(0, 255, 0, 0.5)"}
                  strokeWidth={isWinner ? 2 : 1.5}
                  filter={isWinner ? "url(#heavy-glow)" : val > 0.2 ? "url(#glow)" : undefined}
                  className="transition-all duration-300"
                />
                {/* Digit label */}
                <text
                  x={pos.x + 16}
                  y={pos.y + 4}
                  fill={isWinner ? "#00FF00" : "rgba(255, 255, 255, 0.6)"}
                  fontSize={isWinner ? "13px" : "11px"}
                  fontWeight={isWinner ? "bold" : "normal"}
                  className="font-mono text-[11px] select-none uppercase"
                >
                  [{idx}]
                </text>
              </g>
            );
          })}

          {/* Layer Headers */}
          <text x={40} y={15} fill="rgba(0, 255, 0, 0.5)" fontSize="8px" fontWeight="bold" className="font-mono text-center tracking-wider">
            IN_PIXELS
          </text>
          <text x={40 + svgWidth / 4} y={15} fill="rgba(0, 255, 0, 0.5)" fontSize="8px" fontWeight="bold" className="font-mono text-center tracking-wider">
            HIDDEN_L1
          </text>
          <text x={40 + (svgWidth / 4) * 2} y={15} fill="rgba(0, 255, 0, 0.5)" fontSize="8px" fontWeight="bold" className="font-mono text-center tracking-wider">
            HIDDEN_L2
          </text>
          <text x={svgWidth - 60} y={15} fill="rgba(0, 255, 0, 0.5)" fontSize="8px" fontWeight="bold" className="font-mono text-center tracking-wider">
            OUT_DIGIT
          </text>
        </svg>
      </div>

      {/* Node Info Card */}
      <div className="w-full bg-black border border-white/10 p-3 min-h-[64px] flex items-center justify-between">
        {hoveredNode ? (
          <div className="flex items-center gap-3 text-[#00FF00] text-xs">
            <span className="inline-block px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider bg-black border border-[#00FF00] text-[#00FF00]">
              {hoveredNode.layer}
            </span>
            <p className="font-mono text-[11px] uppercase tracking-tight">{hoveredNode.info}</p>
          </div>
        ) : (
          <div className="text-center w-full">
            <p className="text-[10px] text-white/40 font-mono uppercase tracking-widest">
              Hover graph channels to dump weight weights, biases & activation levels.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

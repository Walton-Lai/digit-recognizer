/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { MentorLesson } from "../types";
import { BookOpen, HelpCircle, GraduationCap, Percent, Code, Activity, Layers, Network } from "lucide-react";

const LESSONS: MentorLesson[] = [
  {
    id: "what-is-neuron",
    title: "What is an Artificial Neuron?",
    category: "basics",
    summary: "The building blocks of Deep Learning: nodes, inputs, weights, and activation functions.",
    content: `An **Artificial Neuron** (or Perceptron) is the fundamental unit of a neural network. It mimics biological neurons by receiving inputs, processing them, and producing an output signal.

### How a Neuron Calculates Output:
1. **Inputs ($x_i$):** These are features or pixels. For our 28x28 digit canvas, there are 784 inputs ($x_1$ to $x_{784}$), representing each pixel's brightness (0 = black, 1 = white).
2. **Weights ($w_i$):** Every input has an associated "weight" which controls how important that input is to the neuron. If weight is positive and large, that pixel strongly excites the neuron. If negative, it dampens it.
3. **Bias ($b$):** The bias is a baseline offset. It controls how easy or hard it is for the neuron to "fire," regardless of the inputs.
4. **Weighted Sum ($z$):** The neuron multiplies each input by its weight, sums them up, and adds the bias:
   $$z = (x_1 \\cdot w_1) + (x_2 \\cdot w_2) + ... + (x_n \\cdot w_n) + b$$
5. **Activation Function ($f$):** To introduce non-linearity (allowing the network to learn complex curves instead of just straight lines), the sum $z$ is passed into an activation function like **ReLU** ($max(0, z)$) or **Sigmoid** ($1 / (1 + e^{-z})$).

Play with the **Live Neuron Simulator** below to see this math in action!`,
  },
  {
    id: "downsampling",
    title: "How the AI 'Sees': Downsampling",
    category: "architecture",
    summary: "Converting a high-resolution 280x280 sketch into a compact 28x28 mathematical input grid.",
    content: `When you draw a digit on the 280x280 sketchpad, the drawing is very high-resolution. Processing a $280 \\times 280 = 78,400$ pixel grid directly would require a massive, slow model.

To make prediction real-time, efficient, and 100% server-free, we downscale the drawing to a **28x28 pixel grid** (784 total inputs). This matches the standard format of the famous **MNIST database** (the gold standard dataset of handwritten digits).

### The Steps in Detail:
1. **Resolution Scale-Down:** We draw your 280x280 canvas onto a hidden 28x28 canvas using hardware-accelerated bilinear interpolation.
2. **Grayscale Normalization:** Each pixel's color is read back. Since we draw white strokes on a black canvas, we map the pixel intensity from standard RGB channel colors into a single numeric scale: from **0.0 (fully black/background)** to **1.0 (fully white/stroke center)**.
3. **Vectorization:** The 28x28 grid is flattened into a single one-dimensional array of 784 decimal values, which is fed directly into the input layer of our neural network!`,
  },
  {
    id: "forward-propagation",
    title: "Forward Propagation: The Signal Flow",
    category: "math",
    summary: "How numbers cascade through layers to reach a final classification decision.",
    content: `**Forward Propagation** is the process where input signals travel forward through hidden layers to produce an output prediction.

In our Digit Recognizer AI, the network architecture is structured as follows:
- **Input Layer:** 784 nodes (one for each pixel in the 28x28 downscaled grid).
- **Hidden Layer 1:** 16 neurons. These detect low-level geometric features like horizontal bars, vertical edges, loops, or diagonal strokes.
- **Hidden Layer 2:** 16 neurons. These assemble the features from Layer 1 into larger components (e.g., combining a curve on top and a straight line on the bottom to identify a "2").
- **Output Layer:** 10 neurons (one for each digit from 0 to 9). 

The output values are converted to probabilities using a **Softmax** activation function, ensuring all 10 outputs sum to exactly 1.0 (100%). The node with the highest probability is declared the AI's final prediction!`,
  },
  {
    id: "backpropagation-training",
    title: "How the AI Learns: Backpropagation",
    category: "training",
    summary: "How gradient descent and loss calculations adjust weights to improve accuracy.",
    content: `How does the neural network know which weights and biases are correct? It **learns** them through trial, error, and calculus!

### The Training Loop:
1. **Predict (Forward Pass):** The network makes a prediction on a training image. Initially, because weights are randomized, the prediction is pure guesswork.
2. **Calculate Loss (Cost):** We measure how wrong the network was using a **Loss Function** (such as Cross-Entropy). If the image is a "7" and the AI predicts "3", the loss is very high. If it correctly predicts "7" with high confidence, the loss is close to zero.
3. **Propagate Backwards (Backpropagation):** Using calculus (the chain rule), we calculate how changing each individual weight and bias will increase or decrease the overall loss. This direction of change is called the **Gradient**.
4. **Adjust (Gradient Descent):** We nudge all weights and biases in the direction that lowers the loss. We multiply this adjustment by a **Learning Rate** (e.g., $\\eta = 0.01$) to make small, stable improvements.
5. **Repeat:** By repeating this over thousands of iterations (epochs), the loss falls, the accuracy climbs, and the model begins to recognize handwritten digits with high fidelity!`,
  },
];

export function MentorPanel() {
  const [selectedLessonId, setSelectedLessonId] = useState<string>("what-is-neuron");
  
  // Neuron Simulator States
  const [x1, setX1] = useState(0.8);
  const [x2, setX2] = useState(0.2);
  const [x3, setX3] = useState(0.9);

  const [w1, setW1] = useState(1.2);
  const [w2, setW2] = useState(-1.5);
  const [w3, setW3] = useState(0.8);

  const [bias, setBias] = useState(-0.3);
  const [activationType, setActivationType] = useState<"relu" | "sigmoid" | "step">("relu");

  const currentLesson = LESSONS.find((l) => l.id === selectedLessonId) || LESSONS[0];

  // Mathematical Calculations
  const weightedSum = x1 * w1 + x2 * w2 + x3 * w3 + bias;
  
  let activationOutput = 0;
  if (activationType === "relu") {
    activationOutput = Math.max(0, weightedSum);
  } else if (activationType === "sigmoid") {
    activationOutput = 1 / (1 + Math.exp(-weightedSum));
  } else {
    activationOutput = weightedSum >= 0 ? 1 : 0;
  }

  return (
    <div className="bg-[#111111] border-2 border-[#00FF00] p-6 w-full shadow-[4px_4px_0px_0px_rgba(0,255,0,0.3)] flex flex-col gap-6">
      {/* Title */}
      <div className="flex items-center gap-2 border-b border-[#00FF00]/30 pb-4">
        <GraduationCap className="h-5 w-5 text-[#00FF00] animate-bounce" />
        <div>
          <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-[#00FF00]">AI Classroom Console</h3>
          <p className="text-[10px] font-mono text-white/50 uppercase tracking-tight">
            Interactive, self-contained educational resources on deep learning mathematics.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Lesson Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-2">
          <span className="text-[10px] font-mono font-bold uppercase text-white/40 tracking-wider mb-1 px-1">
            Lesson Index:
          </span>
          {LESSONS.map((lesson) => {
            const isSelected = lesson.id === selectedLessonId;
            return (
              <button
                key={lesson.id}
                onClick={() => setSelectedLessonId(lesson.id)}
                className={`w-full text-left p-3 border transition-all cursor-pointer uppercase font-mono text-[11px] ${
                  isSelected
                    ? "bg-[#00FF00] border-[#00FF00] text-black font-bold shadow-[2px_2px_0px_0px_rgba(255,255,255,0.15)]"
                    : "bg-black border-white/10 hover:border-[#00FF00] text-[#00FF00]/80 hover:text-[#00FF00]"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {lesson.category === "basics" && <BookOpen className="h-3 w-3 text-white/50" />}
                  {lesson.category === "architecture" && <Layers className="h-3 w-3 text-white/50" />}
                  {lesson.category === "math" && <Percent className="h-3 w-3 text-white/50" />}
                  {lesson.category === "training" && <Activity className="h-3 w-3 text-white/50" />}
                  <span className={`text-[8px] font-bold capitalize tracking-wide ${isSelected ? "text-black/60" : "text-white/40"}`}>
                    [{lesson.category}]
                  </span>
                </div>
                <h4 className="font-bold leading-snug tracking-tighter">
                  {lesson.title}
                </h4>
              </button>
            );
          })}
        </div>

        {/* Lesson Body */}
        <div className="lg:col-span-8 bg-black border border-white/10 p-5 md:p-6 min-h-[320px] flex flex-col gap-4">
          <div className="border-b border-white/10 pb-3">
            <h3 className="font-mono font-bold text-sm uppercase text-[#00FF00] tracking-wide">
              {currentLesson.title}
            </h3>
            <p className="text-[10px] font-mono uppercase text-white/40 mt-1">
              SUMMARY: {currentLesson.summary}
            </p>
          </div>

          <div className="text-[11px] font-mono text-white/70 leading-relaxed space-y-3 whitespace-pre-line tracking-tight">
            {currentLesson.content}
          </div>
        </div>
      </div>

      {/* Neuron Calculator Interactive Playground */}
      <div className="border-t border-[#00FF00]/20 pt-6 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Code className="h-4 w-4 text-[#00FF00]" />
          <h4 className="font-mono font-bold text-xs uppercase tracking-wider text-[#00FF00]">
            Live Neuron Simulator (Hacker Sandbox)
          </h4>
        </div>

        <p className="text-[10px] font-mono uppercase text-white/50 leading-relaxed tracking-tight">
          Adjust the weights and biases manually below to simulate backpropagation gradients in real-time.
        </p>

        {/* Simulator Area */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-black border border-white/10 p-5">
          {/* Sliders Side */}
          <div className="lg:col-span-7 flex flex-col gap-3.5">
            {/* Input 1 & Weight 1 */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Input x1:</span>
                  <span className="text-[#00FF00] font-bold">[{x1.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={x1}
                  onChange={(e) => setX1(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Weight w1:</span>
                  <span className="text-[#00FF00] font-bold">[{w1.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="-2.0"
                  max="2.0"
                  step="0.1"
                  value={w1}
                  onChange={(e) => setW1(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
            </div>

            {/* Input 2 & Weight 2 */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Input x2:</span>
                  <span className="text-[#00FF00] font-bold">[{x2.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={x2}
                  onChange={(e) => setX2(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Weight w2:</span>
                  <span className="text-[#00FF00] font-bold">[{w2.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="-2.0"
                  max="2.0"
                  step="0.1"
                  value={w2}
                  onChange={(e) => setW2(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
            </div>

            {/* Input 3 & Weight 3 */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Input x3:</span>
                  <span className="text-[#00FF00] font-bold">[{x3.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={x3}
                  onChange={(e) => setX3(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Weight w3:</span>
                  <span className="text-[#00FF00] font-bold">[{w3.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="-2.0"
                  max="2.0"
                  step="0.1"
                  value={w3}
                  onChange={(e) => setW3(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>
            </div>

            {/* Bias & Activation selection */}
            <div className="grid grid-cols-2 gap-4 border-t border-white/10 pt-3">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] font-mono uppercase text-white/50">
                  <span>Bias b:</span>
                  <span className="text-[#00FF00] font-bold">[{bias.toFixed(2)}]</span>
                </div>
                <input
                  type="range"
                  min="-2.0"
                  max="2.0"
                  step="0.1"
                  value={bias}
                  onChange={(e) => setBias(Number(e.target.value))}
                  className="w-full accent-[#00FF00] bg-[#111] h-1 cursor-pointer"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-[10px] font-mono uppercase text-white/50">Activation f:</span>
                <div className="flex gap-1.5">
                  {(["relu", "sigmoid", "step"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setActivationType(t)}
                      className={`flex-1 py-1 border text-[9px] font-mono font-bold transition-all uppercase cursor-pointer ${
                        activationType === t
                          ? "bg-[#00FF00] border-[#00FF00] text-black"
                          : "bg-black text-[#00FF00]/60 border-white/10 hover:border-[#00FF00]"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Visual Calculation Node Side */}
          <div className="lg:col-span-5 border-l border-white/10 pl-0 lg:pl-6 flex flex-col items-center justify-center gap-4 py-4 lg:py-0">
            {/* Live Diagram */}
            <div className="relative flex items-center justify-center h-28 w-full">
              {/* Weight lines to Center node */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <svg className="w-full h-full">
                  <line x1="20" y1="20" x2="130" y2="56" stroke={w1 >= 0 ? "#00FF00" : "#FF3333"} strokeWidth={Math.abs(w1) * 2} strokeOpacity="0.4" />
                  <line x1="20" y1="56" x2="130" y2="56" stroke={w2 >= 0 ? "#00FF00" : "#FF3333"} strokeWidth={Math.abs(w2) * 2} strokeOpacity="0.4" />
                  <line x1="20" y1="92" x2="130" y2="56" stroke={w3 >= 0 ? "#00FF00" : "#FF3333"} strokeWidth={Math.abs(w3) * 2} strokeOpacity="0.4" />
                  
                  {/* Output line */}
                  <line x1="170" y1="56" x2="260" y2="56" stroke="#00FF00" strokeWidth={Math.max(0.5, activationOutput * 3)} strokeOpacity={0.6} />
                </svg>
              </div>

              {/* Inputs */}
              <div className="absolute left-2 flex flex-col justify-between h-full py-1">
                <div className="w-6 h-6 rounded-full bg-black flex items-center justify-center text-[8px] font-mono font-bold text-[#00FF00] border border-[#00FF00]/40" style={{ opacity: 0.4 + x1 * 0.6 }}>x1</div>
                <div className="w-6 h-6 rounded-full bg-black flex items-center justify-center text-[8px] font-mono font-bold text-[#00FF00] border border-[#00FF00]/40" style={{ opacity: 0.4 + x2 * 0.6 }}>x2</div>
                <div className="w-6 h-6 rounded-full bg-black flex items-center justify-center text-[8px] font-mono font-bold text-[#00FF00] border border-[#00FF00]/40" style={{ opacity: 0.4 + x3 * 0.6 }}>x3</div>
              </div>

              {/* Center Neuron */}
              <div className="absolute inset-x-0 mx-auto w-10 h-10 rounded-full bg-black flex flex-col items-center justify-center text-[8px] font-bold text-[#00FF00] border-2 border-[#00FF00] shadow-[0_0_8px_rgba(0,255,0,0.2)]">
                <span>SUM</span>
                <span className="text-[7px] font-mono">z={weightedSum.toFixed(1)}</span>
              </div>

              {/* Output Activation */}
              <div
                style={{
                  backgroundColor: `rgba(0, 255, 0, ${0.1 + activationOutput * 0.9})`,
                  boxShadow: activationOutput > 0.1 ? `0 0 12px rgba(0, 255, 0, ${activationOutput * 0.5})` : "none",
                }}
                className="absolute right-2 w-9 h-9 rounded-full bg-black flex items-center justify-center text-[8px] font-mono font-bold text-white border border-white/20 transition-all duration-300"
              >
                a={activationOutput.toFixed(2)}
              </div>
            </div>

            {/* Live Formula math block */}
            <div className="w-full bg-black border border-white/10 p-2 text-center font-mono text-[9px]">
              <span className="text-white/40">z = </span>
              <span className="text-[#00FF00]">({x1.toFixed(1)}&middot;{w1 >= 0 ? `+${w1.toFixed(1)}` : w1.toFixed(1)})</span>
              <span className="text-white/30"> + </span>
              <span className="text-[#00FF00]">({x2.toFixed(1)}&middot;{w2 >= 0 ? `+${w2.toFixed(1)}` : w2.toFixed(1)})</span>
              <span className="text-white/30"> + </span>
              <span className="text-[#00FF00]">({x3.toFixed(1)}&middot;{w3 >= 0 ? `+${w3.toFixed(1)}` : w3.toFixed(1)})</span>
              <span className="text-white/30"> + </span>
              <span className="text-white/70">({bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1)})</span>
              <span className="text-white/30"> = </span>
              <span className="text-[#00FF00] font-bold">{weightedSum.toFixed(2)}</span>
              <div className="mt-1 border-t border-white/10 pt-1">
                <span className="text-white/40">OUT a = f(z) = </span>
                <span className="text-[#00FF00] font-bold">{activationOutput.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

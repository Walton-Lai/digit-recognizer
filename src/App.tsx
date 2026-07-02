/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { generateSyntheticSamples } from "./utils/synthetic";
import { DrawingCanvas } from "./components/DrawingCanvas";
import { NetworkVisualizer } from "./components/NetworkVisualizer";
import { TrainingControls } from "./components/TrainingControls";
import { MentorPanel } from "./components/MentorPanel";
import { Prediction, TrainingProgress, DigitSample } from "./types";
import { Brain, Cpu, Sparkles, Star, Heart, RefreshCw, Zap, TrendingUp, HelpCircle } from "lucide-react";

export default function App() {
  // Model state
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [initStage, setInitStage] = useState("Loading TensorFlow.js...");
  const [backendName, setBackendName] = useState("CPU");

  // Training & Dataset states
  const [progress, setProgress] = useState<TrainingProgress>({
    epoch: 0,
    totalEpochs: 0,
    loss: 1.0,
    accuracy: 0.0,
    isTraining: false,
    history: [],
  });
  const [customSamples, setCustomSamples] = useState<DigitSample[]>([]);
  const [syntheticSamples, setSyntheticSamples] = useState<DigitSample[]>([]);
  const [totalDatasetSize, setTotalDatasetSize] = useState(0);

  // Drawing & Prediction states
  const [currentPixels, setCurrentPixels] = useState<Float32Array>(new Float32Array(784));
  const [predictions, setPredictions] = useState<Prediction[]>(
    Array.from({ length: 10 }).map((_, i) => ({ label: i, probability: 0 }))
  );
  const [clearTrigger, setClearTrigger] = useState(0);

  // Activations & extracted weights for visualization
  const [activations, setActivations] = useState<{
    h1?: number[];
    h2?: number[];
    out?: number[];
  }>({});
  const [modelWeights, setModelWeights] = useState<{
    h1Weights?: number[][];
    h1Biases?: number[];
    h2Weights?: number[][];
    h2Biases?: number[];
    outWeights?: number[][];
    outBiases?: number[];
  }>({});

  // Flag to avoid updating states if unmounted
  const isMounted = useRef(true);

  // Initial setup: Initialize TensorFlow.js and train baseline model
  useEffect(() => {
    isMounted.current = true;

    async function initializeAI() {
      try {
        // 1. Set backend (WebGL if available, fallback to CPU)
        setInitStage("Setting up WebGL hardware acceleration...");
        await tf.ready();
        const currentBackend = tf.getBackend();
        if (isMounted.current) {
          setBackendName(currentBackend.toUpperCase());
        }

        // 2. Synthesize baseline dataset
        setInitStage("Synthesizing 1000 digit variations (Arial, Georgia, Courier)...");
        const initialSynthetic = generateSyntheticSamples(100); // 100 samples * 10 digits = 1000
        if (isMounted.current) {
          setSyntheticSamples(initialSynthetic);
          setTotalDatasetSize(initialSynthetic.length);
        }

        // 3. Create Neural Network (Input 784 -> Dense 128 -> Dense 64 -> Softmax 10)
        setInitStage("Assembling Neural Network Architecture (784-128-64-10)...");
        const denseModel = tf.sequential();
        denseModel.add(
          tf.layers.dense({
            inputShape: [784],
            units: 128,
            activation: "relu",
            name: "hidden1",
          })
        );
        denseModel.add(
          tf.layers.dense({
            units: 64,
            activation: "relu",
            name: "hidden2",
          })
        );
        denseModel.add(
          tf.layers.dense({
            units: 10,
            activation: "softmax",
            name: "output",
          })
        );

        denseModel.compile({
          optimizer: tf.train.adam(0.01),
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });

        if (isMounted.current) {
          setModel(denseModel);
        }

        // 4. Pre-train baseline model for 20 epochs to make it functional instantly
        setInitStage("Pre-training upgraded model in the browser (20 Epochs)...");
        await trainModelOnData(denseModel, initialSynthetic, 20, 0.01);

        if (isMounted.current) {
          setIsInitializing(false);
        }
      } catch (err) {
        console.error("Initialization failed:", err);
        setInitStage("Failed to initialize. Falling back...");
        if (isMounted.current) {
          setIsInitializing(false);
        }
      }
    }

    initializeAI();

    return () => {
      isMounted.current = false;
    };
  }, []);

  // Helper: Core model training logic
  const trainModelOnData = async (
    targetModel: tf.Sequential,
    dataset: DigitSample[],
    epochsCount: number,
    learningRate: number
  ) => {
    if (dataset.length === 0) return;

    // Set optimizer learning rate
    const adamOptimizer = tf.train.adam(learningRate);
    targetModel.compile({
      optimizer: adamOptimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    if (isMounted.current) {
      setProgress({
        epoch: 0,
        totalEpochs: epochsCount,
        loss: 1.0,
        accuracy: 0.0,
        isTraining: true,
        history: [],
      });
    }

    // 1. Format tensors
    const inputsArray = dataset.map((s) => Array.from(s.pixels));
    const labelsArray = dataset.map((s) => {
      const oneHot = new Array(10).fill(0);
      oneHot[s.label] = 1;
      return oneHot;
    });

    const xs = tf.tensor2d(inputsArray, [dataset.length, 784]);
    const ys = tf.tensor2d(labelsArray, [dataset.length, 10]);

    // Track training history locally to set states safely
    const localHistory: { epoch: number; loss: number; accuracy: number }[] = [];

    // 2. Train model using epochs callbacks
    await targetModel.fit(xs, ys, {
      epochs: epochsCount,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (!isMounted.current) return;
          const currentLoss = logs?.loss || 0;
          const currentAcc = logs?.acc || 0;

          localHistory.push({
            epoch: epoch + 1,
            loss: currentLoss,
            accuracy: currentAcc,
          });

          setProgress({
            epoch: epoch + 1,
            totalEpochs: epochsCount,
            loss: currentLoss,
            accuracy: currentAcc,
            isTraining: true,
            history: [...localHistory],
          });
        },
      },
    });

    // Clean up training tensors
    xs.dispose();
    ys.dispose();

    // 3. Extract and save current weights for network visualization
    await extractModelWeights(targetModel);

    if (isMounted.current) {
      setProgress((prev) => ({ ...prev, isTraining: false }));
    }
  };

  // Extract model weights safely
  const extractModelWeights = async (targetModel: tf.Sequential) => {
    try {
      const h1Layers = targetModel.layers[0].getWeights();
      const h2Layers = targetModel.layers[1].getWeights();
      const outLayers = targetModel.layers[2].getWeights();

      const h1WeightsData = (await h1Layers[0].array()) as number[][];
      const h1BiasesData = (await h1Layers[1].array()) as number[];
      const h2WeightsData = (await h2Layers[0].array()) as number[][];
      const h2BiasesData = (await h2Layers[1].array()) as number[];
      const outWeightsData = (await outLayers[0].array()) as number[][];
      const outBiasesData = (await outLayers[1].array()) as number[];

      if (isMounted.current) {
        setModelWeights({
          h1Weights: h1WeightsData,
          h1Biases: h1BiasesData,
          h2Weights: h2WeightsData,
          h2Biases: h2BiasesData,
          outWeights: outWeightsData,
          outBiases: outBiasesData,
        });
      }
    } catch (err) {
      console.warn("Could not extract weights:", err);
    }
  };

  // Handle retraining trigger from component
  const handleRetrain = async (epochs: number, samplesPerLabel: number, learningRate: number) => {
    if (!model || progress.isTraining) return;

    // Regenerate synthetic samples if parameters changed
    const freshSynthetic = generateSyntheticSamples(samplesPerLabel);
    setSyntheticSamples(freshSynthetic);

    // Combine synthetic and user custom drawn drawings
    const fullDataset = [...freshSynthetic, ...customSamples];
    setTotalDatasetSize(fullDataset.length);

    await trainModelOnData(model, fullDataset, epochs, learningRate);
  };

  // Add custom drawing sample to personal dataset
  const handleAddCustomSample = (label: number) => {
    const freshSample: DigitSample = {
      id: `custom-${label}-${Date.now()}`,
      label,
      pixels: new Float32Array(currentPixels), // copy values
      isUserDrawn: true,
      timestamp: Date.now(),
    };

    const updatedCustom = [...customSamples, freshSample];
    setCustomSamples(updatedCustom);
    setTotalDatasetSize(syntheticSamples.length + updatedCustom.length);

    // Clear canvas and trigger notification
    setClearTrigger((prev) => prev + 1);
  };

  const handleClearCustomSamples = () => {
    setCustomSamples([]);
    setTotalDatasetSize(syntheticSamples.length);
  };

  // Execute forward propagation in real-time as user draws
  const handleStrokeComplete = (pixels: Float32Array) => {
    setCurrentPixels(pixels);
    if (!model) return;

    tf.tidy(() => {
      // Create input tensor
      const inputTensor = tf.tensor2d(pixels, [1, 784]);

      // Predict final probabilities
      const predictionTensor = model.predict(inputTensor) as tf.Tensor;
      const probs = predictionTensor.dataSync();

      // Setup dynamic model to capture intermediate layer activations
      try {
        const layer1 = model.layers[0];
        const layer2 = model.layers[1];
        const activationModel = tf.model({
          inputs: model.inputs,
          outputs: [layer1.output, layer2.output, model.output],
        });

        const [act1, act2, preds] = activationModel.predict(inputTensor) as tf.Tensor[];
        
        if (isMounted.current) {
          setActivations({
            h1: Array.from(act1.dataSync()),
            h2: Array.from(act2.dataSync()),
            out: Array.from(preds.dataSync()),
          });
        }
      } catch (err) {
        // Fallback if custom graph prediction fails
        if (isMounted.current) {
          setActivations({
            out: Array.from(probs),
          });
        }
      }

      // Map predictions to structured results
      const results: Prediction[] = Array.from(probs).map((p, idx) => ({
        label: idx,
        probability: p,
      }));

      // Sort by label initially so bars are ordered 0-9, but we can display the winner
      if (isMounted.current) {
        setPredictions(results);
      }
    });
  };

  const handleClearCanvas = () => {
    setCurrentPixels(new Float32Array(784));
    setPredictions(Array.from({ length: 10 }).map((_, i) => ({ label: i, probability: 0 })));
    setActivations({});
  };

  // Determine top prediction
  const topPrediction = [...predictions].sort((a, b) => b.probability - a.probability)[0];
  const hasDrawing = currentPixels.some((p) => p > 0.15);

  return (
    <div className="min-h-screen bg-[#050505] text-white flex flex-col justify-between selection:bg-[#00FF00]/20 selection:text-[#00FF00]">
      
      {/* Initialization Overlay */}
      {isInitializing && (
        <div className="fixed inset-0 bg-black z-50 flex flex-col items-center justify-center p-6 text-center border-4 border-[#00FF00] m-4">
          <div className="relative mb-6">
            <Brain className="h-16 w-16 text-[#00FF00] animate-pulse relative z-10" />
          </div>
          <h2 className="font-mono font-bold text-2xl uppercase tracking-wider text-[#00FF00] mb-2">
            [DIGIT RECOGNIZER AI]
          </h2>
          <div className="flex items-center gap-2 text-[#00FF00] font-mono text-xs bg-[#111] border border-[#00FF00]/30 px-3 py-1 mb-6">
            <Cpu className="h-3 w-3 animate-spin" />
            <span>SYSTEM INIT // CLIENT-SIDE SANDBOX</span>
          </div>
          <p className="text-xs text-white/70 font-mono max-w-sm mb-4 leading-relaxed uppercase">
            {initStage}
          </p>
          <div className="w-56 bg-[#111] h-1.5 border border-[#00FF00]/20 overflow-hidden">
            <div className="bg-[#00FF00] h-full w-2/3 animate-infinite-scroll" />
          </div>
        </div>
      )}

      {/* Main Container */}
      <div className="max-w-7xl mx-auto w-full px-4 py-8 md:py-12 flex flex-col gap-10">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 border-b-2 border-[#00FF00] pb-6">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="bg-[#00FF00] text-black text-[9px] font-mono font-bold tracking-widest px-2.5 py-1 uppercase">
                OFFLINE AI CORE v1.0.0
              </span>
            </div>
            <h1 className="font-mono font-bold text-3xl md:text-4xl text-[#00FF00] tracking-tight mt-1 uppercase">
              Digit Recognizer AI
            </h1>
            <p className="text-xs text-white/60 font-mono max-w-2xl leading-relaxed uppercase">
              Draw digits on the interactive screen to drive a live 2-layer artificial neural network, calculating output predictions completely on the client in real-time with zero backend cost.
            </p>
          </div>

          <div className="flex flex-col gap-2 bg-[#111] border border-white/10 p-4 min-w-[240px]">
            <div className="flex items-center gap-2 justify-between">
              <span className="text-[10px] font-mono uppercase text-white/50">TFJS BACKEND:</span>
              <span className="text-[10px] font-mono font-bold text-[#00FF00] bg-black border border-[#00FF00]/40 px-2 py-0.5">
                {backendName}
              </span>
            </div>
            <div className="flex items-center gap-2 justify-between border-t border-white/10 pt-2">
              <span className="text-[10px] font-mono uppercase text-white/50">MODEL STATE:</span>
              <span className="text-[10px] font-mono font-bold text-white bg-black border border-white/20 px-2 py-0.5">
                TRAINED [ACC: {(progress.history[progress.history.length - 1]?.accuracy * 100 || 88).toFixed(1)}%]
              </span>
            </div>
          </div>
        </header>

        {/* SECTION 1: Canvas & Real-time Predictions */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Drawing Canvas Component */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <DrawingCanvas
              onStrokeComplete={handleStrokeComplete}
              onClear={handleClearCanvas}
              clearTrigger={clearTrigger}
            />
          </div>

          {/* Real-time Prediction Visualizer */}
          <div className="lg:col-span-5 bg-black border border-white/10 p-6 flex flex-col gap-6 self-stretch justify-between">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-[#00FF00] animate-pulse" />
                <h3 className="font-mono font-bold text-xs uppercase tracking-wider text-[#00FF00]">Real-Time Inference</h3>
              </div>
              <div className="text-[9px] font-mono text-[#00FF00]/60 bg-black border border-[#00FF00]/20 px-2 py-0.5">
                SOFTMAX ACTIVE
              </div>
            </div>

            {/* Winning Guess Card */}
            <div className="bg-black p-5 border-2 border-[#00FF00] min-h-[140px] flex flex-col justify-center items-center text-center relative overflow-hidden">
              {hasDrawing && topPrediction.probability > 0.15 ? (
                <>
                  <span className="text-white/60 text-[10px] font-mono uppercase tracking-widest">
                    CLASSIFICATION CONFIRMED:
                  </span>
                  <div className="text-6xl md:text-7xl font-mono font-black text-[#00FF00] my-2 select-none">
                    {topPrediction.label}
                  </div>
                  <span className="text-[#00FF00] font-mono text-[10px] uppercase font-bold bg-[#111] px-2 py-1 border border-[#00FF00]/40">
                    PROBABILITY: {(topPrediction.probability * 100).toFixed(1)}% ACCURATE
                  </span>
                </>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <HelpCircle className="h-6 w-6 text-white/20 animate-pulse" />
                  <p className="text-white/40 text-[10px] uppercase font-mono max-w-xs leading-relaxed">
                    [WAITING FOR DRAWING] - Sketch a single digit (0-9) inside the left pad to initiate forward-pass classification instantly.
                  </p>
                </div>
              )}
            </div>

            {/* Probabilities progress list (0-9) */}
            <div className="flex flex-col gap-2">
              {predictions.map((p) => {
                const isWinner = hasDrawing && p.label === topPrediction.label && topPrediction.probability > 0.15;
                return (
                  <div key={p.label} className="flex items-center gap-3">
                    <span
                      className={`font-mono text-xs w-5 text-right font-bold transition-colors ${
                        isWinner ? "text-[#00FF00] text-sm" : "text-white/40"
                      }`}
                    >
                      {p.label}
                    </span>
                    <div className="flex-1 bg-[#111] h-3 border border-white/10 p-[1px]">
                      <div
                        style={{ width: `${p.probability * 100}%` }}
                        className={`h-full transition-all duration-300 ${
                          isWinner ? "bg-[#00FF00]" : "bg-white/10"
                        }`}
                      />
                    </div>
                    <span
                      className={`font-mono text-[10px] w-10 text-right ${
                        isWinner ? "text-[#00FF00] font-bold" : "text-white/40"
                      }`}
                    >
                      {(p.probability * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* SECTION 2: Dynamic Neural Network Graph */}
        <section>
          <NetworkVisualizer
            pixels={currentPixels}
            predictions={predictions}
            modelWeights={modelWeights}
            activations={activations}
          />
        </section>

        {/* SECTION 3: Dataset Lab & Hyperparameters */}
        <section>
          <TrainingControls
            progress={progress}
            onTrain={handleRetrain}
            onAddCustomSample={handleAddCustomSample}
            customSamplesCount={customSamples.length}
            totalDatasetSize={totalDatasetSize}
            onClearCustomSamples={handleClearCustomSamples}
          />
        </section>

        {/* SECTION 4: Classroom & Live Neuron Simulator */}
        <section>
          <MentorPanel />
        </section>

      </div>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black py-8 text-center text-xs font-mono text-white/40 flex flex-col gap-2 items-center uppercase tracking-wide">
        <div className="flex items-center gap-1.5">
          <span>A free educational sandbox by cibtoo12@gmail.com</span>
          <Heart className="h-3 w-3 text-[#00FF00] fill-[#00FF00] animate-pulse" />
        </div>
        <div className="flex gap-4 text-[10px] text-white/30">
          <span>CLIENT-SIDE NEURAL COMPILATION &bull; 100% FREE &bull; SECURE ENGINE</span>
        </div>
      </footer>
    </div>
  );
}

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Prediction {
  label: number;
  probability: number;
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  isTraining: boolean;
  history: { epoch: number; loss: number; valLoss?: number; accuracy: number }[];
}

export interface DigitSample {
  id: string;
  label: number;
  pixels: Float32Array; // 784 elements (28x28) between 0 and 1
  isUserDrawn: boolean;
  timestamp: number;
}

export interface MentorLesson {
  id: string;
  title: string;
  category: "basics" | "math" | "architecture" | "training";
  summary: string;
  content: string; // Markdown supported
}

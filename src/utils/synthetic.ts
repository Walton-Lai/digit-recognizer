/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DigitSample } from "../types";

// Random number in range [min, max]
function randomRange(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Generates synthetic handwritten-like digits using the browser HTML5 Canvas.
 * Draws standard fonts with various augmentations (rotation, scale, position shifts, font styles, and noise).
 */
export function generateSyntheticSamples(countPerDigit: number = 30): DigitSample[] {
  if (typeof document === "undefined") {
    return [];
  }

  const samples: DigitSample[] = [];
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  if (!ctx) {
    return [];
  }

  const fonts = [
    "sans-serif",
    "serif",
    "monospace",
    "Arial",
    "Courier New",
    "Georgia",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
    "Impact",
  ];

  for (let digit = 0; digit <= 9; digit++) {
    for (let i = 0; i < countPerDigit; i++) {
      // 1. Clear canvas
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, 28, 28);

      // 2. Set up drawing state with augmentations
      ctx.save();
      
      // Translate to center of canvas for rotation, then translate back
      const xShift = randomRange(-2.5, 2.5);
      const yShift = randomRange(-2.5, 2.5);
      ctx.translate(14 + xShift, 14 + yShift);

      // Apply random rotation (in radians)
      const angle = randomRange(-0.25, 0.25); // ~14 degrees max
      ctx.rotate(angle);

      // Font size variations
      const fontSize = Math.round(randomRange(16, 22));
      const fontName = fonts[Math.floor(Math.random() * fonts.length)];
      ctx.font = `bold ${fontSize}px ${fontName}`;

      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "white";

      // Slight scale warping
      const scaleX = randomRange(0.85, 1.15);
      const scaleY = randomRange(0.85, 1.15);
      ctx.scale(scaleX, scaleY);

      // Draw the digit
      ctx.fillText(digit.toString(), 0, 0);
      ctx.restore();

      // 3. Read back pixels and normalize
      const imgData = ctx.getImageData(0, 0, 28, 28);
      const data = imgData.data;
      const pixels = new Float32Array(784);

      for (let p = 0; p < 784; p++) {
        // Red channel (index p * 4) as grayscale representation
        let val = data[p * 4] / 255.0;

        // Apply slight noise to simulate variations
        val += randomRange(-0.05, 0.05);
        pixels[p] = Math.max(0, Math.min(1, val)); // Clamp between 0 and 1
      }

      samples.push({
        id: `synthetic-${digit}-${i}-${Date.now()}`,
        label: digit,
        pixels,
        isUserDrawn: false,
        timestamp: Date.now(),
      });
    }
  }

  // Shuffle samples using Fisher-Yates
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = samples[i];
    samples[i] = samples[j];
    samples[j] = temp;
  }

  return samples;
}

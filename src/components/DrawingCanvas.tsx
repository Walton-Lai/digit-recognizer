/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useState, useEffect } from "react";
import { Trash2, Undo, Paintbrush, CircleHelp } from "lucide-react";

interface DrawingCanvasProps {
  onStrokeComplete: (pixels: Float32Array) => void;
  onClear: () => void;
  // Allows parent to reset canvas programmatically
  clearTrigger?: number;
}

export function DrawingCanvas({
  onStrokeComplete,
  onClear,
  clearTrigger = 0,
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(18); // Recommended brush size for MNIST-like scaling
  const [previewPixels, setPreviewPixels] = useState<Float32Array>(new Float32Array(784));
  const [hoveredPixel, setHoveredPixel] = useState<{ x: number; y: number; val: number } | null>(null);
  
  // History stack for Undo
  const historyRef = useRef<ImageData[]>([]);

  // Initialize Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set background to black (MNIST standard: white drawings on black background)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Save initial blank state
    const initialData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    historyRef.current = [initialData];
  }, []);

  // Watch for external clear triggers
  useEffect(() => {
    if (clearTrigger > 0) {
      handleClear();
    }
  }, [clearTrigger]);

  // Handle Undo action
  const handleUndo = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const history = historyRef.current;
    if (history.length <= 1) {
      // Nothing to undo (only the initial state left)
      return;
    }

    const newHistory = [...history];
    newHistory.pop(); // Remove current state
    const prevState = newHistory[newHistory.length - 1]; // Previous state to restore

    ctx.putImageData(prevState, 0, 0);
    historyRef.current = newHistory;

    // Refresh downscaled pixels
    updateDownscaledPixels(true);
  };

  // Keep a stable ref to handleUndo to prevent stale closures in event listener
  const handleUndoRef = useRef(handleUndo);
  useEffect(() => {
    handleUndoRef.current = handleUndo;
  });

  // Global Keyboard Shortcut listener for Ctrl+Z / Cmd+Z
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isZ = e.key.toLowerCase() === "z";
      const isCtrlOrCmd = e.ctrlKey || e.metaKey;
      if (isZ && isCtrlOrCmd && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        handleUndoRef.current();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  // Drawing state handlers
  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();

    let clientX = 0;
    let clientY = 0;

    if ("touches" in e) {
      if (e.touches.length === 0) return null;
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    // Account for canvas scaling (rect.width/height vs canvas.width/height)
    return {
      x: ((clientX - rect.left) / rect.width) * canvas.width,
      y: ((clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    // Prevent scrolling on mobile touch
    if ("touches" in e) {
      e.preventDefault();
    }
    
    const pos = getMousePos(e);
    if (!pos) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = "#FFFFFF"; // Drawing in pure white

    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    if ("touches" in e) {
      e.preventDefault();
    }

    const pos = getMousePos(e);
    if (!pos) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    // Downscale in real-time for continuous updates
    updateDownscaledPixels();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    // Save drawn frame state to history stack
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const state = ctx.getImageData(0, 0, canvas.width, canvas.height);
        historyRef.current = [...historyRef.current, state];
      }
    }

    updateDownscaledPixels(true); // Final stroke callback
  };

  // Extract, Normalize/Center Bounding Box, and Downscale to 28x28 Grayscale (MNIST style)
  const updateDownscaledPixels = (isStrokeFinished: boolean = false) => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!canvas || !hiddenCanvas) return;

    const hCtx = hiddenCanvas.getContext("2d", { willReadFrequently: true });
    if (!hCtx) return;

    // 1. Clear hidden canvas with black background
    hCtx.fillStyle = "#000000";
    hCtx.fillRect(0, 0, 28, 28);

    // 2. MNIST Bounding Box Normalization: Scan 280x280 canvas to find non-black bounding box
    const mainCtx = canvas.getContext("2d");
    if (!mainCtx) return;
    const mainData = mainCtx.getImageData(0, 0, canvas.width, canvas.height).data;

    let minX = canvas.width;
    let maxX = 0;
    let minY = canvas.height;
    let maxY = 0;
    let found = false;

    // Scan every 2 pixels for performance (takes <0.2ms)
    for (let y = 0; y < canvas.height; y += 2) {
      for (let x = 0; x < canvas.width; x += 2) {
        const idx = (y * canvas.width + x) * 4;
        const val = mainData[idx]; // Red channel represents brightness (white on black)
        if (val > 15) { // Threshold for drawn strokes
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (found) {
      const w = maxX - minX + 1;
      const h = maxY - minY + 1;

      if (w > 4 && h > 4) {
        // Find largest dimension
        const size = Math.max(w, h);
        // MNIST digits are typically centered within a 20x20 box inside the 28x28 viewport (with 4px padding)
        // We'll target 18px for excellent margin clearance and alignment
        const targetSize = 18;
        const scale = targetSize / size;

        // Draw cropped region centered in 28x28 space
        const dx = 14 - (w * scale) / 2;
        const dy = 14 - (h * scale) / 2;

        hCtx.drawImage(
          canvas,
          minX, minY, w, h, // Source crop
          dx, dy, w * scale, h * scale // Destination centered placement
        );
      } else {
        // Fallback for extremely small/single click drawings
        hCtx.drawImage(canvas, 0, 0, 28, 28);
      }
    } else {
      // Fallback/No Drawing
      hCtx.drawImage(canvas, 0, 0, 28, 28);
    }

    // 3. Read back pixels
    const imgData = hCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;
    const pixels = new Float32Array(784);

    // Convert to normalized grayscale
    for (let i = 0; i < 784; i++) {
      const val = data[i * 4] / 255.0;
      pixels[i] = val;
    }

    setPreviewPixels(pixels);

    // Call callback with latest normalized vector
    onStrokeComplete(pixels);
  };

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear main canvas with black
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Clear preview
    const cleared = new Float32Array(784);
    setPreviewPixels(cleared);
    
    onClear();

    // Push cleared state onto history so they can UNDO a clear!
    const clearedData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    historyRef.current = [...historyRef.current, clearedData];
  };

  // Helper to draw grid overlay or highlight cells
  const handleGridHover = (e: React.MouseEvent<HTMLDivElement>, cellIndex: number) => {
    const x = cellIndex % 28;
    const y = Math.floor(cellIndex / 28);
    const val = previewPixels[cellIndex];
    setHoveredPixel({ x, y, val });
  };

  const handleGridLeave = () => {
    setHoveredPixel(null);
  };

  return (
    <div className="flex flex-col lg:flex-row items-stretch gap-8 w-full justify-center">
      {/* Draw Board Side */}
      <div className="flex flex-col items-center gap-4 bg-[#111111] border-2 border-[#00FF00] p-6 w-full max-w-sm shadow-[4px_4px_0px_0px_rgba(0,255,0,0.3)]">
        <div className="flex justify-between items-center w-full border-b border-[#00FF00]/30 pb-2">
          <div className="flex items-center gap-2">
            <Paintbrush className="h-4 w-4 text-[#00FF00]" />
            <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-[#00FF00]">Sketchpad</h3>
          </div>
          <div className="text-[10px] font-mono text-[#00FF00] bg-black border border-[#00FF00]/40 px-2 py-0.5">
            280 x 280 BUFFER
          </div>
        </div>

        {/* Drawing Surface */}
        <div className="relative border-2 border-[#00FF00] overflow-hidden cursor-crosshair bg-black">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            className="block touch-none"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
          <div className="absolute top-2 left-2 text-[8px] text-white/30 uppercase font-mono tracking-widest">
            User_Sketch_Buffer_01
          </div>
        </div>

        {/* Brush Controls */}
        <div className="flex items-center gap-4 w-full justify-between pt-1">
          <span className="text-xs text-white/70 font-mono uppercase tracking-tight">Brush Weight:</span>
          <div className="flex gap-2">
            {[12, 18, 24].map((size) => (
              <button
                key={size}
                onClick={() => setBrushSize(size)}
                className={`px-2.5 py-1 text-xs font-mono border uppercase tracking-tighter transition-all cursor-pointer ${
                  brushSize === size
                    ? "bg-[#00FF00] text-black font-bold border-[#00FF00]"
                    : "bg-black text-[#00FF00] border-[#00FF00]/30 hover:border-[#00FF00]"
                }`}
              >
                {size === 12 ? "Thin" : size === 18 ? "Med" : "Thick"}
              </button>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-3 w-full mt-2">
          <button
            onClick={handleClear}
            className="flex items-center justify-center gap-2 py-2.5 px-4 border-2 border-[#00FF00]/50 text-[#00FF00] bg-black hover:bg-[#00FF00] hover:text-black transition-colors font-mono font-bold text-xs uppercase cursor-pointer"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
          <button
            onClick={handleUndo}
            className="flex items-center justify-center gap-2 py-2.5 px-4 bg-[#00FF00] text-black hover:bg-black hover:text-[#00FF00] hover:border-[#00FF00] border-2 border-transparent transition-all font-mono font-bold text-xs uppercase cursor-pointer"
          >
            <Undo className="h-3.5 w-3.5" />
            Undo
          </button>
        </div>
      </div>

      {/* Downscaled Neural Input View */}
      <div className="flex flex-col items-center gap-4 bg-[#111111] border border-white/20 p-6 w-full max-w-sm">
        <div className="flex justify-between items-center w-full border-b border-white/10 pb-2">
          <div className="flex items-center gap-2">
            <CircleHelp className="h-4 w-4 text-[#00FF00]" />
            <h3 className="font-mono font-bold text-sm uppercase tracking-wider text-white">Neural Input</h3>
          </div>
          <div className="text-[10px] font-mono text-[#00FF00] bg-black border border-white/10 px-2 py-0.5">
            28 x 28 TENSOR
          </div>
        </div>

        {/* 28x28 Downscaled Visualizer Grid */}
        <div className="relative p-1 bg-black border border-[#00FF00]/20">
          <div 
            style={{ gridTemplateColumns: "repeat(28, minmax(0, 1fr))" }}
            className="grid gap-[1px] w-[224px] h-[224px] bg-[#111111]"
            onMouseLeave={handleGridLeave}
          >
            {Array.from({ length: 784 }).map((_, index) => {
              const val = previewPixels[index] || 0;
              // Compute background color based on pixel value for brutalist green glow
              const bgStyle = {
                backgroundColor: `rgba(0, 255, 0, ${val})`,
                boxShadow: val > 0.5 ? `0 0 4px rgba(0, 255, 0, ${val * 0.5})` : "none"
              };

              return (
                <div
                  key={index}
                  style={bgStyle}
                  className={`w-2 h-2 rounded-[1px] border-[0.5px] border-black/50 transition-colors cursor-pointer ${
                    val > 0.1 ? "" : "bg-black"
                  }`}
                  onMouseEnter={(e) => handleGridHover(e, index)}
                />
              );
            })}
          </div>
        </div>

        {/* Pixel Detail / Debugger Bar */}
        <div className="w-full bg-black border border-white/10 p-3 min-h-[56px] flex items-center justify-center text-center">
          {hoveredPixel ? (
            <div className="flex items-center justify-between w-full text-[10px] font-mono uppercase">
              <span className="text-white/50">Pixel index:</span>
              <span className="text-[#00FF00] font-bold">({hoveredPixel.x},{hoveredPixel.y})</span>
              <span className="text-white/50">Val:</span>
              <span className="text-[#00FF00] font-bold">{(hoveredPixel.val).toFixed(3)}</span>
            </div>
          ) : (
            <p className="text-[10px] text-white/40 font-mono uppercase tracking-tight">
              Hover matrix cells to read input vector features.
            </p>
          )}
        </div>
      </div>

      {/* Hidden 28x28 canvas for image processing */}
      <canvas
        ref={hiddenCanvasRef}
        width={28}
        height={28}
        className="hidden"
      />
    </div>
  );
}

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, RotateCcw, Save, SlidersHorizontal, ZoomIn, ZoomOut, Maximize } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const POINT_LABELS = [
  "1 - Left FT box × baseline",
  "2 - Right FT box × baseline",
  "3 - Left FT line end",
  "4 - Right FT line end",
  "5 - Left baseline corner",
  "6 - Right baseline corner",
  "7 - Left 3pt × baseline",
  "8 - Right 3pt × baseline",
  "9 - Top of 3pt arc",
  "10 - Left near sideline",
  "11 - Right near sideline",
];

export default function CalibrationModal({ file, onClose, onSaved }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const pointsRef = useRef([]); // mirror of points state for use inside event handlers
  const statusRef = useRef('uploading');

  const [points, setPoints] = useState([]);
  const [status, setStatus] = useState('uploading');
  const [errorMsg, setErrorMsg] = useState('');
  const [fileId, setFileId] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Zoom / pan state — stored in refs so event handlers always see latest value
  const zoom = useRef(1);
  const panX = useRef(0);
  const panY = useRef(0);
  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Keep refs in sync
  const syncPoints = (p) => { pointsRef.current = p; setPoints(p); };
  const syncStatus = (s) => { statusRef.current = s; setStatus(s); };

  // ── Draw everything onto the canvas ──────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(panX.current, panY.current);
    ctx.scale(zoom.current, zoom.current);

    // Image
    ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);

    // Points
    pointsRef.current.forEach(([x, y], i) => {
      const r = 6 / zoom.current;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5 / zoom.current;
      ctx.stroke();

      ctx.fillStyle = 'white';
      ctx.font = `bold ${14 / zoom.current}px sans-serif`;
      ctx.fillText(i + 1, x + (9 / zoom.current), y + (5 / zoom.current));
    });

    ctx.restore();
  }, []);

  // ── Fit canvas display size to container ─────────────────────────────────
  const fitCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imageRef.current;
    if (!canvas || !container || !img) return;

    // Canvas internal resolution = image resolution
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // CSS display size = container size (canvas scales via CSS, but we handle
    // coordinate mapping ourselves so clicks stay accurate)
    canvas.style.width = '100%';
    canvas.style.height = '100%';

    // Reset zoom to fit
    resetZoom();
  }, []);

  const resetZoom = useCallback(() => {
    zoom.current = 1;
    panX.current = 0;
    panY.current = 0;
    draw();
  }, [draw]);

  // ── Convert mouse event → image coordinates ───────────────────────────────
  const toImageCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    // Position inside the CSS-scaled canvas element
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;
    // Canvas internal pixel ratio
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    // Remove pan and zoom to get image coords
    const imgX = (cssX * scaleX - panX.current) / zoom.current;
    const imgY = (cssY * scaleY - panY.current) / zoom.current;
    return [Math.round(imgX), Math.round(imgY)];
  };

  // ── Mouse handlers ────────────────────────────────────────────────────────
  const handleMouseDown = (e) => {
    if (e.button === 1 || e.button === 2) { // middle or right click → pan
      isPanning.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
      e.preventDefault();
    }
  };

  const handleMouseMove = (e) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    panX.current += dx * scaleX;
    panY.current += dy * scaleY;
    draw();
  };

  const handleMouseUp = () => { isPanning.current = false; };

  const handleClick = (e) => {
    if (statusRef.current !== 'ready') return;
    if (pointsRef.current.length >= 11) return;
    if (isPanning.current) return;

    const [x, y] = toImageCoords(e);
    const newPoints = [...pointsRef.current, [x, y]];
    syncPoints(newPoints);
    draw();
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.5, Math.min(10, zoom.current * factor));

    // Zoom toward mouse position
    panX.current = mouseX - (mouseX - panX.current) * (newZoom / zoom.current);
    panY.current = mouseY - (mouseY - panY.current) * (newZoom / zoom.current);
    zoom.current = newZoom;
    draw();
  };

  const handleZoomBtn = (direction) => {
    const canvas = canvasRef.current;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const factor = direction === 'in' ? 1.3 : 1 / 1.3;
    const newZoom = Math.max(0.5, Math.min(10, zoom.current * factor));
    panX.current = cx - (cx - panX.current) * (newZoom / zoom.current);
    panY.current = cy - (cy - panY.current) * (newZoom / zoom.current);
    zoom.current = newZoom;
    draw();
  };

  // ── Upload + frame load ───────────────────────────────────────────────────
  useEffect(() => {
    const uploadAndLoad = async () => {
      try {
        syncStatus('uploading');
        const formData = new FormData();
        formData.append('file', file);
        const uploadRes = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
        if (!uploadRes.ok) throw new Error("Upload failed.");
        const { file_id } = await uploadRes.json();
        setFileId(file_id);

        const img = new Image();
        img.onload = () => {
          imageRef.current = img;
          fitCanvas();
          syncStatus('ready');
        };
        img.onerror = () => { throw new Error("Failed to load frame."); };
        img.src = `${API_URL}/calibration/frame/${file_id}`;
      } catch (err) {
        setErrorMsg(err.message);
        syncStatus('error');
      }
    };
    uploadAndLoad();
  }, [fitCanvas]);

  // Attach wheel listener as non-passive so preventDefault works
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, []);

  // ── Actions ───────────────────────────────────────────────────────────────
  const handleUndo = () => {
    const newPoints = pointsRef.current.slice(0, -1);
    syncPoints(newPoints);
    draw();
  };

  const handleReset = () => {
    syncPoints([]);
    draw();
  };

  const handleSubmit = async () => {
    if (pointsRef.current.length < 11 || !fileId) return;
    try {
      syncStatus('saving');
      const res = await fetch(`${API_URL}/calibration/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ required_pts: pointsRef.current }),
      });
      if (!res.ok) throw new Error("Calibration failed on server.");
      syncStatus('done');
      if (onSaved) onSaved();

      // Load reprojection preview and draw it onto the canvas
      const previewImg = new Image();
      previewImg.onload = () => {
        imageRef.current = previewImg;
        const canvas = canvasRef.current;
        canvas.width = previewImg.naturalWidth;
        canvas.height = previewImg.naturalHeight;
        syncPoints([]);
        zoom.current = 1;
        panX.current = 0;
        panY.current = 0;
        draw();
      };
      previewImg.src = `${API_URL}/calibration/preview?t=${Date.now()}`;

    } catch (err) {
      setErrorMsg(err.message);
      syncStatus('error');
    }
  };

  const nextLabel = points.length < 11 ? POINT_LABELS[points.length] : null;

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-2">
      <div className="bg-slate-800 rounded-2xl p-6 w-full flex flex-col gap-4" style={{ height: '100vh', maxWidth: '100vw' }}>

        {/* Header */}
        <div className="flex justify-between items-center flex-shrink-0">
          <h2 className="text-white font-bold text-xl flex items-center gap-2">
            <SlidersHorizontal size={20} /> Court Calibration
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X size={24} />
          </button>
        </div>

        {/* Status bar */}
        <div className="flex-shrink-0 bg-slate-900/50 rounded-xl px-4 py-2 flex items-center justify-between">
          <span className="text-sm text-slate-400">
            {status === 'uploading' && <span className="animate-pulse">Uploading video and extracting frame...</span>}
            {status === 'ready' && nextLabel && <>Next: <strong className="text-orange-400 ml-1">{nextLabel}</strong></>}
            {status === 'ready' && !nextLabel && <span className="text-green-400 font-bold">All 11 points placed — ready to save!</span>}
            {status === 'saving' && "Saving calibration..."}
            {status === 'done' && <span className="text-green-400 font-bold">✓ Calibration saved!</span>}
            {status === 'error' && <span className="text-red-400">{errorMsg}</span>}
          </span>
          <span className="text-sm font-bold text-white ml-4">{points.length} / 11</span>
        </div>

        {/* Canvas container */}
        <div
          ref={containerRef}
          className="flex-1 overflow-hidden rounded-xl border border-slate-600 bg-slate-900 flex items-center justify-center"
          onContextMenu={(e) => e.preventDefault()}
        >
          {status === 'uploading' && (
            <div className="text-slate-400 animate-pulse">Uploading & extracting frame...</div>
          )}
          <canvas
            ref={canvasRef}
            onClick={handleClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{
              cursor: status === 'ready' && points.length < 11 ? 'crosshair' : 'grab',
              display: status === 'uploading' ? 'none' : 'block',
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />
        </div>

        {/* Reprojection Preview */}
        {previewUrl && (
          <div className="flex-shrink-0 rounded-xl border border-slate-600 overflow-hidden bg-slate-900 flex flex-col items-center gap-2 p-3">
            <span className="text-xs text-slate-400 uppercase tracking-wider font-bold">Reprojection Preview</span>
            <img
              src={previewUrl}
              alt="Reprojection preview"
              className="max-h-48 rounded-lg object-contain"
            />
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-between flex-shrink-0">
          <div className="flex gap-3 items-center">
            {/* Zoom buttons */}
            <button onClick={() => handleZoomBtn('in')} className="text-slate-400 hover:text-white transition-colors" title="Zoom in">
              <ZoomIn size={18} />
            </button>
            <button onClick={() => handleZoomBtn('out')} className="text-slate-400 hover:text-white transition-colors" title="Zoom out">
              <ZoomOut size={18} />
            </button>
            <button onClick={resetZoom} className="text-slate-400 hover:text-white transition-colors" title="Reset zoom">
              <Maximize size={18} />
            </button>
            <span className="text-slate-600">|</span>
            <button
              onClick={handleUndo}
              disabled={points.length === 0 || status !== 'ready'}
              className="flex items-center gap-1 text-slate-400 hover:text-white text-sm disabled:opacity-40 transition-colors"
            >
              <RotateCcw size={14} /> Undo
            </button>
            <button
              onClick={handleReset}
              disabled={points.length === 0 || status !== 'ready'}
              className="flex items-center gap-1 text-slate-400 hover:text-white text-sm disabled:opacity-40 transition-colors"
            >
              <X size={14} /> Reset
            </button>
            <span className="text-slate-500 text-xs ml-2">Scroll to zoom · Right-click drag to pan</span>
          </div>

          <div>
            {status === 'done' ? (
              <button
                onClick={onClose}
                className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-xl font-bold transition-all"
              >
                ✓ Close
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={points.length < 11 || status !== 'ready'}
                className="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-2 rounded-xl font-bold transition-all"
              >
                <Save size={16} />
                {status === 'saving' ? 'Saving...' : 'Save Calibration'}
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
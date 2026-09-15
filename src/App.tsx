import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, History, Check, Zap, ZapOff, RotateCcw, Image as ImageIcon, Scan, ChevronLeft, Send, Loader2, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Scan as ScanRecord, Point } from './types';

// Order 4 points as [top-left, top-right, bottom-right, bottom-left]
function orderPoints(pts: Point[]): [Point, Point, Point, Point] {
  const sums  = pts.map(p => p.x + p.y);
  const diffs = pts.map(p => p.x - p.y);
  const tl = pts[sums.indexOf(Math.min(...sums))];
  const br = pts[sums.indexOf(Math.max(...sums))];
  const tr = pts[diffs.indexOf(Math.max(...diffs))];
  const bl = pts[diffs.indexOf(Math.min(...diffs))];
  return [tl, tr, br, bl];
}

export default function App() {
  const [view, setView] = useState<'scanner' | 'history' | 'detail'>('scanner');
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [selectedScan, setSelectedScan] = useState<ScanRecord | null>(null);
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [isAutoCapture, setIsAutoCapture] = useState(true);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cvReady, setCvReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Off-screen canvas used for OpenCV processing (scaled down for perf)
  const procCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // Corners detected by OpenCV at time of capture (overlay canvas coordinates)
  const capturedCornersRef = useRef<Point[] | null>(null);

  // Wait for OpenCV.js WASM to finish initializing
  useEffect(() => {
    if (!procCanvasRef.current) {
      procCanvasRef.current = document.createElement('canvas');
    }
    const check = () => {
      const cv = (window as any).cv;
      if (cv && typeof cv.Mat === 'function') {
        setCvReady(true);
      } else {
        setTimeout(check, 250);
      }
    };
    check();
  }, []);

  // Fetch scan history
  useEffect(() => {
    const fetchScans = async () => {
      try {
        const response = await fetch('/api/scans');
        if (response.ok) {
          setScans(await response.json());
        }
      } catch (error) {
        console.error('Fetch scans error:', error);
      }
    };
    fetchScans();
  }, [view]);

  // Initialize Camera
  const startCamera = useCallback(async () => {
    if (streamRef.current) {
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
      }
      return;
    }
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setCameraError('Camera access denied or not available. Please check your permissions.');
    }
  }, []);

  useEffect(() => {
    if (view === 'scanner' && !capturedImage) {
      startCamera();
    }
  }, [view, capturedImage, startCamera]);

  // Only release the camera when actually leaving the scanner
  useEffect(() => {
    if (view !== 'scanner' && streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  }, [view]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Real OpenCV edge detection & auto-capture
  useEffect(() => {
    if (view !== 'scanner' || capturedImage || !isAutoCapture || !cvReady) return;

    const cv = (window as any).cv;
    const procCanvas = procCanvasRef.current;
    if (!cv || !procCanvas) return;

    let animationFrameId: number;
    let stableFrameCount = 0;
    let lastCorners: Point[] | null = null;
    let captureCalled = false;

    const STABLE_FRAMES = 40;
    const STABILITY_PX = 40; // max per-corner drift to count as stable

    const drawGuide = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const m = w * 0.08;
      const gw = w - m * 2;
      const gh = gw * 1.4;
      const gy = (h - gh) / 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 2;
      ctx.setLineDash([20, 10]);
      ctx.strokeRect(m, gy, gw, gh);
      ctx.setLineDash([]);
    };

    const processFrame = () => {
      const video = videoRef.current;
      const overlay = overlayCanvasRef.current;

      if (!video || !overlay || video.readyState < 2 || video.videoWidth === 0) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      // Keep overlay matched to rendered video size
      if (overlay.width !== video.clientWidth || overlay.height !== video.clientHeight) {
        overlay.width = video.clientWidth;
        overlay.height = video.clientHeight;
      }

      // Scale video down for fast processing (~640px wide)
      const PROC_W = 640;
      const scale = PROC_W / video.videoWidth;
      procCanvas.width = PROC_W;
      procCanvas.height = Math.round(video.videoHeight * scale);

      const pCtx = procCanvas.getContext('2d', { willReadFrequently: true });
      if (!pCtx) { animationFrameId = requestAnimationFrame(processFrame); return; }
      pCtx.drawImage(video, 0, 0, procCanvas.width, procCanvas.height);

      const ctx = overlay.getContext('2d');
      if (!ctx) { animationFrameId = requestAnimationFrame(processFrame); return; }

      let src: any, gray: any, blurred: any, edges: any,
          contours: any, hierarchy: any, kernel: any;
      try {
        src       = cv.imread(procCanvas);
        gray      = new cv.Mat();
        blurred   = new cv.Mat();
        edges     = new cv.Mat();
        contours  = new cv.MatVector();
        hierarchy = new cv.Mat();

        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
        cv.Canny(blurred, edges, 50, 150);

        // Dilate slightly to close edge gaps
        kernel = cv.Mat.ones(3, 3, cv.CV_8U);
        cv.dilate(edges, edges, kernel);

        cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

        const minArea = procCanvas.width * procCanvas.height * 0.15;
        const scaleX  = overlay.width  / procCanvas.width;
        const scaleY  = overlay.height / procCanvas.height;

        let bestCorners: Point[] | null = null;
        let bestArea = 0;

        for (let i = 0; i < contours.size(); i++) {
          const cnt  = contours.get(i);
          const area = cv.contourArea(cnt);

          if (area > minArea && area > bestArea) {
            const peri = cv.arcLength(cnt, true);
            // Try progressively looser epsilon until we get a quad
            for (const eps of [0.02, 0.03, 0.05]) {
              const approx = new cv.Mat();
              cv.approxPolyDP(cnt, approx, eps * peri, true);
              if (approx.rows === 4) {
                const corners: Point[] = [];
                for (let j = 0; j < 4; j++) {
                  corners.push({
                    x: approx.data32S[j * 2]     * scaleX,
                    y: approx.data32S[j * 2 + 1] * scaleY,
                  });
                }
                bestCorners = corners;
                bestArea    = area;
                approx.delete();
                break;
              }
              approx.delete();
            }
          }
          cnt.delete();
        }

        ctx.clearRect(0, 0, overlay.width, overlay.height);

        if (bestCorners) {
          const isStable = lastCorners !== null && lastCorners.every((pt, i) => {
            const dx = pt.x - bestCorners![i].x;
            const dy = pt.y - bestCorners![i].y;
            return Math.sqrt(dx * dx + dy * dy) < STABILITY_PX;
          });

          stableFrameCount = isStable ? stableFrameCount + 1 : 0;
          lastCorners      = bestCorners;

          const stable = stableFrameCount >= STABLE_FRAMES;
          setIsFocused(stable);

          ctx.strokeStyle = stable ? '#34c759' : '#0a84ff';
          ctx.lineWidth   = 3;
          ctx.setLineDash(stable ? [] : [10, 5]);
          ctx.beginPath();
          ctx.moveTo(bestCorners[0].x, bestCorners[0].y);
          for (let i = 1; i < 4; i++) ctx.lineTo(bestCorners[i].x, bestCorners[i].y);
          ctx.closePath();
          ctx.stroke();
          ctx.fillStyle = stable ? 'rgba(52,199,89,0.15)' : 'rgba(10,132,255,0.15)';
          ctx.fill();
          ctx.setLineDash([]);

          if (stable && !captureCalled) {
            capturedCornersRef.current = bestCorners;
            captureCalled = true;
            handleCapture();
          }
        } else {
          stableFrameCount = 0;
          lastCorners      = null;
          setIsFocused(false);
          drawGuide(ctx, overlay.width, overlay.height);
        }
      } catch (err) {
        console.error('OpenCV frame error:', err);
      } finally {
        src?.delete();
        gray?.delete();
        blurred?.delete();
        edges?.delete();
        contours?.delete();
        hierarchy?.delete();
        kernel?.delete();
      }

      animationFrameId = requestAnimationFrame(processFrame);
    };

    animationFrameId = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(animationFrameId);
  }, [view, capturedImage, isAutoCapture, cvReady]);

  const handleCapture = () => {
    if (isCapturing || capturedImage || isProcessing) return;

    setIsCapturing(true);

    const flash = document.createElement('div');
    flash.className = 'fixed inset-0 bg-white z-[100] opacity-0 transition-opacity duration-100';
    document.body.appendChild(flash);
    setTimeout(() => flash.style.opacity = '1', 10);
    setTimeout(() => {
      flash.style.opacity = '0';
      setTimeout(() => document.body.removeChild(flash), 100);
    }, 150);

    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);

        setIsProcessing(true);

        setTimeout(() => {
          let finalDataUrl = canvas.toDataURL('image/jpeg', 0.9);

          // Apply perspective crop if we have corner points
          const cv = (window as any).cv;
          const corners = capturedCornersRef.current;
          const overlay = overlayCanvasRef.current;

          if (cv && corners && overlay && overlay.width > 0 && overlay.height > 0) {
            try {
              const PADDING = 15; // px in full-res video coords

              // Map corners from overlay display coords → full-res video coords
              const sx = video.videoWidth  / overlay.width;
              const sy = video.videoHeight / overlay.height;
              const videoPts = corners.map(p => ({ x: p.x * sx, y: p.y * sy }));

              const [tl, tr, br, bl] = orderPoints(videoPts);

              // Expand each corner outward by PADDING pixels
              const cx = (tl.x + tr.x + br.x + bl.x) / 4;
              const cy = (tl.y + tr.y + br.y + bl.y) / 4;
              const pad = (pt: Point): Point => {
                const dx = pt.x - cx;
                const dy = pt.y - cy;
                const len = Math.hypot(dx, dy) || 1;
                return {
                  x: Math.max(0, Math.min(video.videoWidth  - 1, pt.x + (dx / len) * PADDING)),
                  y: Math.max(0, Math.min(video.videoHeight - 1, pt.y + (dy / len) * PADDING)),
                };
              };
              const [ptl, ptr, pbr, pbl] = [pad(tl), pad(tr), pad(br), pad(bl)];

              const maxW = Math.round(Math.max(
                Math.hypot(ptr.x - ptl.x, ptr.y - ptl.y),
                Math.hypot(pbr.x - pbl.x, pbr.y - pbl.y),
              ));
              const maxH = Math.round(Math.max(
                Math.hypot(pbl.x - ptl.x, pbl.y - ptl.y),
                Math.hypot(pbr.x - ptr.x, pbr.y - ptr.y),
              ));

              if (maxW > 50 && maxH > 50) {
                const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
                  ptl.x, ptl.y,
                  ptr.x, ptr.y,
                  pbr.x, pbr.y,
                  pbl.x, pbl.y,
                ]);
                const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
                  0,        0,
                  maxW - 1, 0,
                  maxW - 1, maxH - 1,
                  0,        maxH - 1,
                ]);

                const M   = cv.getPerspectiveTransform(srcPts, dstPts);
                const src = cv.imread(canvas);
                const dst = new cv.Mat();
                cv.warpPerspective(src, dst, M, new cv.Size(maxW, maxH));

                const warpCanvas = document.createElement('canvas');
                cv.imshow(warpCanvas, dst);
                finalDataUrl = warpCanvas.toDataURL('image/jpeg', 0.9);

                srcPts.delete(); dstPts.delete(); M.delete(); src.delete(); dst.delete();
              }
            } catch (err) {
              console.error('Perspective warp error:', err);
            }
          }

          capturedCornersRef.current = null;
          setCapturedImage(finalDataUrl);
          setIsProcessing(false);
          setIsCapturing(false);

          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
          }
        }, 1200);
      }
    }
  };

  const saveAndSendScan = async () => {
    if (!capturedImage) return;

    setIsSending(true);
    try {
      const saveResponse = await fetch('/api/scans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: capturedImage }),
      });

      if (!saveResponse.ok) {
        alert('Failed to save scan.');
        return;
      }

      const saved: ScanRecord = await saveResponse.json();

      const sendResponse = await fetch(`/api/scans/${saved.id}/send`, { method: 'POST' });
      if (!sendResponse.ok) {
        const body = await sendResponse.json().catch(() => ({}));
        alert(`Scan saved, but sending it failed: ${body.error || sendResponse.status}`);
      }

      setCapturedImage(null);
      setView('history');
    } catch (error) {
      console.error('saveAndSendScan error:', error);
      alert(`Error: ${error}`);
    } finally {
      setIsSending(false);
    }
  };

  const deleteScan = async (id: string) => {
    try {
      const response = await fetch(`/api/scans/${id}`, { method: 'DELETE' });
      if (response.ok) {
        setScans(prev => prev.filter(s => s.id !== id));
        if (selectedScan?.id === id) {
          setSelectedScan(null);
          setView('history');
        }
      }
    } catch (error) {
      console.error('Error deleting scan:', error);
    }
  };

  const resendScan = async (scan: ScanRecord) => {
    setIsSending(true);
    try {
      const response = await fetch(`/api/scans/${scan.id}/send`, { method: 'POST' });
      if (response.ok) {
        alert('Scan sent successfully!');
        setScans(prev => prev.map(s => s.id === scan.id ? { ...s, sent: true } : s));
      } else {
        const body = await response.json().catch(() => ({}));
        alert(`Failed to send scan: ${body.error || response.status}`);
      }
    } catch (error) {
      console.error('resendScan error:', error);
      alert('Failed to send scan.');
    } finally {
      setIsSending(false);
    }
  };

  const downloadImage = (dataUrl: string) => {
    // On iOS Safari, <a download> does not save to camera roll.
    // Opening in a new tab lets the user long-press to save the image.
    window.open(dataUrl, '_blank');
  };

  return (
    <div className="flex flex-col bg-black text-white overflow-hidden safe-area-inset" style={{ position: 'fixed', inset: 0, touchAction: 'none' }}>

      {/* Main Viewport */}
      <div className="relative flex-1 overflow-hidden">

        {/* Scanner View */}
        {view === 'scanner' && (
          <div className="absolute inset-0 flex flex-col min-h-0 overflow-hidden" style={{ touchAction: 'none' }}>
            {!capturedImage ? (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none z-10"
                />

                {cameraError && (
                  <div className="absolute inset-0 z-30 flex flex-col items-center justify-center p-8 text-center bg-black/80">
                    <div className="p-4 rounded-full bg-red-500/20 text-red-500 mb-4">
                      <Camera size={48} />
                    </div>
                    <h3 className="text-xl font-bold mb-2">Camera Error</h3>
                    <p className="text-ios-gray mb-6">{cameraError}</p>
                    <button
                      onClick={startCamera}
                      className="px-6 py-3 bg-blue-500 rounded-xl font-bold ios-btn-active"
                    >
                      Try Again
                    </button>
                  </div>
                )}

                {/* Scanner UI Overlays */}
                <div className="absolute top-0 left-0 right-0 p-6 pt-16 flex justify-between items-center z-30 bg-gradient-to-b from-black/80 to-transparent">
                  <button
                    onClick={() => setView('history')}
                    className="p-4 rounded-full bg-white/20 backdrop-blur-xl border border-white/20 ios-btn-active shadow-2xl text-white"
                  >
                    <History size={28} />
                  </button>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => setIsFlashOn(!isFlashOn)}
                      className={`p-4 rounded-full backdrop-blur-xl border border-white/20 ios-btn-active shadow-2xl ${isFlashOn ? 'bg-yellow-400 text-black' : 'bg-white/20 text-white'}`}
                    >
                      {isFlashOn ? <Zap size={28} /> : <ZapOff size={28} />}
                    </button>
                    <button
                      onClick={() => setIsAutoCapture(!isAutoCapture)}
                      className={`px-6 py-3 rounded-full text-sm font-bold backdrop-blur-xl border border-white/20 ios-btn-active shadow-2xl ${isAutoCapture ? 'bg-blue-500 text-white' : 'bg-white/20 text-white'}`}
                    >
                      {isAutoCapture ? 'AUTO' : 'MANUAL'}
                    </button>
                  </div>
                </div>

                {!isFocused && cvReady && (
                  <div className="absolute bottom-32 left-0 right-0 flex justify-center pointer-events-none z-20">
                    <span className="text-[11px] font-bold tracking-widest uppercase text-white/50 bg-black/40 px-4 py-2 rounded-full">
                      Point at a document
                    </span>
                  </div>
                )}

                <div className="absolute bottom-0 left-0 right-0 p-12 pb-24 flex justify-center items-center z-30 bg-gradient-to-t from-black/80 to-transparent">
                  <button
                    onClick={handleCapture}
                    disabled={isCapturing || isProcessing || !!cameraError}
                    className="w-24 h-24 rounded-full border-4 border-white p-2 ios-btn-active relative shadow-[0_0_30px_rgba(255,255,255,0.3)] disabled:opacity-50"
                  >
                    <div className={`w-full h-full rounded-full transition-all duration-300 ${isFocused ? 'bg-green-500 scale-90 shadow-[0_0_25px_rgba(34,197,94,0.6)]' : 'bg-white'}`} />
                    {isProcessing && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                      </div>
                    )}
                  </button>
                </div>

                {isProcessing && (
                  <div className="absolute inset-0 z-50 flex flex-col items-center justify-center glass">
                    <div className="w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-6" />
                    <h2 className="text-xl font-semibold">Processing...</h2>
                    <p className="text-ios-gray mt-2">Cropping and straightening</p>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col w-full h-full bg-black">
                <div className="flex-1 min-h-0 p-4 flex flex-col">
                  <div className="text-center mb-4 mt-4">
                    <h2 className="text-xl font-bold">Is it readable?</h2>
                    <p className="text-sm text-ios-gray mt-1 px-8">Make sure the page is clean and all text is clearly readable before sending.</p>
                  </div>
                  <div className="flex-1 min-h-0">
                    <img src={capturedImage} className="w-full h-full object-contain rounded-2xl shadow-2xl" alt="Captured" />
                  </div>
                </div>
                <div className="flex-shrink-0 px-10 py-6 glass flex justify-around items-center">
                  <button
                    onClick={() => setCapturedImage(null)}
                    className="flex flex-col items-center gap-3 text-white ios-btn-active"
                  >
                    <div className="p-5 rounded-full bg-white/20 backdrop-blur-xl border border-white/10 shadow-xl">
                      <RotateCcw size={32} />
                    </div>
                    <span className="text-sm font-bold">Retake</span>
                  </button>
                  <button
                    onClick={() => downloadImage(capturedImage)}
                    className="flex flex-col items-center gap-3 text-white ios-btn-active"
                  >
                    <div className="p-5 rounded-full bg-white/20 backdrop-blur-xl border border-white/10 shadow-xl">
                      <ImageIcon size={32} />
                    </div>
                    <span className="text-sm font-bold">Save Local</span>
                  </button>
                  <button
                    onClick={saveAndSendScan}
                    disabled={isSending}
                    className="flex flex-col items-center gap-3 text-white ios-btn-active disabled:opacity-50"
                  >
                    <div className="p-5 rounded-full bg-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.5)]">
                      {isSending ? <Loader2 className="animate-spin" size={32} /> : <Check size={32} />}
                    </div>
                    <span className="text-sm font-bold">{isSending ? 'Sending...' : 'Save & Send'}</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* History View */}
        {view === 'history' && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            className="absolute inset-0 bg-black flex flex-col"
          >
            <div className="p-6 flex justify-between items-center glass sticky top-0 z-20">
              <h1 className="text-2xl font-bold">Scans</h1>
              <button
                onClick={() => setView('scanner')}
                className="p-2 rounded-full bg-blue-500 text-white ios-btn-active"
              >
                <Camera size={24} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 grid grid-cols-2 gap-4 pb-20">
              {scans.length === 0 ? (
                <div className="col-span-2 flex flex-col items-center justify-center h-64 text-ios-gray">
                  <Scan size={48} strokeWidth={1} className="mb-4 opacity-20" />
                  <p>No scans yet</p>
                </div>
              ) : (
                scans.map(scan => (
                  <motion.div
                    layoutId={scan.id}
                    key={scan.id}
                    onClick={() => {
                      setSelectedScan(scan);
                      setView('detail');
                    }}
                    className="aspect-[3/4] bg-ios-surface rounded-xl overflow-hidden relative ios-btn-active"
                  >
                    <img src={scan.image} className="w-full h-full object-cover" alt="Scan" />
                    <div className="absolute top-2 right-2">
                      <span className={`text-[10px] font-bold px-2 py-1 rounded-full ${scan.sent ? 'bg-green-500/80' : 'bg-red-500/80'}`}>
                        {scan.sent ? 'Sent' : 'Not sent'}
                      </span>
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
                      <p className="text-[10px] font-medium truncate opacity-80">{new Date(scan.timestamp).toLocaleString()}</p>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        )}

        {/* Detail View */}
        {view === 'detail' && selectedScan && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute inset-0 bg-black z-50 flex flex-col"
          >
            <div className="p-6 flex justify-between items-center glass">
              <button
                onClick={() => setView('history')}
                className="flex items-center gap-1 text-blue-500 font-medium ios-btn-active"
              >
                <ChevronLeft size={24} />
                <span>Back</span>
              </button>
              <button
                onClick={() => deleteScan(selectedScan.id)}
                className="flex items-center gap-1 text-red-500 font-medium ios-btn-active"
              >
                <Trash2 size={18} />
                Delete
              </button>
            </div>
            <div className="flex-1 p-4 flex items-center justify-center">
              <img
                src={selectedScan.image}
                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                alt="Scan"
              />
            </div>
            <div className="p-8 glass flex flex-col items-center">
              <p className="text-ios-gray text-sm">{new Date(selectedScan.timestamp).toLocaleString()}</p>
              <p className={`text-xs font-bold mt-2 ${selectedScan.sent ? 'text-green-500' : 'text-red-500'}`}>
                {selectedScan.sent ? 'Sent to webhook' : 'Not yet sent'}
              </p>

              <div className="mt-8 flex flex-col gap-4 w-full">
                <div className="flex gap-4">
                  <button
                    onClick={() => downloadImage(selectedScan.image)}
                    className="flex-1 py-4 rounded-2xl bg-white/10 font-semibold ios-btn-active flex items-center justify-center gap-2"
                  >
                    <ImageIcon size={20} />
                    Save to Photos
                  </button>
                  <button
                    onClick={() => resendScan(selectedScan)}
                    disabled={isSending}
                    className="flex-1 py-4 rounded-2xl bg-blue-500 font-semibold ios-btn-active flex items-center justify-center gap-2"
                  >
                    {isSending ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
                    {isSending ? 'Sending...' : 'Resend'}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

      </div>

      {/* Hidden Canvas for Processing */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Tab Bar (only in history) */}
      {view === 'history' && (
        <div className="h-20 glass border-t border-white/5 flex justify-around items-center px-6 pb-4">
          <button className="flex flex-col items-center gap-1 text-blue-500">
            <History size={24} />
            <span className="text-[10px] font-medium">Recent</span>
          </button>
          <button
            onClick={() => setView('scanner')}
            className="flex flex-col items-center gap-1 text-ios-gray"
          >
            <Camera size={24} />
            <span className="text-[10px] font-medium">Scan</span>
          </button>
        </div>
      )}
    </div>
  );
}

import { useRef, useEffect, useState, useCallback } from 'react';

// ─── Catmull-Rom Spline ────────────────────────────────────────────────────
// Evaluates a smooth curve through control points (no wavy spikes).
// Each segment uses the two surrounding points as tangent guides.
function catmullRomPoint(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return [
    0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
    0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
  ];
}

// Evaluate N dense points along a Catmull-Rom spline through control points
function evalSpline(cps, nSamples = 60) {
  if (cps.length < 2) return cps;
  if (cps.length === 2) {
    const pts = [];
    for (let i = 0; i <= nSamples; i++) {
      const t = i / nSamples;
      pts.push([cps[0][0] + t * (cps[1][0] - cps[0][0]), cps[0][1] + t * (cps[1][1] - cps[0][1])]);
    }
    return pts;
  }
  const pts = [];
  const n = cps.length;
  const samplesPerSeg = Math.max(4, Math.round(nSamples / (n - 1)));
  for (let i = 0; i < n - 1; i++) {
    const p0 = cps[Math.max(0, i - 1)];
    const p1 = cps[i];
    const p2 = cps[i + 1];
    const p3 = cps[Math.min(n - 1, i + 2)];
    for (let s = 0; s < samplesPerSeg; s++) {
      pts.push(catmullRomPoint(p0, p1, p2, p3, s / samplesPerSeg));
    }
  }
  pts.push(cps[n - 1]);
  return pts;
}

// ─── Grid from two spline curves ─────────────────────────────────────────
// Generate col_lines by interpolating between top and bottom splines
function buildGridFromCurves(topCPs, botCPs, nCols = 8, nRows = 20) {
  const topDense = evalSpline(topCPs, nCols);
  const botDense = evalSpline(botCPs, nCols);
  // Sample nCols+1 points evenly
  const step = (topDense.length - 1) / nCols;
  const colLines = [];
  for (let ci = 0; ci <= nCols; ci++) {
    const idx = Math.round(ci * step);
    const tp = topDense[Math.min(idx, topDense.length - 1)];
    const bp = botDense[Math.min(idx, botDense.length - 1)];
    const col = [];
    for (let ri = 0; ri <= nRows; ri++) {
      const t = ri / nRows;
      col.push([tp[0] + t * (bp[0] - tp[0]), tp[1] + t * (bp[1] - tp[1])]);
    }
    colLines.push(col);
  }
  return colLines;
}

// ─── Component ────────────────────────────────────────────────────────────
export default function CanvasEditor({
  imageSrc, imageWidth, imageHeight,
  corners, onCornersChange,
  dewarpGrid = null, onDewarpGridChange = null,
  contentRect = null, onContentRectChange = null,
}) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });

  // Dragging state
  const [draggingCP, setDraggingCP] = useState(null);
  const [draggingCorner, setDraggingCorner] = useState(null);
  const [draggingContentHandle, setDraggingContentHandle] = useState(null); // index 0-7
  const [draggingContentRect, setDraggingContentRect] = useState(null);     // { startX, startY, rect: [x1, y1, x2, y2] }

  const MAX_W = 900, MAX_H = 640;
  const CP_RADIUS = 8;
  const CORNER_RADIUS = 10;
  const HANDLE_RADIUS = 6;

  // ── Draw ─────────────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgRef.current) return;
    const ctx = canvas.getContext('2d');
    canvas.width = MAX_W; canvas.height = MAX_H;

    const scale = scaleRef.current;
    const offset = offsetRef.current;

    // Background checkerboard
    ctx.fillStyle = '#0d0d15';
    ctx.fillRect(0, 0, MAX_W, MAX_H);
    const tile = 16;
    for (let y = 0; y < MAX_H; y += tile)
      for (let x = 0; x < MAX_W; x += tile) {
        ctx.fillStyle = ((x / tile + y / tile) % 2) === 0 ? '#12121e' : '#0f0f1a';
        ctx.fillRect(x, y, tile, tile);
      }

    // Image
    const iw = imgRef.current.width * scale;
    const ih = imgRef.current.height * scale;
    ctx.drawImage(imgRef.current, offset.x, offset.y, iw, ih);

    // Coordinate helpers for drawing
    const toCanvas = (p) => {
      const px = p.x !== undefined ? p.x : p[0];
      const py = p.y !== undefined ? p.y : p[1];
      return {
        x: px * scale + offset.x,
        y: py * scale + offset.y,
      };
    };

    // ── Poly dewarp curves (4 B-spline curves from dewarp2) ────────────────
    if (dewarpGrid && dewarpGrid.detected && dewarpGrid.polyCurves) {
      const curves = dewarpGrid.polyCurves;
      const curveNames = ['Top', 'Upper-Mid', 'Mid', 'Lower-Mid', 'Bottom'];
      curves.forEach((curve, ci) => {
        // Use precomputed points or generate from control points via evalSpline
        const pts = (curve.points && curve.points.length > 0)
          ? curve.points
          : evalSpline(curve.control_points.map(c => [c.x, c.y]), 80).map(p => ({ x: p[0], y: p[1] }));
          
        if (!pts || pts.length < 2) return;

        // Draw the curve line
        ctx.beginPath();
        pts.forEach((p, i) => {
          const cp = toCanvas(p);
          i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
        });
        ctx.strokeStyle = curve.color || ['#ef4444', '#22c55e', '#f97316', '#3b82f6', '#eab308'][ci];
        ctx.lineWidth = 1.5; // Thinner line
        ctx.setLineDash([]);
        ctx.stroke();

        // Glow effect
        ctx.beginPath();
        pts.forEach((p, i) => {
          const cp = toCanvas(p);
          i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
        });
        ctx.strokeStyle = (curve.color || ['#ef4444', '#22c55e', '#f97316', '#3b82f6', '#eab308'][ci]).replace(')', ', 0.25)').replace('rgb', 'rgba');
        ctx.lineWidth = 4; // Thinner glow
        ctx.stroke();
        ctx.lineWidth = 1.5;

        // Draw control points
        const cps = curve.control_points;
        if (cps && cps.length > 0) {
          cps.forEach((p, idx) => {
            const cp = toCanvas(p);
            const isDragging = draggingCP && draggingCP.type === 'poly' && draggingCP.curveIndex === ci && draggingCP.idx === idx;

            // Halo
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, isDragging ? 12 : 9, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
            ctx.fill();

            // Main dot
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, isDragging ? 7 : 6, 0, Math.PI * 2);
            ctx.fillStyle = curve.color || ['#ef4444', '#22c55e', '#f97316', '#3b82f6', '#eab308'][ci];
            ctx.fill();
            
            // Border
            ctx.lineWidth = 2;
            ctx.strokeStyle = isDragging ? '#173b3c' : '#ffffff';
            ctx.stroke();

            // White center
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
          });
        }

        // Label on the left
        const firstPt = toCanvas(pts[0]);
        ctx.font = '600 10px Inter, system-ui';
        ctx.fillStyle = curve.color || ['#ef4444', '#22c55e', '#f97316', '#3b82f6', '#eab308'][ci];
        ctx.textAlign = 'left';
        ctx.fillText(curveNames[ci] || curve.name, firstPt.x + 6, firstPt.y - 5);
      });

      // Info label
      ctx.font = '600 11px Inter, system-ui';
      ctx.fillStyle = 'rgba(179,157,219,0.9)';
      ctx.textAlign = 'left';
      ctx.fillText(`Poly Curves · 5 B-spline text-density curves detected`, offset.x + 6, offset.y + 16);

    // ── ScanTailor dewarp grid (sparse control points + spline) ────────────
    } else if (dewarpGrid && dewarpGrid.detected && dewarpGrid.topCPs && dewarpGrid.botCPs) {
      const topCPs = dewarpGrid.topCPs;
      const botCPs = dewarpGrid.botCPs;
      const topDense = evalSpline(topCPs, 80);
      const botDense = evalSpline(botCPs, 80);
      const colLines = buildGridFromCurves(topCPs, botCPs, 20, 20);

      // Draw inner col lines (vertical generatrices) — blue thin
      ctx.strokeStyle = 'rgba(70, 120, 255, 0.55)';
      ctx.lineWidth = 0.8;
      ctx.setLineDash([]);
      for (let ci = 1; ci < colLines.length - 1; ci++) {
        const col = colLines[ci];
        ctx.beginPath();
        col.forEach((p, i) => {
          const cp = toCanvas(p);
          i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
        });
        ctx.stroke();
      }

      // Draw inner horizontal rows — blue thin
      ctx.strokeStyle = 'rgba(70, 120, 255, 0.55)';
      ctx.lineWidth = 0.8;
      const nRows = colLines[0].length;
      for (let ri = 1; ri < nRows - 1; ri++) {
        ctx.beginPath();
        colLines.forEach((col, ci) => {
          const cp = toCanvas(col[ri]);
          ci === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
        });
        ctx.stroke();
      }

      // Draw TOP spline — red, thick
      ctx.strokeStyle = 'rgba(255, 60, 60, 0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      topDense.forEach((p, i) => {
        const cp = toCanvas(p);
        i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
      });
      ctx.stroke();

      // Draw BOTTOM spline — red, thick
      ctx.beginPath();
      botDense.forEach((p, i) => {
        const cp = toCanvas(p);
        i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
      });
      ctx.stroke();

      // Draw LEFT and RIGHT edges
      ctx.strokeStyle = 'rgba(255, 60, 60, 0.75)';
      ctx.lineWidth = 1.5;
      const leftCol = colLines[0], rightCol = colLines[colLines.length - 1];
      [leftCol, rightCol].forEach(col => {
        ctx.beginPath();
        col.forEach((p, i) => {
          const cp = toCanvas(p);
          i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
        });
        ctx.stroke();
      });

      // Draw sparse CONTROL POINTS on top and bottom curves
      const drawCPs = (cps, curve) => {
        cps.forEach((p, idx) => {
          const cp = toCanvas(p);
          const isEnd = idx === 0 || idx === cps.length - 1;
          const isDragging = draggingCP && draggingCP.curve === curve && draggingCP.idx === idx;

          // Glow
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, CP_RADIUS + 5, 0, Math.PI * 2);
          ctx.fillStyle = isDragging ? 'rgba(255,100,0,0.25)' : (isEnd ? 'rgba(255,60,60,0.2)' : 'rgba(80,160,255,0.2)');
          ctx.fill();

          // Main dot
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, CP_RADIUS, 0, Math.PI * 2);
          ctx.fillStyle = isDragging ? '#ff6422' : (isEnd ? '#ff3333' : '#4a9eff');
          ctx.fill();

          // White center
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#fff';
          ctx.fill();
        });
      };
      drawCPs(topCPs, 'top');
      drawCPs(botCPs, 'bot');

      // Info label
      ctx.font = '600 11px Inter, system-ui';
      ctx.fillStyle = 'rgba(80,130,255,0.9)';
      ctx.textAlign = 'left';
      ctx.fillText(`ScanTailor Grid · ${dewarpGrid.row_count || 0} text lines · drag control points`, offset.x + 6, offset.y + 16);

    } else if (!dewarpGrid && corners && corners.length === 4) {
      // ── Perspective mode ──────────────────────────────────────────────
      ctx.beginPath();
      corners.forEach((c, i) => {
        const cp = toCanvas(c);
        i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
      });
      ctx.closePath();
      ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(0,229,255,0.06)'; ctx.fill();

      corners.forEach((c, i) => {
        const cp = toCanvas(c);
        ctx.beginPath(); ctx.arc(cp.x, cp.y, CORNER_RADIUS + 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,229,255,0.2)'; ctx.fill();
        ctx.beginPath(); ctx.arc(cp.x, cp.y, CORNER_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = draggingCorner === i ? '#ff6b35' : '#00e5ff'; ctx.fill();
        ctx.beginPath(); ctx.arc(cp.x, cp.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#fff'; ctx.fill();
        ctx.font = '600 11px Inter'; ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
        ctx.fillText(['TL', 'TR', 'BR', 'BL'][i], cp.x, cp.y - CORNER_RADIUS - 8);
      });
    } else if (!dewarpGrid && !corners && contentRect && contentRect.length === 4) {
      // ── Content Selection mode ─────────────────────────────────────────
      const [x1, y1, x2, y2] = contentRect;
      const cp1 = toCanvas([x1, y1]);
      const cp2 = toCanvas([x2, y2]);

      // Semi-transparent blue fill
      ctx.fillStyle = 'rgba(38, 198, 218, 0.14)';
      ctx.fillRect(cp1.x, cp1.y, cp2.x - cp1.x, cp2.y - cp1.y);

      // Dash/Solid outline
      ctx.strokeStyle = '#26c6da';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(cp1.x, cp1.y, cp2.x - cp1.x, cp2.y - cp1.y);
      ctx.setLineDash([]);

      // Draw handles
      const handles = [
        { x: cp1.x, y: cp1.y }, // TL (0)
        { x: cp2.x, y: cp1.y }, // TR (1)
        { x: cp2.x, y: cp2.y }, // BR (2)
        { x: cp1.x, y: cp2.y }, // BL (3)
        { x: (cp1.x + cp2.x) / 2, y: cp1.y }, // T (4)
        { x: cp2.x, y: (cp1.y + cp2.y) / 2 }, // R (5)
        { x: (cp1.x + cp2.x) / 2, y: cp2.y }, // B (6)
        { x: cp1.x, y: (cp1.y + cp2.y) / 2 }, // L (7)
      ];

      handles.forEach((h, i) => {
        ctx.beginPath();
        ctx.arc(h.x, h.y, HANDLE_RADIUS + 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(38, 198, 218, 0.2)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(h.x, h.y, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = draggingContentHandle === i ? '#ff6b35' : '#26c6da';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(h.x, h.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      });
    }
  }, [corners, dewarpGrid, contentRect, draggingCP, draggingCorner, draggingContentHandle]);

  useEffect(() => { draw(); }, [draw]);

  // ── Load image ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      const s = Math.min(MAX_W / img.width, MAX_H / img.height, 1);
      scaleRef.current = s;
      offsetRef.current = { x: (MAX_W - img.width * s) / 2, y: (MAX_H - img.height * s) / 2 };
      draw(); // Draw immediately upon loading
    };
    img.src = imageSrc;
  }, [imageSrc, draw]);

  // ── Coordinate helpers ───────────────────────────────────────────────────
  const toImage = useCallback((cx, cy) => {
    const scale = scaleRef.current;
    const offset = offsetRef.current;
    return {
      x: Math.max(0, Math.min(imageWidth, (cx - offset.x) / scale)),
      y: Math.max(0, Math.min(imageHeight, (cy - offset.y) / scale)),
    };
  }, [imageWidth, imageHeight]);

  const toCanvas = useCallback((p) => {
    const scale = scaleRef.current;
    const offset = offsetRef.current;
    const px = p.x !== undefined ? p.x : p[0];
    const py = p.y !== undefined ? p.y : p[1];
    return {
      x: px * scale + offset.x,
      y: py * scale + offset.y,
    };
  }, []);

  const getMousePos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  // ── Hit testing ──────────────────────────────────────────────────────────
  const findCP = (mx, my) => {
    if (dewarpGrid?.polyCurves) {
      for (let ci = 0; ci < dewarpGrid.polyCurves.length; ci++) {
        const cps = dewarpGrid.polyCurves[ci].control_points;
        if (!cps) continue;
        for (let idx = 0; idx < cps.length; idx++) {
          const cp = toCanvas(cps[idx]);
          if (Math.hypot(mx - cp.x, my - cp.y) <= 14) return { type: 'poly', curveIndex: ci, idx };
        }
      }
      return null;
    }

    if (!dewarpGrid?.topCPs) return null;
    for (const [curve, cps] of [
      ['top',      dewarpGrid.topCPs],
      ['bot',      dewarpGrid.botCPs],
    ]) {
      if (!cps) continue;
      for (let idx = 0; idx < cps.length; idx++) {
        const cp = toCanvas(cps[idx]);
        if (Math.hypot(mx - cp.x, my - cp.y) <= CP_RADIUS + 8) return { type: 'scantailor', curve, idx };
      }
    }
    return null;
  };

  const findCorner = (mx, my) => {
    if (!corners || dewarpGrid?.detected) return -1;
    for (let i = 0; i < corners.length; i++) {
      const cp = toCanvas(corners[i]);
      if (Math.hypot(mx - cp.x, my - cp.y) <= CORNER_RADIUS + 6) return i;
    }
    return -1;
  };

  // ── Mouse handlers ───────────────────────────────────────────────────────
  const handleMouseDown = (e) => {
    const { x, y } = getMousePos(e);
    if (dewarpGrid?.detected && onDewarpGridChange) {
      const hit = findCP(x, y);
      if (hit) { setDraggingCP(hit); e.preventDefault(); }
    } else if (corners && corners.length === 4) {
      const idx = findCorner(x, y);
      if (idx >= 0) { setDraggingCorner(idx); e.preventDefault(); }
    } else if (contentRect && contentRect.length === 4) {
      const [x1, y1, x2, y2] = contentRect;
      const cp1 = toCanvas([x1, y1]);
      const cp2 = toCanvas([x2, y2]);

      const handles = [
        { x: cp1.x, y: cp1.y }, // TL
        { x: cp2.x, y: cp1.y }, // TR
        { x: cp2.x, y: cp2.y }, // BR
        { x: cp1.x, y: cp2.y }, // BL
        { x: (cp1.x + cp2.x) / 2, y: cp1.y }, // T
        { x: cp2.x, y: (cp1.y + cp2.y) / 2 }, // R
        { x: (cp1.x + cp2.x) / 2, y: cp2.y }, // B
        { x: cp1.x, y: (cp1.y + cp2.y) / 2 }, // L
      ];

      for (let i = 0; i < handles.length; i++) {
        if (Math.hypot(x - handles[i].x, y - handles[i].y) <= 12) {
          setDraggingContentHandle(i);
          e.preventDefault();
          return;
        }
      }

      // Check inside
      if (x >= Math.min(cp1.x, cp2.x) && x <= Math.max(cp1.x, cp2.x) &&
        y >= Math.min(cp1.y, cp2.y) && y <= Math.max(cp1.y, cp2.y)) {
        setDraggingContentRect({ startX: x, startY: y, rect: [...contentRect] });
        e.preventDefault();
        return;
      }
    }
  };

  const handleMouseMove = (e) => {
    const { x, y } = getMousePos(e);
    const canvas = canvasRef.current;

    if (dewarpGrid?.detected && onDewarpGridChange) {
      canvas.style.cursor = findCP(x, y) || draggingCP ? 'grab' : 'default';
      if (!draggingCP) return;

      const imgPt = toImage(x, y);

      if (draggingCP.type === 'poly') {
        const { curveIndex, idx } = draggingCP;
        const newGrid = JSON.parse(JSON.stringify(dewarpGrid));
        const curve = newGrid.polyCurves[curveIndex];

        // Bounds logic from dewarp2
        const horizontalMargin = Math.max(2, imageWidth * 0.004);
        const verticalMargin = Math.max(8, imageHeight * 0.035);
        
        const prevX = idx === 0 ? 0 : curve.control_points[idx - 1].x + horizontalMargin;
        const nextX = idx === curve.control_points.length - 1 ? imageWidth - 1 : curve.control_points[idx + 1].x - horizontalMargin;
        
        const prevY = curveIndex === 0 ? 0 : newGrid.polyCurves[curveIndex - 1].control_points[idx].y + verticalMargin;
        const nextY = curveIndex === newGrid.polyCurves.length - 1 ? imageHeight - 1 : newGrid.polyCurves[curveIndex + 1].control_points[idx].y - verticalMargin;
        
        curve.points = []; // Clear to force evalSpline on next draw
        curve.control_points[idx] = {
          x: Math.min(nextX, Math.max(prevX, imgPt.x)),
          y: Math.min(nextY, Math.max(prevY, imgPt.y)),
        };
        onDewarpGridChange(newGrid);
        return;
      }

      const { curve, idx } = draggingCP;
      const newGrid = JSON.parse(JSON.stringify(dewarpGrid));
      const cps =
        curve === 'top'      ? newGrid.topCPs :
                               newGrid.botCPs;
      const n = cps.length;
      const isEndpoint = idx === 0 || idx === n - 1;

      if (!isEndpoint) {
        // Simple move — only this control point
        cps[idx] = [imgPt.x, imgPt.y];
      } else {
        // ── Endpoint lever rotation (ScanTailor logic) ────────────────
        const originIdx = idx === 0 ? n - 1 : 0;
        const origin = cps[originIdx];
        const oldPt = cps[idx];
        const fromVec = { x: oldPt[0] - origin[0], y: oldPt[1] - origin[1] };
        const toVec = { x: imgPt.x - origin[0], y: imgPt.y - origin[1] };
        const fromLen2 = fromVec.x ** 2 + fromVec.y ** 2;

        if (fromLen2 > 1) {
          const dot = fromVec.x * toVec.x + fromVec.y * toVec.y;
          const cross = fromVec.x * toVec.y - fromVec.y * toVec.x;
          const a = dot / fromLen2;
          const b = cross / fromLen2;
          for (let i = 0; i < n; i++) {
            const dx = cps[i][0] - origin[0];
            const dy = cps[i][1] - origin[1];
            cps[i] = [
              origin[0] + a * dx - b * dy,
              origin[1] + b * dx + a * dy,
            ];
          }
        }
      }
      onDewarpGridChange(newGrid);
    } else if (corners && corners.length === 4) {
      canvas.style.cursor = findCorner(x, y) >= 0 || draggingCorner !== null ? 'grab' : 'default';
      if (draggingCorner !== null && corners && onCornersChange) {
        const imgPt = toImage(x, y);
        const newCorners = [...corners];
        newCorners[draggingCorner] = imgPt;
        onCornersChange(newCorners);
      }
    } else if (contentRect && contentRect.length === 4) {
      const scale = scaleRef.current;

      // Update cursor on hover
      if (draggingContentHandle === null && !draggingContentRect) {
        const [x1, y1, x2, y2] = contentRect;
        const cp1 = toCanvas([x1, y1]);
        const cp2 = toCanvas([x2, y2]);

        const handles = [
          { x: cp1.x, y: cp1.y, cursor: 'nwse-resize' }, // TL
          { x: cp2.x, y: cp1.y, cursor: 'nesw-resize' }, // TR
          { x: cp2.x, y: cp2.y, cursor: 'nwse-resize' }, // BR
          { x: cp1.x, y: cp2.y, cursor: 'nesw-resize' }, // BL
          { x: (cp1.x + cp2.x) / 2, y: cp1.y, cursor: 'ns-resize' }, // T
          { x: cp2.x, y: (cp1.y + cp2.y) / 2, cursor: 'ew-resize' }, // R
          { x: (cp1.x + cp2.x) / 2, y: cp2.y, cursor: 'ns-resize' }, // B
          { x: cp1.x, y: (cp1.y + cp2.y) / 2, cursor: 'ew-resize' }, // L
        ];

        let hit = false;
        for (let i = 0; i < handles.length; i++) {
          if (Math.hypot(x - handles[i].x, y - handles[i].y) <= 12) {
            canvas.style.cursor = handles[i].cursor;
            hit = true;
            break;
          }
        }

        if (!hit) {
          if (x >= Math.min(cp1.x, cp2.x) && x <= Math.max(cp1.x, cp2.x) &&
            y >= Math.min(cp1.y, cp2.y) && y <= Math.max(cp1.y, cp2.y)) {
            canvas.style.cursor = 'move';
            hit = true;
          }
        }

        if (!hit) {
          canvas.style.cursor = 'default';
        }
      }

      // Dragging handle
      if (draggingContentHandle !== null && onContentRectChange) {
        const imgPt = toImage(x, y);
        const newRect = [...contentRect];
        const hIdx = draggingContentHandle;

        if (hIdx === 0) { newRect[0] = imgPt.x; newRect[1] = imgPt.y; }
        else if (hIdx === 1) { newRect[2] = imgPt.x; newRect[1] = imgPt.y; }
        else if (hIdx === 2) { newRect[2] = imgPt.x; newRect[3] = imgPt.y; }
        else if (hIdx === 3) { newRect[0] = imgPt.x; newRect[3] = imgPt.y; }
        else if (hIdx === 4) { newRect[1] = imgPt.y; }
        else if (hIdx === 5) { newRect[2] = imgPt.x; }
        else if (hIdx === 6) { newRect[3] = imgPt.y; }
        else if (hIdx === 7) { newRect[0] = imgPt.x; }

        onContentRectChange([
          Math.round(newRect[0]),
          Math.round(newRect[1]),
          Math.round(newRect[2]),
          Math.round(newRect[3])
        ]);
        return;
      }

      // Dragging entire rect
      if (draggingContentRect && onContentRectChange) {
        const { startX, startY, rect } = draggingContentRect;
        const dx = (x - startX) / scale;
        const dy = (y - startY) / scale;

        let nx1 = rect[0] + dx;
        let ny1 = rect[1] + dy;
        let nx2 = rect[2] + dx;
        let ny2 = rect[3] + dy;

        const rw = rect[2] - rect[0];
        const rh = rect[3] - rect[1];

        if (nx1 < 0) { nx1 = 0; nx2 = rw; }
        if (nx2 > imageWidth) { nx2 = imageWidth; nx1 = imageWidth - rw; }
        if (ny1 < 0) { ny1 = 0; ny2 = rh; }
        if (ny2 > imageHeight) { ny2 = imageHeight; ny1 = imageHeight - rh; }

        onContentRectChange([
          Math.round(nx1),
          Math.round(ny1),
          Math.round(nx2),
          Math.round(ny2)
        ]);
        return;
      }
    }
  };

  const handleMouseUp = () => {
    setDraggingCP(null);
    setDraggingCorner(null);
    setDraggingContentHandle(null);
    setDraggingContentRect(null);
  };

  return (
    <div className="canvas-editor">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ borderRadius: '12px', display: 'block', width: `${MAX_W}px`, height: `${MAX_H}px` }}
      />
      <div className="canvas-hint">
        {dewarpGrid?.detected && dewarpGrid?.polyCurves ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#b39ddb" strokeWidth="2">
              <path d="M4 12 Q8 6 12 12 Q16 18 20 12" />
            </svg>
            Drag <strong>control points</strong> to adjust curves · <strong style={{color:'#b39ddb'}}>click Apply Poly Dewarp</strong> to flatten
          </>
        ) : dewarpGrid?.detected ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" strokeWidth="2">
              <path d="M4 4h16v16H4zM4 12h16M12 4v16" />
            </svg>
            Drag <strong>red endpoints</strong> to tilt curves · drag <strong>blue midpoints</strong> to adjust individual sections
          </>
        ) : corners && corners.length === 4 ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 9l7-7 7 7M5 15l7 7 7-7" />
            </svg>
            Drag the corner points to adjust document boundaries
          </>
        ) : contentRect && contentRect.length === 4 ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#26c6da" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M9 9h6v6H9z" />
            </svg>
            Drag handles to <strong>resize</strong> · drag inside the box to <strong>move</strong> content boundary
          </>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 4v16M4 12h16" />
            </svg>
            Use controls in the sidebar to configure operations
          </>
        )}
      </div>
    </div>
  );
}

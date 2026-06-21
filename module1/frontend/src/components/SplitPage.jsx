import { useState, useCallback, useEffect, useRef } from 'react';
import ApplyAllButton from './ApplyAllButton';

/**
 * SplitPage — ScanTailor Advanced–style Split Pages panel (#2)
 *
 * Exports:
 *   default SplitPage        — sidebar panel
 *   SplitBoundaryOverlay     — canvas overlay (mounted by App.jsx)
 *
 * Overlay implements the full ScanTailor interaction model:
 *   • 4 draggable corner handles defining the page boundary (blue)
 *   • A draggable cutter line with TWO independent endpoints
 *     — drag body: translates horizontally (pure move)
 *     — drag endpoint: rotates the cutter (top endpoint stays on top edge,
 *                      bottom stays on bottom edge of boundary polygon)
 *   • Left / right page region highlight (click to select)
 *   • Dark off-cut mask outside the boundary
 */

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers (ratios-space, mirrors page_layout.py)
// ─────────────────────────────────────────────────────────────────────────────

/** Interpolate y along a line from p1→p2 at a given x (all in ratio coords) */
const lerp_y = (p1, p2, x) => {
  const dx = p2.x - p1.x;
  if (Math.abs(dx) < 1e-6) return (p1.y + p2.y) / 2;
  const t = (x - p1.x) / dx;
  return p1.y + t * (p2.y - p1.y);
};

/** Clamp a value between min and max */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ─────────────────────────────────────────────────────────────────────────────
// SplitBoundaryOverlay — exported for App.jsx
// ─────────────────────────────────────────────────────────────────────────────

export function SplitBoundaryOverlay({
  // --- Boundary polygon (4 corner ratio points) ---
  boundary,              // [{x,y}, {x,y}, {x,y}, {x,y}] → [TL, TR, BR, BL]
  onBoundaryChange,

  // --- Cutter line (two draggable endpoints in ratio coords) ---
  cutter,                // {top: {x,y}, bottom: {x,y}} | null
  onCutterChange,

  // --- Fallback: simple x-ratio when cutter is null ---
  xRatio,
  onXRatioChange,

  // --- Layout info ---
  layoutType,            // 'two_pages' | 'single_uncut' | 'single_cut'
  selectedSide,
  onSideSelect,
  showSplitLine,
}) {
  const wrapRef = useRef(null);

  // Dragging state: null | 'corner-0..3' | 'cutter-body' | 'cutter-top' | 'cutter-bottom'
  const [dragging,    setDragging]    = useState(null);
  const [dragStart,   setDragStart]   = useState({ x: 0, y: 0 });
  const [cutterStart, setCutterStart] = useState(null); // snapshot of cutter on body-drag start
  const [hoverCorner, setHoverCorner] = useState(null);
  const [hoverCutter, setHoverCutter] = useState(null); // null | 'body' | 'top' | 'bottom'

  // === Derived boundary (default = 2% inset rectangle) ===
  const pts = boundary || [
    { x: 0.02, y: 0.02 }, { x: 0.98, y: 0.02 },
    { x: 0.98, y: 0.98 }, { x: 0.02, y: 0.98 },
  ];
  const [tl, tr, br, bl] = pts;

  // === Derived cutter endpoints ===
  const rawX = xRatio ?? 0.5;
  const resolvedCutter = cutter || {
    top:    { x: rawX, y: lerp_y(tl, tr, rawX) },
    bottom: { x: rawX, y: lerp_y(bl, br, rawX) },
  };
  const cutTop = resolvedCutter.top;
  const cutBot = resolvedCutter.bottom;

  // Clamp x within boundary
  const minX = Math.max(tl.x, bl.x) + 0.01;
  const maxX = Math.min(tr.x, br.x) - 0.01;

  // === SVG helpers ===
  const pct = (r) => `${(r * 100).toFixed(3)}%`;

  const quadPoints = pts.map(p => `${pct(p.x)},${pct(p.y)}`).join(' ');

  // Left page polygon: TL → cutTop → cutBot → BL
  const leftPoly  = [tl, cutTop, cutBot, bl].map(p => `${pct(p.x)},${pct(p.y)}`).join(' ');
  // Right page polygon: cutTop → TR → BR → cutBot
  const rightPoly = [cutTop, tr, br, cutBot].map(p => `${pct(p.x)},${pct(p.y)}`).join(' ');

  // === Pointer utilities ===
  const getRatios = (e) => {
    const el = wrapRef.current;
    if (!el) return { xr: 0, yr: 0 };
    const rect = el.getBoundingClientRect();
    return {
      xr: clamp((e.clientX - rect.left) / rect.width,  0, 1),
      yr: clamp((e.clientY - rect.top)  / rect.height, 0, 1),
    };
  };

  // === Global mouse tracking ===
  useEffect(() => {
    if (!dragging) return;

    const onMove = (e) => {
      const { xr, yr } = getRatios(e);

      if (dragging.startsWith('corner-')) {
        const idx = +dragging.split('-')[1];
        const next = [...pts];
        next[idx] = { x: xr, y: yr };
        onBoundaryChange?.(next);
        return;
      }

      if (dragging === 'cutter-top') {
        const cx = clamp(xr, minX, maxX);
        const cy = lerp_y(tl, tr, cx);           // snap to top boundary edge
        const next = { ...resolvedCutter, top: { x: cx, y: cy } };
        onCutterChange?.(next);
        onXRatioChange?.((next.top.x + next.bottom.x) / 2);
        return;
      }

      if (dragging === 'cutter-bottom') {
        const cx = clamp(xr, minX, maxX);
        const cy = lerp_y(bl, br, cx);           // snap to bottom boundary edge
        const next = { ...resolvedCutter, bottom: { x: cx, y: cy } };
        onCutterChange?.(next);
        onXRatioChange?.((next.top.x + next.bottom.x) / 2);
        return;
      }

      if (dragging === 'cutter-body' && cutterStart) {
        const dx = xr - dragStart.x;
        const topX = clamp(cutterStart.top.x + dx, minX, maxX);
        const botX = clamp(cutterStart.bottom.x + dx, minX, maxX);
        const next = {
          top:    { x: topX, y: lerp_y(tl, tr, topX) },
          bottom: { x: botX, y: lerp_y(bl, br, botX) },
        };
        onCutterChange?.(next);
        onXRatioChange?.((topX + botX) / 2);
      }
    };

    const onUp = () => {
      setDragging(null);
      setCutterStart(null);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging, pts, resolvedCutter, cutterStart, dragStart, tl, tr, bl, br, minX, maxX]);

  const startCornerDrag = (e, idx) => {
    e.preventDefault(); e.stopPropagation();
    setDragging(`corner-${idx}`);
  };

  const startCutterTopDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragging('cutter-top');
  };

  const startCutterBotDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragging('cutter-bottom');
  };

  const startBodyDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    const { xr, yr } = getRatios(e);
    setDragging('cutter-body');
    setDragStart({ x: xr, y: yr });
    setCutterStart({ ...resolvedCutter });
  };

  const isTwoPage       = layoutType === 'two_pages';
  const activeDrag      = dragging !== null;
  const cutterDragging  = dragging?.startsWith('cutter');
  const cutterColor     = cutterDragging ? 'rgba(255,210,50,1)' : 'rgba(140,80,255,0.92)';

  // Both regions use a blue mask. Selected = bright; unselected = near-invisible; none = subtle.
  const leftAlpha  = selectedSide === 'left'  ? 0.28 : selectedSide === 'right' ? 0.03 : 0.08;
  const rightAlpha = selectedSide === 'right' ? 0.28 : selectedSide === 'left'  ? 0.03 : 0.08;
  const leftStroke  = selectedSide === 'left'  ? 'rgba(40,130,255,0.9)' : 'transparent';
  const rightStroke = selectedSide === 'right' ? 'rgba(40,130,255,0.9)' : 'transparent';

  return (
    <div
      ref={wrapRef}
      style={{
        position: 'absolute', inset: 0,
        pointerEvents: activeDrag ? 'all' : 'none',
        zIndex: 15,
        cursor: cutterDragging ? 'col-resize' : activeDrag ? 'grabbing' : 'default',
      }}
    >
      {/* ── SVG: outlines, regions, cutter ── */}
      <svg width="100%" height="100%"
           style={{ position: 'absolute', inset: 0, overflow: 'visible' }}>
        <defs>
          <mask id="offcut-mask">
            <rect width="100%" height="100%" fill="white" />
            <polygon points={quadPoints} fill="black" />
          </mask>
        </defs>

        {/* Off-cut darkening (outside boundary) */}
        <rect width="100%" height="100%"
              fill="rgba(0,0,0,0.38)"
              mask="url(#offcut-mask)"
              style={{ pointerEvents: 'none' }} />

        {/* Boundary outline */}
        <polygon points={quadPoints}
                 fill="rgba(30,100,255,0.04)"
                 stroke="rgba(50,140,255,0.85)"
                 strokeWidth="1.5" strokeDasharray="7 4"
                 style={{ pointerEvents: 'none' }} />

        {/* ── Left page region — thin blue selection mask ── */}
        {isTwoPage && showSplitLine && (
          <polygon points={leftPoly}
                   fill={`rgba(30,100,255,${leftAlpha})`}
                   stroke={leftStroke}
                   strokeWidth="1.5"
                   style={{ pointerEvents: 'all', cursor: 'pointer' }}
                   onClick={() => onSideSelect?.('left')} />
        )}

        {/* ── Right page region — thin blue selection mask ── */}
        {isTwoPage && showSplitLine && (
          <polygon points={rightPoly}
                   fill={`rgba(30,100,255,${rightAlpha})`}
                   stroke={rightStroke}
                   strokeWidth="1.5"
                   style={{ pointerEvents: 'all', cursor: 'pointer' }}
                   onClick={() => onSideSelect?.('right')} />
        )}

        {/* ── Cutter line ── */}
        {showSplitLine && (<>
          {/* Wide invisible hit area for body drag */}
          <line x1={pct(cutTop.x)} y1={pct(cutTop.y)}
                x2={pct(cutBot.x)} y2={pct(cutBot.y)}
                stroke="transparent" strokeWidth="16"
                style={{ pointerEvents: 'all', cursor: 'col-resize' }}
                onMouseDown={startBodyDrag}
                onMouseEnter={() => setHoverCutter('body')}
                onMouseLeave={() => setHoverCutter(null)} />

          {/* Visible cutter line */}
          <line x1={pct(cutTop.x)} y1={pct(cutTop.y)}
                x2={pct(cutBot.x)} y2={pct(cutBot.y)}
                stroke={cutterColor}
                strokeWidth={cutterDragging ? '2.5' : '1.8'}
                strokeDasharray="9 5"
                style={{ pointerEvents: 'none' }} />

          {/* Cutter midpoint grab indicator */}
          <circle cx={pct((cutTop.x + cutBot.x) / 2)}
                  cy={pct((cutTop.y + cutBot.y) / 2)}
                  r="5"
                  fill={hoverCutter === 'body' ? 'rgba(200,160,255,0.9)' : 'rgba(140,80,255,0.5)'}
                  stroke="white" strokeWidth="1.5"
                  style={{ pointerEvents: 'none' }} />

          {/* L/R page labels */}
          {isTwoPage && (<>
            {/* LEFT label — both labels use blue tones */}
            <text x={pct((tl.x + cutTop.x) / 2)} y={pct(cutTop.y - 0.018)}
                  textAnchor="middle" dominantBaseline="auto"
                  fontSize="11" fontWeight="700" fontFamily="Inter, sans-serif"
                  fill={selectedSide === 'left' ? 'rgba(80,160,255,1)' : 'rgba(80,160,255,0.42)'}
                  style={{ pointerEvents: 'none', userSelect: 'none' }}>
              LEFT
            </text>
            {/* RIGHT label — same blue palette */}
            <text x={pct((cutTop.x + tr.x) / 2)} y={pct(cutTop.y - 0.018)}
                  textAnchor="middle" dominantBaseline="auto"
                  fontSize="11" fontWeight="700" fontFamily="Inter, sans-serif"
                  fill={selectedSide === 'right' ? 'rgba(80,160,255,1)' : 'rgba(80,160,255,0.42)'}
                  style={{ pointerEvents: 'none', userSelect: 'none' }}>
              RIGHT
            </text>
          </>)}
        </>)}
      </svg>

      {/* ── Cutter endpoint handles (DOM — easier drag) ── */}
      {showSplitLine && (<>
        {/* TOP endpoint — diamond */}
        <div
          onMouseDown={startCutterTopDrag}
          onMouseEnter={() => setHoverCutter('top')}
          onMouseLeave={() => setHoverCutter(null)}
          style={{
            position: 'absolute',
            left: `${cutTop.x * 100}%`, top: `${cutTop.y * 100}%`,
            transform: 'translate(-50%, -50%) rotate(45deg)',
            width: 18, height: 18,
            background: dragging === 'cutter-top' || hoverCutter === 'top'
              ? 'rgba(255,210,50,0.95)' : 'rgba(140,80,255,0.9)',
            border: '2px solid white',
            cursor: 'col-resize', pointerEvents: 'all', zIndex: 22,
            boxShadow: '0 0 0 3px rgba(140,80,255,0.2), 0 2px 6px rgba(0,0,0,0.4)',
            transition: 'background 0.12s',
          }}
        />
        {/* BOTTOM endpoint — diamond */}
        <div
          onMouseDown={startCutterBotDrag}
          onMouseEnter={() => setHoverCutter('bottom')}
          onMouseLeave={() => setHoverCutter(null)}
          style={{
            position: 'absolute',
            left: `${cutBot.x * 100}%`, top: `${cutBot.y * 100}%`,
            transform: 'translate(-50%, -50%) rotate(45deg)',
            width: 18, height: 18,
            background: dragging === 'cutter-bottom' || hoverCutter === 'bottom'
              ? 'rgba(255,210,50,0.95)' : 'rgba(140,80,255,0.9)',
            border: '2px solid white',
            cursor: 'col-resize', pointerEvents: 'all', zIndex: 22,
            boxShadow: '0 0 0 3px rgba(140,80,255,0.2), 0 2px 6px rgba(0,0,0,0.4)',
            transition: 'background 0.12s',
          }}
        />
      </>)}

      {/* ── Corner handles (circles) ── */}
      {pts.map((pt, i) => (
        <div key={i}
          onMouseDown={(e) => startCornerDrag(e, i)}
          onMouseEnter={() => setHoverCorner(i)}
          onMouseLeave={() => setHoverCorner(null)}
          style={{
            position: 'absolute',
            left: `${pt.x * 100}%`, top: `${pt.y * 100}%`,
            transform: 'translate(-50%, -50%)',
            width: 24, height: 24, borderRadius: '50%',
            background: dragging === `corner-${i}`
              ? 'rgba(255,210,50,0.95)'
              : hoverCorner === i
                ? 'rgba(80,170,255,0.95)'
                : 'rgba(30,120,255,0.88)',
            border: '2.5px solid rgba(255,255,255,0.95)',
            cursor: 'grab', pointerEvents: 'all', zIndex: 20,
            boxShadow: hoverCorner === i || dragging === `corner-${i}`
              ? '0 0 0 4px rgba(30,120,255,0.2), 0 2px 8px rgba(0,0,0,0.4)'
              : '0 0 0 3px rgba(30,120,255,0.14), 0 2px 6px rgba(0,0,0,0.3)',
            transition: 'background 0.12s, box-shadow 0.12s',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <div style={{ width: 7, height: 7, borderRadius: '50%',
                        background: 'white', opacity: 0.88 }} />
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SplitPage — sidebar panel
// ─────────────────────────────────────────────────────────────────────────────

const LAYOUT_ICONS = {
  two_pages: { label: 'Two Pages', icon: '▭▭' },
};

const confColor = (c) =>
  c >= 0.8 ? '#4caf50' : c >= 0.6 ? '#ff9800' : '#f44336';

export default function SplitPage({
  onDetect,
  onSpineDetect,
  onApply,
  loading,
  hasImage,
  panelOpen,
  splitXRatio,
  onSplitXChange,
  imageWidth,
  onDetectResult,
  pageCount = 1,
}) {
  const [layoutType,    setLayoutType]    = useState(null);
  const [splitX,        setSplitX]        = useState(0.5);
  const [confidence,    setConfidence]    = useState(null);
  const [splitMode,     setSplitMode]     = useState('auto');
  const [detecting,     setDetecting]     = useState(false);
  const [spineDetecting, setSpineDetecting] = useState(false);
  const [spineResult,   setSpineResult]   = useState(null);  // { detected, spine_x_ratio, ... }
  const [activeMethod,  setActiveMethod]  = useState(null);  // 'spine' | 'geometry'
  const [contentX1,     setContentX1]     = useState(0);
  const [contentX2,     setContentX2]     = useState(1);
  const [selectedSide,  setSelectedSide]  = useState(null);
  const [boundary,      setBoundary]      = useState(null);
  const [cutter,        setCutter]        = useState(null);
  const [pageLayout,    setPageLayout]    = useState(null);

  const dis = loading || !hasImage;

  // ── Push full overlay state up ────────────────────────────────────────────
  const pushOverlay = useCallback((overrides = {}) => {
    const data = {
      splitXRatio:    splitX,
      contentX1Ratio: contentX1,
      contentX2Ratio: contentX2,
      layoutType:     layoutType,
      selectedSide,
      splitMode,
      boundary,
      cutter,
      pageLayout,
      ...overrides,
    };
    onDetectResult?.(data);
    const lt = data.layoutType;
    const show = lt !== 'single_uncut' || data.splitMode === 'manual';
    onSplitXChange(show ? data.splitXRatio : null);
  }, [splitX, contentX1, contentX2, layoutType, selectedSide, splitMode,
      boundary, cutter, pageLayout, onDetectResult, onSplitXChange]);

  // ── Auto-detect (Geometry / ScanTailor) ──────────────────────────────────
  const runDetect = useCallback(async () => {
    if (!hasImage) return;
    setDetecting(true);
    setActiveMethod('geometry');
    setSpineResult(null);
    try {
      const res = await onDetect();
      if (!res) return;

      const lt  = res.layout_type ?? 'two_pages';
      const sx  = res.split_x_ratio ?? 0.5;
      const cx1 = res.content_x1_ratio ?? 0;
      const cx2 = res.content_x2_ratio ?? 1;
      const pl  = res.page_layout ?? null;

      // Build boundary from page_layout outline (ratio coords)
      let bnd = null;
      if (pl?.outline && imageWidth) {
        const imgH = (pl.outline[2]?.[1] || 1); // not ideal but sufficient for ratio
        bnd = pl.outline.map(([px, py]) => ({
          x: px / (imageWidth  || 1),
          y: py / (imgH        || 1),
        }));
      } else if (res.boundary) {
        bnd = res.boundary.map(([x, y]) => ({ x, y }));
      }

      // Build cutter from page_layout cutter1 (ratio coords)
      let ct = null;
      if (pl?.cutter1 && imageWidth) {
        const [[cx1p, cy1p], [cx2p, cy2p]] = pl.cutter1;
        const iw = imageWidth || 1;
        const ih = pl.outline ? Math.max(...pl.outline.map(p => p[1])) : iw * 1.4;
        ct = {
          top:    { x: cx1p / iw, y: cy1p / ih },
          bottom: { x: cx2p / iw, y: cy2p / ih },
        };
      }

      setLayoutType(lt); setSplitX(sx); setConfidence(res.confidence);
      setContentX1(cx1); setContentX2(cx2);
      setSelectedSide(null); setBoundary(bnd); setCutter(ct);
      setPageLayout(pl);

      const data = {
        splitXRatio: sx, contentX1Ratio: cx1, contentX2Ratio: cx2,
        layoutType: lt, selectedSide: null, splitMode: 'auto',
        boundary: bnd, cutter: ct, pageLayout: pl,
      };
      onDetectResult?.(data);
      onSplitXChange(lt !== 'single_uncut' ? sx : null);
    } finally {
      setDetecting(false);
    }
  }, [onDetect, hasImage, imageWidth, onSplitXChange, onDetectResult]);

  // ── Auto-detect when the panel first opens ───────────────────────────────
  // Do NOT auto-run either method; wait for user to click a button.
  useEffect(() => {
    if (panelOpen && hasImage) {
      // Reset detection state so overlay is clear until user picks a method
      setLayoutType(null);
      setSpineResult(null);
      setActiveMethod(null);
    }
  }, [panelOpen, hasImage]); // eslint-disable-line

  // ── Spine detection (fast projection + Hough) ─────────────────────────────
  const runSpineDetect = useCallback(async () => {
    if (!hasImage || !onSpineDetect) return;
    setSpineDetecting(true);
    setActiveMethod('spine');
    setSpineResult(null);
    try {
      const res = await onSpineDetect();
      if (!res) return;
      setSpineResult(res);
      if (res.detected) {
        const sx = res.spine_x_ratio ?? 0.5;
        setLayoutType('two_pages');
        setSplitX(sx);
        setConfidence(null);
        setPageLayout(null);   // pixel-crop mode — no geometry warp
        setBoundary(null);
        setCutter(null);
        setSelectedSide(null);
        const data = {
          splitXRatio: sx, contentX1Ratio: 0, contentX2Ratio: 1,
          layoutType: 'two_pages', selectedSide: null, splitMode: 'auto',
          boundary: null, cutter: null, pageLayout: null,
        };
        onDetectResult?.(data);
        onSplitXChange?.(sx);
      } else {
        onSplitXChange?.(null);
      }
    } catch (e) {
      console.error('Spine detect error', e);
    } finally {
      setSpineDetecting(false);
    }
  }, [hasImage, onSpineDetect, onDetectResult, onSplitXChange]);

  // ── Mode / layout type changes ────────────────────────────────────────────
  const handleModeChange = (mode) => {
    setSplitMode(mode);
    pushOverlay({ splitMode: mode });
  };

  const handleLayoutChange = (lt) => {
    setLayoutType(lt);
    setSelectedSide(null);
    pushOverlay({ layoutType: lt, selectedSide: null });
  };

  const handleSlider = (v) => {
    setSplitX(v);
    pushOverlay({ splitXRatio: v, cutter: null }); // null cutter → auto-from-xRatio
  };

  const handleSideSelect = (side) => {
    setSelectedSide(side);
    pushOverlay({ selectedSide: side });
  };

  const handleBoundaryChange = (b) => {
    setBoundary(b);
    pushOverlay({ boundary: b });
  };

  const handleCutterChange = (c) => {
    setCutter(c);
    pushOverlay({ cutter: c, splitXRatio: (c.top.x + c.bottom.x) / 2 });
  };

  // ── Apply ─────────────────────────────────────────────────────────────────
  const handleApply = async () => {
    if (!layoutType) return;
    await onApply({
      splitX:       splitX,
      layoutType,
      contentX1,
      contentX2,
      selectedSide: selectedSide ?? 'both',
      boundary,
      cutter,
      pageLayout,   // pass full geometry model to backend
    });
  };

  const confPct = confidence != null ? Math.round(confidence * 100) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* ── Page Layout type ─────────────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-3)',
                      letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Page Layout
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {Object.entries(LAYOUT_ICONS).map(([lt, { label, icon }]) => (
            <button key={lt} disabled={dis} onClick={() => handleLayoutChange(lt)}
              title={label}
              style={{
                flex: 1, padding: '6px 4px', borderRadius: 6,
                border: layoutType === lt ? '1.5px solid var(--color-cyan)' : '1.5px solid var(--color-border)',
                background: layoutType === lt ? 'rgba(0,229,255,0.12)' : 'var(--color-surface-3)',
                color: layoutType === lt ? 'var(--color-cyan)' : 'var(--color-text-3)',
                cursor: dis ? 'not-allowed' : 'pointer', fontSize: 15,
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
                transition: 'all 0.15s',
              }}>
              <span>{icon}</span>
              <span style={{ fontSize: 9, letterSpacing: '0.05em' }}>{label.toUpperCase()}</span>
            </button>
          ))}
        </div>

        {detecting && (
          <div style={{ fontSize: 11, color: 'var(--color-text-3)',
                        display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</span>
            Analysing…
          </div>
        )}
        {!detecting && layoutType && (
          <div style={{ fontSize: 11, color: 'var(--color-text-2)',
                        display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: 'var(--color-cyan)', fontWeight: 600 }}>Auto detected</span>
            {confPct != null && (
              <span style={{ background: confColor(confidence), color: '#fff',
                             padding: '1px 6px', borderRadius: 10, fontSize: 10 }}>
                {confPct}%
              </span>
            )}
          </div>
        )}
      </div>

      <div style={{ height: 1, background: 'var(--color-border)', opacity: 0.5 }} />

      {/* ── Split Line mode ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-3)',
                      letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Split Line
        </div>
        <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden',
                      border: '1.5px solid var(--color-border)' }}>
          {['auto', 'manual'].map((m) => (
            <button key={m} disabled={dis} onClick={() => handleModeChange(m)}
              style={{
                flex: 1, padding: '5px 0',
                background: splitMode === m ? 'rgba(0,229,255,0.15)' : 'var(--color-surface-3)',
                color: splitMode === m ? 'var(--color-cyan)' : 'var(--color-text-3)',
                border: 'none', cursor: dis ? 'not-allowed' : 'pointer',
                fontWeight: 600, fontSize: 11, letterSpacing: '0.06em',
                textTransform: 'uppercase', transition: 'all 0.15s',
              }}>
              {m === 'auto' ? 'Auto' : 'Manual'}
            </button>
          ))}
        </div>

        {splitMode === 'manual' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between',
                          fontSize: 10, color: 'var(--color-text-3)' }}>
              <span>Position</span>
              <span>{Math.round(splitX * 100)}%</span>
            </div>
            <input type="range" min={0} max={100}
              value={Math.round(splitX * 100)}
              onChange={(e) => handleSlider(e.target.value / 100)}
              disabled={dis}
              style={{ width: '100%', accentColor: 'var(--color-cyan)' }} />
          </div>
        )}
      </div>

      {/* ── Interaction hint ─────────────────────────────────────────────── */}
      <div style={{
        fontSize: 10, color: 'var(--color-text-3)', lineHeight: 1.6,
        background: 'var(--color-surface-3)', borderRadius: 6,
        padding: '7px 8px', border: '1px solid var(--color-border)',
      }}>
        🔵 Drag <strong>blue corners</strong> to adjust page boundary.<br />
        ◈ Drag <strong>purple diamonds</strong> to rotate the split line.<br />
        ↔ Drag the <strong>line body</strong> to translate it.<br />
        {layoutType === 'two_pages' && '↕ Click a region to select left or right page.'}
      </div>

      {/* ── Selected side indicator — always blue ──────────────────────────── */}
      {layoutType === 'two_pages' && selectedSide && (
        <div style={{
          padding: '6px 10px', borderRadius: 8, fontSize: 12, fontWeight: 600,
          background: 'rgba(30,100,255,0.14)',
          color: 'rgba(100,170,255,1)',
          border: '1.5px solid rgba(40,120,255,0.45)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span
            style={{
              width: 10, height: 10, borderRadius: 2,
              background: 'rgba(40,120,255,0.8)',
              display: 'inline-block', flexShrink: 0,
            }}
          />
          {selectedSide === 'left' ? 'Left page selected' : 'Right page selected'}
          <span style={{ marginLeft: 'auto', fontSize: 10, opacity: 0.65 }}>
            Click canvas to change
          </span>
        </div>
      )}

      {/* ── Page geometry info ───────────────────────────────────────────── */}
      {pageLayout && (
        <div style={{
          fontSize: 10, color: 'var(--color-text-3)', lineHeight: 1.5,
          background: 'rgba(140,80,255,0.06)', borderRadius: 6,
          padding: '6px 8px', border: '1px solid rgba(140,80,255,0.2)',
        }}>
          ✦ <strong style={{ color: 'rgba(180,120,255,1)' }}>Geometry-first</strong> mode active.<br />
          Split uses polygon clip + perspective warp (ScanTailor pipeline).
        </div>
      )}

      {splitMode === 'auto' && (
        <>
          <div style={{ height: 1, background: 'var(--color-border)', opacity: 0.5 }} />

          {/* ── Split Method ─────────────────────────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-3)',
                          letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              Split Method
            </div>

        <div style={{ display: 'flex', gap: 7 }}>

          {/* ── Option 1: Auto-Detect Spine ── */}
          <button
            disabled={dis || spineDetecting || detecting}
            onClick={runSpineDetect}
            title="Fast: projection-profile dip + Hough vote → pixel crop"
            style={{
              flex: 1, padding: '8px 6px', borderRadius: 8, border: 'none',
              background: activeMethod === 'spine'
                ? 'linear-gradient(135deg, #00c9a7 0%, #00bcd4 100%)'
                : 'var(--color-surface-3)',
              color: activeMethod === 'spine' ? '#fff' : 'var(--color-text-2)',
              cursor: dis ? 'not-allowed' : 'pointer',
              fontWeight: 700, fontSize: 11,
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
              boxShadow: activeMethod === 'spine' ? '0 2px 10px rgba(0,201,167,0.3)' : 'none',
              transition: 'all 0.2s', opacity: dis ? 0.5 : 1,
            }}>
            <span style={{ fontSize: 17 }}>{spineDetecting ? '⟳' : '🔵'}</span>
            <span>{spineDetecting ? 'Detecting…' : 'Auto-Detect Spine'}</span>
            <span style={{ fontSize: 9, opacity: 0.75, letterSpacing: '0.05em' }}>
              FAST · PIXEL CROP
            </span>
          </button>

          {/* ── Option 2: Geometry Split (ScanTailor) ── */}
          <button
            disabled={dis || detecting || spineDetecting}
            onClick={runDetect}
            title="Full ScanTailor two-pass: VertLineFinder + ContentSpanFinder + perspective warp"
            style={{
              flex: 1, padding: '8px 6px', borderRadius: 8, border: 'none',
              background: activeMethod === 'geometry'
                ? 'linear-gradient(135deg, #7c4dff 0%, #5c6bc0 100%)'
                : 'var(--color-surface-3)',
              color: activeMethod === 'geometry' ? '#fff' : 'var(--color-text-2)',
              cursor: dis ? 'not-allowed' : 'pointer',
              fontWeight: 700, fontSize: 11,
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
              boxShadow: activeMethod === 'geometry' ? '0 2px 10px rgba(124,77,255,0.3)' : 'none',
              transition: 'all 0.2s', opacity: dis ? 0.5 : 1,
            }}>
            <span style={{ fontSize: 17 }}>{detecting ? '⟳' : '✦'}</span>
            <span>{detecting ? 'Analysing…' : 'Geometry Split'}</span>
            <span style={{ fontSize: 9, opacity: 0.75, letterSpacing: '0.05em' }}>
              SCANTAILOR · WARP
            </span>
          </button>

        </div>

        {/* Status feedback row */}
        {activeMethod === 'spine' && spineResult && (
          <div style={{
            fontSize: 11, padding: '5px 9px', borderRadius: 6,
            display: 'flex', alignItems: 'center', gap: 6,
            background: spineResult.detected
              ? 'rgba(0,201,167,0.10)' : 'rgba(255,171,64,0.10)',
            border: `1px solid ${spineResult.detected ? 'rgba(0,201,167,0.3)' : 'rgba(255,171,64,0.3)'}`,
            color: spineResult.detected ? '#00c9a7' : 'var(--color-amber)',
          }}>
            {spineResult.detected ? (
              <>✓ Spine found at <strong>{Math.round(spineResult.spine_x_ratio * 100)}%</strong>
              &nbsp;— drag line to adjust</>
            ) : (
              <>⚠ No spine detected — try Geometry Split</>
            )}
          </div>
        )}

        {activeMethod === 'geometry' && !detecting && layoutType && (
          <div style={{ fontSize: 11, color: 'var(--color-text-2)',
                        display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: 'rgba(140,80,255,1)', fontWeight: 600 }}>✦ Geometry detected</span>
            {confPct != null && (
              <span style={{ background: confColor(confidence), color: '#fff',
                             padding: '1px 6px', borderRadius: 10, fontSize: 10 }}>
                {confPct}%
              </span>
            )}
          </div>
        )}
      </div>
        </>
      )}

      {/* ── Apply Split ───────────────────────────────────────────────────── */}
      <button disabled={dis || !layoutType} onClick={handleApply}
        style={{
          padding: '9px 12px', borderRadius: 8, fontSize: 13,
          background: dis || !layoutType
            ? 'var(--color-surface-3)'
            : 'linear-gradient(135deg, #0070f3 0%, #00bcd4 100%)',
          border: 'none',
          color: dis || !layoutType ? 'var(--color-text-3)' : '#fff',
          cursor: dis || !layoutType ? 'not-allowed' : 'pointer',
          fontWeight: 700,
          boxShadow: dis ? 'none' : '0 2px 12px rgba(0,180,255,0.25)',
          transition: 'all 0.15s',
        }}>
        ✂ Apply Split
      </button>

      {/* Apply to All — disabled for Split (inserting pages mid-loop breaks indices) */}
      <ApplyAllButton
        disabled
        pageCount={pageCount}
        title="Split inserts new pages into the list — apply per page manually to avoid conflicts"
        label={`✂ Apply Split to All ${pageCount} Pages (N/A)`}
      />
    </div>
  );
}

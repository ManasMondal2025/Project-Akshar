import { useState, useCallback, useEffect, useRef } from 'react';

/**
 * Deskew — ScanTailor-style panel (#3)
 *
 * – Replaces the old inline deskew section in Controls.jsx
 * – Auto tab: detect + show angle, Apply button → calls onDeskewAuto
 * – Manual tab: angle slider + numeric input → calls onAnglePreviewChange(deg)
 *   for live canvas arc preview; Apply → calls onDeskewManual(angle)
 *
 * Also exports DeskewArcOverlay — rendered by App.jsx on the canvas.
 */
export default function Deskew({
  angle = 0,
  onDeskewAuto,
  onDeskewManual,
  onModeChange,     // (mode: 'auto'|'manual'|null) → tells App to show/hide arc
  onAngleChange,    // (deg) → tells App the current preview angle
  loading,
  hasImage,
}) {
  const [tab, setTab] = useState('auto'); // 'auto' | 'manual'
  const [autoAngle, setAutoAngle] = useState(null);
  const [detecting, setDetecting] = useState(false);

  const dis = loading || !hasImage;

  // Tell App when mode changes (so it can show/hide arc overlay)
  useEffect(() => {
    onModeChange(tab === 'manual' ? 'manual' : 'auto');
  }, [tab, onModeChange]);

  const switchTab = (t) => {
    setTab(t);
    if (t === 'auto') onAngleChange(0);
  };

  const handleAutoDetect = useCallback(async () => {
    setDetecting(true);
    try {
      const res = await onDeskewAuto();
      if (res?.detected_angle != null) setAutoAngle(res.detected_angle);
    } finally {
      setDetecting(false);
    }
  }, [onDeskewAuto]);

  const handleApplyManual = useCallback(async () => {
    await onDeskewManual(angle);
  }, [angle, onDeskewManual]);

  return (
    <>
      {/* Auto / Manual tabs */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 12, borderRadius: 6, overflow: 'hidden',
        border: '1px solid var(--color-border)' }}>
        {['auto', 'manual'].map(t => (
          <button key={t} onClick={() => switchTab(t)}
            style={{
              flex: 1, padding: '5px 0', fontSize: 12, fontWeight: 600, border: 'none',
              background: tab === t ? 'rgba(255,171,64,0.18)' : 'var(--color-surface-3)',
              color: tab === t ? 'var(--color-amber)' : 'var(--color-text-3)',
              cursor: 'pointer', textTransform: 'capitalize',
            }}
          >{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Auto tab ── */}
      {tab === 'auto' && (
        <>
          {autoAngle != null && (
            <div style={{ textAlign: 'center', marginBottom: 10 }}>
              <span style={{
                fontSize: 22, fontWeight: 700,
                color: Math.abs(autoAngle) > 0.1 ? 'var(--color-amber)' : 'var(--color-green)',
              }}>
                {autoAngle > 0 ? '+' : ''}{autoAngle.toFixed(2)}°
              </span>
              <div style={{ fontSize: 10, color: 'var(--color-text-3)', marginTop: 2 }}>
                detected skew angle
              </div>
            </div>
          )}
          <button
            className="btn w-full"
            onClick={handleAutoDetect}
            disabled={dis || detecting}
            id="btn-deskew-auto-detect"
            style={{
              background: 'linear-gradient(135deg,#ffab40,#e08030)', color: '#07090f',
              boxShadow: !dis ? '0 0 18px rgba(255,171,64,0.3)' : 'none',
              marginBottom: 6,
            }}
          >
            {detecting ? <span className="spinner" /> : '⟳'} Auto-Detect & Correct
          </button>
        </>
      )}

      {/* ── Manual tab ── */}
      {tab === 'manual' && (
        <>
          {/* Angle display + spin controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <div style={{
              flex: 1, background: 'var(--color-surface-3)',
              border: '1px solid var(--color-border)', borderRadius: 6,
              padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 4,
            }}>
              <input
                type="number" step="0.1" min="-45" max="45"
                value={Number(angle).toFixed(2)}
                onChange={e => onAngleChange(parseFloat(e.target.value) || 0)}
                style={{
                  flex: 1, background: 'none', border: 'none', outline: 'none',
                  color: 'var(--color-text)', fontSize: 14, fontWeight: 600,
                }}
                id="deskew-angle-input"
              />
              <span style={{ color: 'var(--color-text-3)', fontSize: 12 }}>°</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <button onClick={() => onAngleChange(Math.min(45, angle + 0.1))}
                style={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-border)',
                  borderRadius: 4, padding: '2px 5px', cursor: 'pointer', color: 'var(--color-text)' }}>▲</button>
              <button onClick={() => onAngleChange(Math.max(-45, angle - 0.1))}
                style={{ background: 'var(--color-surface-3)', border: '1px solid var(--color-border)',
                  borderRadius: 4, padding: '2px 5px', cursor: 'pointer', color: 'var(--color-text)' }}>▼</button>
            </div>
          </div>

          {/* Slider */}
          <div className="slider-row" style={{ marginBottom: 12 }}>
            <input type="range" className="slider" min={-45} max={45} step={0.1}
              value={angle}
              onChange={e => onAngleChange(parseFloat(e.target.value))} />
            <span className="slider-value" style={{ color: 'var(--color-amber)', minWidth: 44 }}>
              {angle > 0 ? '+' : ''}{Number(angle).toFixed(1)}°
            </span>
          </div>

          <div style={{ fontSize: 10, color: 'var(--color-text-3)', marginBottom: 10, textAlign: 'center' }}>
            Use Ctrl+Wheel on canvas to rotate, or drag arc handles
          </div>

          <button
            className="btn w-full"
            onClick={handleApplyManual}
            disabled={dis || angle === 0}
            id="btn-deskew-manual-apply"
            style={{
              background: !dis && angle !== 0 ? 'linear-gradient(135deg,#ffab40,#e08030)' : undefined,
              color: '#07090f',
              boxShadow: !dis && angle !== 0 ? '0 0 18px rgba(255,171,64,0.3)' : 'none',
              marginBottom: 6,
            }}
          >
            ⟳ Apply Manual Deskew
          </button>

          <button
            className="btn btn-ghost btn-sm w-full"
            onClick={() => onAngleChange(0)}
            id="btn-deskew-reset"
          >Reset to 0°</button>
        </>
      )}

      {/* Apply To... (visible in both tabs) */}
      <div style={{ marginTop: 6 }}>
        <button
          className="btn btn-ghost btn-sm w-full"
          disabled={dis}
          id="btn-deskew-apply-to"
          style={{ color: 'var(--color-text-2)', border: '1px solid var(--color-border)' }}
        >Apply To…</button>
      </div>
    </>
  );
}

/* ─── Canvas Arc Overlay — exported for App.jsx ──────────────────────────── */

export function DeskewArcOverlay({ angle, onAngleChange }) {
  const [dragging, setDragging] = useState(null); // 'left' | 'right' | null
  const startRef = useRef({ y: 0, angle: 0 });

  const onHandleDown = (e, side) => {
    e.preventDefault();
    startRef.current = { y: e.clientY, angle };
    setDragging(side);
  };

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging) return;
      const deltaY = e.clientY - startRef.current.y;
      // Right handle drag up (deltaY < 0) → negative angle (CCW)
      // Left handle drag up (deltaY < 0) → positive angle (CW)
      const sign = dragging === 'right' ? 1 : -1;
      const newAngle = Math.max(-45, Math.min(45,
        startRef.current.angle + sign * deltaY * 0.25));
      onAngleChange(newAngle);
    };
    const onUp = () => setDragging(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging, onAngleChange]);

  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 12, pointerEvents: 'none' }}>
      {/* Grid overlay */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: `
          repeating-linear-gradient(0deg, rgba(60,140,255,0.12) 0 1px, transparent 1px 50px),
          repeating-linear-gradient(90deg, rgba(60,140,255,0.12) 0 1px, transparent 1px 50px)
        `,
      }} />

      {/* SVG arc with draggable handles */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
        style={{
          position: 'absolute', inset: 0,
          width: '100%', height: '100%',
          cursor: dragging ? 'ns-resize' : 'default',
        }}
      >
        {/* Rotate group by current angle around centre */}
        <g transform={`rotate(${angle}, 50, 50)`}>
          {/* Ellipse outline */}
          <ellipse cx="50" cy="50" rx="40" ry="28"
            fill="none" stroke="#3c8cff" strokeWidth="0.45" opacity="0.85" />

          {/* Horizontal diameter */}
          <line x1="10" y1="50" x2="90" y2="50"
            stroke="#3c8cff" strokeWidth="0.3" strokeDasharray="2,2" opacity="0.7" />

          {/* Vertical diameter */}
          <line x1="50" y1="22" x2="50" y2="78"
            stroke="#3c8cff" strokeWidth="0.3" strokeDasharray="2,2" opacity="0.7" />

          {/* Left handle */}
          <circle
            cx="10" cy="50" r="2.2"
            fill="#3c8cff" stroke="#fff" strokeWidth="0.6"
            style={{ cursor: 'ns-resize', pointerEvents: 'all' }}
            onMouseDown={(e) => onHandleDown(e, 'left')}
          />

          {/* Right handle */}
          <circle
            cx="90" cy="50" r="2.2"
            fill="#3c8cff" stroke="#fff" strokeWidth="0.6"
            style={{ cursor: 'ns-resize', pointerEvents: 'all' }}
            onMouseDown={(e) => onHandleDown(e, 'right')}
          />
        </g>

        {/* Angle label (always upright) */}
        <text x="50" y="96" textAnchor="middle"
          fill="#3c8cff" fontSize="4" fontFamily="monospace" opacity="0.9">
          {angle > 0 ? '+' : ''}{angle.toFixed(2)}°
        </text>
      </svg>
    </div>
  );
}

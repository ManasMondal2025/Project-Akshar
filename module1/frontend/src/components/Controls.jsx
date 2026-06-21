import { useState } from 'react';
import StPanel from './StPanel';
import ApplyAllButton from './ApplyAllButton';

/**
 * Controls — Sidebar control panel for the Image Workbench.
 * Panels 5-8 of the ScanTailor workflow.
 */
export default function Controls({
  onTransform, onDewarp, onDewarpAuto, onDewarpPoly,
  onEnhance, onReset, onUndo, onExportPdf,
  onAnalyzeGrid, onEstimatePolyCurves,
  onDewarpAutoAll, onDewarpPolyAll, onEnhanceAll,
  loading, batchLoading, batchProgress,
  hasImage, hasTransformed, canUndo, pageCount,
  dewarpGridActive = false,
  polyGridActive = false,
  perspectiveOpen, onPerspectiveToggle,
  dewarpOpen, onDewarpToggle,
  dewarpMode = 'manual', onDewarpModeChange,
}) {
  const [outputFormat, setOutputFormat] = useState('bw');

  const disabled = loading || batchLoading || !hasImage;

  return (
    <div className="st-panels-block">

      {/* ── Panel 5: Perspective Transform ── */}
      <StPanel number={5} title="Perspective Correct" colorClass="panel-cyan" open={perspectiveOpen} onToggle={onPerspectiveToggle}>
        <p style={{ fontSize: 11, color: 'var(--color-text-3)', lineHeight: 1.55, marginBottom: 10 }}>
          Drag the corner handles on the canvas, then apply.
        </p>
        <button
          className="btn w-full"
          onClick={onTransform}
          disabled={disabled}
          id="btn-transform"
          style={{ background: 'linear-gradient(135deg,#7c4dff,#5b37c0)', color: '#fff',
            boxShadow: disabled ? 'none' : '0 0 18px rgba(124,77,255,0.3)' }}
        >
          {loading ? <span className="spinner" /> : '⊞'} Apply Transform
        </button>
      </StPanel>

      {/* ── Panel 6: Grid Dewarp ── */}
      <StPanel number={6} title="Grid Dewarp" colorClass="panel-blue" open={dewarpOpen} onToggle={onDewarpToggle}>

        {/* ── Mode toggle: Auto / Manual / Poly ── */}
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-3)',
                        letterSpacing: '0.07em', textTransform: 'uppercase', marginBottom: 6 }}>
            Mode
          </div>
          <div style={{ display: 'flex', borderRadius: 7, overflow: 'hidden',
                        border: '1.5px solid var(--color-border)' }}>
            {['auto', 'manual', 'poly'].map((m) => (
              <button key={m} disabled={disabled} onClick={() => onDewarpModeChange?.(m)}
                style={{
                  flex: 1, padding: '5px 0',
                  background: dewarpMode === m ? 'rgba(0,229,255,0.15)' : 'var(--color-surface-3)',
                  color: dewarpMode === m ? 'var(--color-cyan)' : 'var(--color-text-3)',
                  border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
                  fontWeight: 700, fontSize: 11, letterSpacing: '0.06em',
                  textTransform: 'uppercase', transition: 'all 0.15s',
                }}>
                {m === 'auto' ? '✦ Auto' : m === 'manual' ? '⊞ Manual' : '〜 Poly'}
              </button>
            ))}
          </div>
        </div>

        {/* ── AUTO mode ── */}
        {dewarpMode === 'auto' && (
          <>
            <div style={{
              fontSize: 10, color: 'var(--color-text-3)', lineHeight: 1.6,
              background: 'rgba(0,229,255,0.05)', borderRadius: 6,
              padding: '7px 8px', border: '1px solid rgba(0,229,255,0.18)',
              marginBottom: 10,
            }}>
              ✦ <strong style={{ color: 'var(--color-cyan)' }}>Neural network mode</strong><br />
              ICCV 2023 model automatically detects and corrects document warping.<br />
              No grid interaction required.
            </div>
            <button
              className="btn w-full"
              onClick={onDewarpAuto}
              disabled={disabled}
              id="btn-dewarp-auto"
              style={{
                background: disabled
                  ? 'var(--color-surface-3)'
                  : 'linear-gradient(135deg,#00e5ff,#7c4dff)',
                color: disabled ? 'var(--color-text-3)' : '#fff',
                boxShadow: disabled ? 'none' : '0 0 18px rgba(0,229,255,0.3)',
              }}
            >
              {loading ? <span className="spinner" /> : '✦'} Apply Auto Dewarp
            </button>
            <ApplyAllButton
              onClick={onDewarpAutoAll}
              pageCount={pageCount}
              disabled={disabled}
              loading={batchLoading}
              progress={batchProgress}
              label={`✦ Auto-Dewarp All ${pageCount} Pages`}
            />
          </>
        )}

        {/* ── MANUAL mode ── */}
        {dewarpMode === 'manual' && (
          <>
            {/* Analyze grid toggle */}
            <button
              className="btn w-full btn-sm"
              onClick={onAnalyzeGrid}
              disabled={disabled}
              id="btn-analyze-grid"
              style={{
                marginBottom: 10,
                background: dewarpGridActive ? 'rgba(60,140,255,0.18)' : 'var(--color-surface-3)',
                color: dewarpGridActive ? '#5b9fff' : 'var(--color-text-2)',
                border: `1px solid ${dewarpGridActive ? 'rgba(60,140,255,0.45)' : 'var(--color-border)'}`,
                boxShadow: dewarpGridActive ? '0 0 12px rgba(60,140,255,0.2)' : 'none',
              }}
            >
              {dewarpGridActive ? '🔵 Re-Analyze Grid' : '🔵 Analyze & Show Grid'}
            </button>

            <button
              className="btn w-full"
              onClick={() => onDewarp(1.0)}
              disabled={disabled || !dewarpGridActive}
              id="btn-dewarp"
              style={{
                background: 'linear-gradient(135deg,#00e5ff,#00b4cc)', color: '#07090f',
                boxShadow: (disabled || !dewarpGridActive) ? 'none' : '0 0 18px rgba(0,229,255,0.3)',
              }}
            >
              {loading ? <span className="spinner" /> : '⌇'} Apply Grid Dewarp
            </button>
            {/* Manual dewarp is per-page (needs per-image grid analysis) */}
            <ApplyAllButton
              disabled
              pageCount={pageCount}
              title="Manual grid dewarp requires per-page grid analysis — use Auto Dewarp mode for batch processing"
              label={`⌇ Grid Dewarp All ${pageCount} Pages (N/A)`}
            />
          </>
        )}

        {/* ── POLY mode ── */}
        {dewarpMode === 'poly' && (
          <>
            <div style={{
              fontSize: 10, color: 'var(--color-text-3)', lineHeight: 1.6,
              background: 'rgba(124,77,255,0.07)', borderRadius: 6,
              padding: '7px 8px', border: '1px solid rgba(124,77,255,0.22)',
              marginBottom: 10,
            }}>
              〜 <strong style={{ color: '#b39ddb' }}>B-spline curve mode</strong><br />
              Estimates 4 text-density curves and applies a spline-based deformation.<br />
              No model required — works on all document images.
            </div>

            {/* Analyze & show curves toggle */}
            <button
              className="btn w-full btn-sm"
              onClick={onEstimatePolyCurves}
              disabled={disabled}
              id="btn-estimate-poly-curves"
              style={{
                marginBottom: 10,
                background: polyGridActive ? 'rgba(124,77,255,0.18)' : 'var(--color-surface-3)',
                color: polyGridActive ? '#b39ddb' : 'var(--color-text-2)',
                border: `1px solid ${polyGridActive ? 'rgba(124,77,255,0.45)' : 'var(--color-border)'}`,
                boxShadow: polyGridActive ? '0 0 12px rgba(124,77,255,0.2)' : 'none',
              }}
            >
              {polyGridActive ? '〜 Re-Analyze Curves' : '〜 Analyze & Show Curves'}
            </button>

            <button
              className="btn w-full"
              onClick={onDewarpPoly}
              disabled={disabled}
              id="btn-dewarp-poly"
              style={{
                background: disabled
                  ? 'var(--color-surface-3)'
                  : 'linear-gradient(135deg,#7c4dff,#b39ddb)',
                color: disabled ? 'var(--color-text-3)' : '#fff',
                boxShadow: disabled ? 'none' : '0 0 18px rgba(124,77,255,0.3)',
                marginBottom: 8,
              }}
            >
              {loading ? <span className="spinner" /> : '〜'} Apply Poly Dewarp
            </button>
            <ApplyAllButton
              onClick={onDewarpPolyAll}
              pageCount={pageCount}
              disabled={disabled}
              loading={batchLoading}
              label="⌇ Poly Dewarp All Pages"
              title="Runs automatic B-spline poly dewarp on all pages in the batch."
            />
          </>
        )}
      </StPanel>


      {/* ── Panel 7: Enhancement ── */}
      <StPanel number={7} title="Enhancement" colorClass="panel-purple">
        <div className="field-label" style={{ marginBottom: 8 }}>Output Mode</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 14 }}>
          {[
            { id: 'bw',    label: 'B/W',   icon: '◧', desc: 'Black & white' },
            { id: 'color', label: 'Color', icon: '▦', desc: 'Full color' },
          ].map(f => (
            <label key={f.id} style={{ cursor: 'pointer' }}>
              <input type="radio" name="output-format" value={f.id} style={{ display: 'none' }}
                checked={outputFormat === f.id} onChange={() => setOutputFormat(f.id)} />
              <div style={{
                padding: '10px', borderRadius: 8, cursor: 'pointer', textAlign: 'center',
                border: `1px solid ${outputFormat === f.id ? 'rgba(0,230,118,0.5)' : 'var(--color-border)'}`,
                background: outputFormat === f.id ? 'rgba(0,230,118,0.1)' : 'var(--color-surface-3)',
                transition: 'all 0.15s',
              }}>
                <div style={{ fontSize: 20, marginBottom: 4 }}>{f.icon}</div>
                <div style={{ fontSize: 12, fontWeight: 700,
                  color: outputFormat === f.id ? 'var(--color-green)' : 'var(--color-text)' }}>
                  {f.label}
                </div>
                <div style={{ fontSize: 10, color: 'var(--color-text-3)', marginTop: 2 }}>{f.desc}</div>
              </div>
            </label>
          ))}
        </div>

        <button
          className="btn w-full"
          onClick={() => onEnhance('otsu', outputFormat)}
          disabled={disabled}
          id="btn-enhance"
          style={{ background: 'linear-gradient(135deg,#00e676,#00b852)', color: '#07090f',
            boxShadow: disabled ? 'none' : '0 0 18px rgba(0,230,118,0.3)' }}
        >
          {loading ? <span className="spinner" /> : '✦'} Apply Enhancement
        </button>
        <ApplyAllButton
          onClick={() => onEnhanceAll?.('otsu', outputFormat)}
          pageCount={pageCount}
          disabled={disabled}
          loading={batchLoading}
          progress={batchProgress}
          label={`✦ Enhance All ${pageCount} Pages`}
        />
      </StPanel>

      {/* ── Actions (always visible) ────────────────────────────── */}
      <div className="st-panel open" style={{ borderLeft: '2px solid var(--color-text-3)' }}>
        <div className="st-panel-header open" style={{ cursor: 'default', opacity: 0.85 }}>
          <span className="st-panel-num">⚙</span>
          <span className="st-panel-title">Actions</span>
        </div>
        <div className="st-panel-body">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            <button className="btn btn-ghost btn-sm" onClick={onUndo}
              disabled={!canUndo || loading} id="btn-undo">↩ Undo</button>
            <button className="btn btn-ghost btn-sm" onClick={onReset}
              disabled={!hasTransformed || loading} id="btn-reset">↺ Reset</button>
          </div>
          <button
            className="btn btn-secondary w-full"
            onClick={onExportPdf}
            disabled={!hasImage || loading}
            id="btn-export"
          >
            ⬇ Export PDF ({pageCount} page{pageCount !== 1 ? 's' : ''})
          </button>
        </div>
      </div>
    </div>
  );
}

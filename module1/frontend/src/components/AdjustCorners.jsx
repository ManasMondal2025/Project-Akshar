import { useState, useCallback } from 'react';
import ApplyAllButton from './ApplyAllButton';

/**
 * AdjustCorners — ScanTailor-style panel (panel #2)
 * Auto-detects document corners, lets the user drag them on the canvas,
 * then applies perspective warp to crop/straighten the document.
 *
 * method / onMethodChange — controls which backend algorithm is used:
 *   'classical'  → existing edge-detection pipeline (default)
 *   'cnn'        → OptimizedMobileNetwork CNN (from akshar_ai)
 */
export default function AdjustCorners({
  onDetect,
  onApply,
  onApplyAll,
  loading,
  batchLoading,
  batchProgress,
  hasImage,
  pageCount = 1,
  cornersActive,          // boolean — are corners currently shown on canvas?
  onCornersActiveChange,  // (bool) => void — toggle corner overlay visibility
  method = 'classical',   // 'classical' | 'cnn'
  onMethodChange,         // (string) => void
}) {
  const [detected, setDetected] = useState(false);

  const handleDetect = useCallback(async () => {
    const result = await onDetect?.();
    if (result) {
      setDetected(true);
      onCornersActiveChange?.(true);
    }
  }, [onDetect, onCornersActiveChange]);

  const handleApply = useCallback(async () => {
    await onApply?.();
    setDetected(false);
    onCornersActiveChange?.(false);
  }, [onApply, onCornersActiveChange]);

  const handleApplyAll = useCallback(async () => {
    await onApplyAll?.();
    setDetected(false);
    onCornersActiveChange?.(false);
  }, [onApplyAll, onCornersActiveChange]);

  const dis = loading || batchLoading || !hasImage;

  return (
    <>
      <div style={{ textAlign: 'center' }}>
        <div className="field-label" style={{ marginBottom: 10 }}>Corner Detection</div>

        <p style={{
          fontSize: 11, color: 'var(--color-text-3)',
          margin: '0 0 12px', lineHeight: 1.5,
        }}>
          Auto-detect document edges, then drag corners to fine-tune before applying perspective correction.
        </p>

        {/* ── Method Selector ───────────────────────────────────────────── */}
        <div style={{ marginBottom: 12 }}>
          <div style={{
            fontSize: 10,
            color: 'var(--color-text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: 6,
            fontWeight: 600,
          }}>
            Detection Method
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
            {/* Classical Method button */}
            <button
              id="btn-method-classical"
              className="btn btn-sm"
              onClick={() => onMethodChange?.('classical')}
              disabled={dis}
              style={{
                background: method === 'classical'
                  ? 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%)'
                  : 'var(--color-surface-3)',
                color: method === 'classical' ? '#fff' : 'var(--color-text-2)',
                border: method === 'classical'
                  ? 'none'
                  : '1px solid var(--color-border)',
                fontSize: 10,
                padding: '5px 4px',
                fontWeight: method === 'classical' ? 700 : 400,
                transition: 'all 0.15s ease',
              }}
            >
              Classical Method
            </button>

            {/* CNN Model button */}
            <button
              id="btn-method-cnn"
              className="btn btn-sm"
              onClick={() => onMethodChange?.('cnn')}
              disabled={dis}
              style={{
                background: method === 'cnn'
                  ? 'linear-gradient(135deg, #ff6d00 0%, #ff00ea 100%)'
                  : 'var(--color-surface-3)',
                color: method === 'cnn' ? '#fff' : 'var(--color-text-2)',
                border: method === 'cnn'
                  ? 'none'
                  : '1px solid var(--color-border)',
                fontSize: 10,
                padding: '5px 4px',
                fontWeight: method === 'cnn' ? 700 : 400,
                transition: 'all 0.15s ease',
              }}
            >
              Using CNN Model
            </button>
          </div>
        </div>
        {/* ── End Method Selector ───────────────────────────────────────── */}

        {/* Detect button */}
        <button
          className="btn btn-sm"
          onClick={handleDetect}
          disabled={dis}
          id="btn-corners-detect"
          style={{
            width: '100%',
            marginBottom: 8,
            background: cornersActive
              ? 'var(--color-surface-3)'
              : 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%)',
            color: cornersActive ? 'var(--color-text-2)' : '#fff',
            border: cornersActive ? '1px solid var(--color-cyan)' : 'none',
          }}
        >
          {loading ? (
            <span className="spinner" style={{ width: 14, height: 14, borderWidth: 1.5, marginRight: 6 }} />
          ) : null}
          {cornersActive ? '⟳ Re-detect Corners' : '⊞ Detect Corners'}
        </button>

        {/* Corner info badge */}
        {cornersActive && (
          <div style={{
            fontSize: 11, color: 'var(--color-cyan)',
            marginBottom: 10, fontWeight: 600,
          }}>
            ✓ Corners detected — drag to adjust
          </div>
        )}

        {/* Apply / Hide row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 4 }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              onCornersActiveChange?.(false);
              setDetected(false);
            }}
            disabled={dis || !cornersActive}
            id="btn-corners-hide"
          >✕ Hide</button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleApply}
            disabled={dis || !cornersActive}
            id="btn-corners-apply"
            style={{
              background: !cornersActive ? 'var(--color-surface-3)' : undefined,
              color: !cornersActive ? 'var(--color-text-3)' : undefined,
            }}
          >
            ✓ Apply
          </button>
        </div>
      </div>

      {/* Apply to All Pages */}
      <ApplyAllButton
        onClick={handleApplyAll}
        pageCount={pageCount}
        disabled={dis || !cornersActive}
        loading={batchLoading}
        progress={batchProgress}
        title={!cornersActive
          ? 'Detect corners first, then apply to all'
          : `Detect & apply corners to all ${pageCount} pages`}
        label={`⊞ Detect & Apply to All ${pageCount} Pages`}
      />
    </>
  );
}

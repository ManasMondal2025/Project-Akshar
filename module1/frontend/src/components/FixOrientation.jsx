import { useState, useCallback } from 'react';
import ApplyAllButton from './ApplyAllButton';

/**
 * FixOrientation — ScanTailor-style panel (panel #1)
 * Three buttons: CCW, CW, and a visual upright indicator.
 * Tracks accumulated rotation locally; sends one /orient/rotate call per click.
 */
export default function FixOrientation({
  onRotate, onRotateAll,
  loading, batchLoading, batchProgress,
  hasImage, pageCount = 1,
}) {
  const [totalRot, setTotalRot] = useState(0); // 0 | 90 | 180 | 270
  const [lastDelta, setLastDelta] = useState(null); // last angle clicked

  const rotate = useCallback(async (delta) => {
    await onRotate(delta);
    setTotalRot(prev => ((prev + delta) % 360 + 360) % 360);
    setLastDelta(delta);
  }, [onRotate]);

  const handleReset = useCallback(async () => {
    if (totalRot === 0) return;
    const back = (360 - totalRot) % 360;
    if (back !== 0) await onRotate(back > 180 ? back - 360 : back);
    setTotalRot(0);
    setLastDelta(null);
  }, [totalRot, onRotate]);

  const dis = loading || batchLoading || !hasImage;
  const rotLabel = totalRot === 0 ? null
    : totalRot === 90  ? '90° CW'
    : totalRot === 180 ? '180°'
    : '90° CCW';

  const deltaLabel = lastDelta === null ? null
    : lastDelta === 90 ? '90° CW'
    : lastDelta === -90 ? '90° CCW'
    : '180°';

  return (
    <>
      <div style={{ textAlign: 'center' }}>
        <div className="field-label" style={{ marginBottom: 10 }}>Rotate</div>

        {/* Three orientation buttons */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: 10, marginBottom: 8 }}>
          <button
            className="orient-btn"
            onClick={() => rotate(-90)}
            disabled={dis}
            title="Rotate 90° Counter-clockwise"
            id="btn-orient-ccw"
          >↺</button>

          <button
            className="orient-btn orient-btn-up"
            disabled
            title="Current upright"
            style={{ cursor: 'default', opacity: 0.6 }}
          >↑</button>

          <button
            className="orient-btn"
            onClick={() => rotate(90)}
            disabled={dis}
            title="Rotate 90° Clockwise"
            id="btn-orient-cw"
          >↻</button>
        </div>

        {/* Current rotation badge */}
        {rotLabel && (
          <div style={{
            fontSize: 11, color: 'var(--color-cyan)',
            marginBottom: 8, fontWeight: 600,
          }}>
            Current: {rotLabel}
          </div>
        )}

        {/* Buttons row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 4 }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleReset}
            disabled={dis || totalRot === 0}
            id="btn-orient-reset"
          >↺ Reset</button>
          <button
            className="btn btn-sm"
            disabled
            id="btn-orient-apply"
            style={{ background: 'var(--color-surface-3)', color: 'var(--color-text-3)',
              border: '1px solid var(--color-border)', cursor: 'default' }}
          >Single Page</button>
        </div>
      </div>

      {/* Apply to All Pages */}
      <ApplyAllButton
        onClick={() => lastDelta !== null && onRotateAll?.(lastDelta)}
        pageCount={pageCount}
        disabled={dis || lastDelta === null}
        loading={batchLoading}
        progress={batchProgress}
        title={lastDelta === null
          ? 'Click a rotate button first, then apply to all'
          : `Apply ${deltaLabel} to all ${pageCount} pages`}
        label={lastDelta !== null
          ? `⟳ Apply ${deltaLabel} to All ${pageCount} Pages`
          : `⟳ Apply to All ${pageCount} Pages`}
      />
    </>
  );
}

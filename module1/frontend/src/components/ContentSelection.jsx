import { useState, useCallback } from 'react';
import ApplyAllButton from './ApplyAllButton';

/**
 * ContentSelection — ScanTailor-style panel (#4)
 *
 * Auto-detects the content boundary (text/graphics area) and crops the image
 * to remove empty borders and noise outside the content region.
 *
 * Uses the ScanTailor Advanced Select Content algorithm:
 *   Wolf binarization → shadow/garbage detection → despeckle →
 *   maximum whitespace content-block construction → text-mask estimation →
 *   edge trimming.
 */
export default function ContentSelection({
  onDetect, onApply, onApplyAll, onClearPreview,
  loading, batchLoading, batchProgress,
  hasImage, pageCount = 1,
}) {
  const [detecting, setDetecting] = useState(false);
  const [contentBox, setContentBox] = useState(null);  // { x, y, width, height }
  const [imgSize, setImgSize] = useState(null);        // { width, height }

  const dis = loading || batchLoading || !hasImage;

  const handleDetect = useCallback(async () => {
    setDetecting(true);
    try {
      const res = await onDetect();
      if (res) {
        setContentBox(res.content_box || null);
        setImgSize(res.image_width && res.image_height
          ? { width: res.image_width, height: res.image_height }
          : null);
      }
    } finally { setDetecting(false); }
  }, [onDetect]);

  const handleApply = useCallback(async () => {
    await onApply();
    // Clear detection result after apply (image changed)
    setContentBox(null);
    setImgSize(null);
  }, [onApply]);

  const handleClear = useCallback(() => {
    setContentBox(null);
    setImgSize(null);
    onClearPreview?.();
  }, [onClearPreview]);

  return (
    <>
      {/* Description */}
      <div style={{
        fontSize: 10, color: 'var(--color-text-3)', lineHeight: 1.6,
        background: 'rgba(0,200,200,0.05)', borderRadius: 6,
        padding: '7px 8px', border: '1px solid rgba(0,200,200,0.18)',
        marginBottom: 10,
      }}>
        ✦ <strong style={{ color: 'var(--color-cyan)' }}>ScanTailor Select Content</strong><br />
        Automatically detects the content boundary using Wolf binarization
        and maximum whitespace analysis, then crops to the content area.
      </div>

      {/* Detect Button */}
      <button
        className="btn w-full"
        onClick={handleDetect}
        disabled={dis || detecting}
        id="btn-content-detect"
        style={{
          background: !dis ? 'linear-gradient(135deg,#26c6da,#00acc1)' : undefined,
          color: !dis ? '#07090f' : 'var(--color-text-3)',
          boxShadow: !dis ? '0 0 18px rgba(38,198,218,0.3)' : 'none',
          marginBottom: 8,
        }}
      >
        {detecting ? <span className="spinner" /> : '◫'} Detect Content Box
      </button>

      {/* Detection result */}
      {contentBox && imgSize && (
        <div style={{
          background: 'var(--color-surface-3)',
          border: '1px solid var(--color-border)',
          borderRadius: 6, padding: '8px 10px',
          marginBottom: 10,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-cyan)', marginBottom: 6 }}>
            Content Box Detected
          </div>
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4,
            fontSize: 10, color: 'var(--color-text-2)',
          }}>
            <span>X: {contentBox.x}px</span>
            <span>Y: {contentBox.y}px</span>
            <span>Width: {contentBox.width}px</span>
            <span>Height: {contentBox.height}px</span>
          </div>
          <div style={{
            fontSize: 10, color: 'var(--color-text-3)', marginTop: 4,
            borderTop: '1px solid var(--color-border)', paddingTop: 4,
          }}>
            Image: {imgSize.width} × {imgSize.height}px
          </div>
        </div>
      )}

      {contentBox === null && !detecting && imgSize && (
        <div style={{
          fontSize: 11, color: 'var(--color-text-3)', textAlign: 'center',
          marginBottom: 10,
        }}>
          No content box detected
        </div>
      )}

      {/* Actions Row */}
      <div style={{ display: 'grid', gridTemplateColumns: contentBox ? '1fr 1fr' : '1fr', gap: 6, marginBottom: 6 }}>
        {contentBox && (
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleClear}
            disabled={dis}
            id="btn-content-clear"
          >
            ✕ Discard
          </button>
        )}
        <button
          className="btn btn-sm"
          onClick={handleApply}
          disabled={dis}
          id="btn-content-apply"
          style={{
            background: !dis ? 'linear-gradient(135deg,#00bcd4,#0097a7)' : undefined,
            color: !dis ? '#fff' : 'var(--color-text-3)',
            boxShadow: !dis ? '0 0 18px rgba(0,188,212,0.3)' : 'none',
          }}
        >
          {loading ? <span className="spinner" /> : '⊡ Apply'}
        </button>
      </div>

      {/* Apply to All Pages */}
      <ApplyAllButton
        onClick={() => onApplyAll?.()}
        pageCount={pageCount}
        disabled={dis}
        loading={batchLoading}
        progress={batchProgress}
        label={`⊡ Select Content on All ${pageCount} Pages`}
      />
    </>
  );
}

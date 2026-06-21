/**
 * Preview — Before/after image comparison strip at the bottom of the canvas.
 */
export default function Preview({ originalSrc, processedSrc, activeFilter }) {
  if (!originalSrc) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      padding: '10px 18px',
      width: '100%',
      background: 'var(--color-surface)',
      borderTop: '1px solid var(--color-border)',
      flexShrink: 0,
    }}>
      {/* Label */}
      <span style={{
        fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: '0.06em', color: 'var(--color-text-3)', whiteSpace: 'nowrap'
      }}>
        Compare
      </span>

      {/* Original thumb */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-3)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Original
        </span>
        <img
          src={originalSrc}
          alt="Original"
          id="preview-original"
          style={{
            width: 260, height: 200, objectFit: 'contain',
            borderRadius: 6, border: '1px solid var(--color-border)',
            background: 'var(--color-bg)',
          }}
        />
      </div>

      {/* Arrow */}
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
        stroke="var(--color-text-3)" strokeWidth="2" style={{ flexShrink: 0 }}>
        <polyline points="9 18 15 12 9 6" />
      </svg>

      {/* Processed thumb */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
        <span style={{
          fontSize: 10,
          color: processedSrc ? 'var(--color-cyan)' : 'var(--color-text-3)',
          textTransform: 'uppercase', letterSpacing: '0.04em',
          maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
        }}>
          {activeFilter || 'Processed'}
        </span>
        {processedSrc ? (
          <img
            src={processedSrc}
            alt="Processed"
            id="preview-processed"
            style={{
              width: 260, height: 200, objectFit: 'contain',
              borderRadius: 6,
              border: '1px solid rgba(0,229,255,0.4)',
              background: 'var(--color-bg)',
              boxShadow: '0 0 8px rgba(0,229,255,0.2)',
            }}
          />
        ) : (
          <div style={{
            width: 260, height: 200, borderRadius: 6,
            border: '1px dashed var(--color-border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, color: 'var(--color-text-3)',
          }}>
            No edits yet
          </div>
        )}
      </div>
    </div>
  );
}

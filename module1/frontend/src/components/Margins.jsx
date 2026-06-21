import { useState, useCallback } from 'react';
import ApplyAllButton from './ApplyAllButton';

export default function Margins({
  onDetect, onApply, onApplyAll,
  loading, batchLoading, batchProgress,
  hasImage, pageCount = 1,
}) {
  const [autoMargins, setAutoMargins] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [top,    setTop]    = useState(5.0);
  const [bottom, setBottom] = useState(5.0);
  const [left,   setLeft]   = useState(10.0);
  const [right,  setRight]  = useState(10.0);
  const [linkTB, setLinkTB] = useState(false);
  const [linkLR, setLinkLR] = useState(true);
  const [hMode, setHMode] = useState('Manual');
  const [vMode, setVMode] = useState('Manual');
  const [alignPos, setAlignPos] = useState(4);
  const [matchSize, setMatchSize] = useState(false);

  const dis = loading || batchLoading || !hasImage;

  const setTopV    = (v) => { setTop(v);    if (linkTB) setBottom(v); };
  const setBottomV = (v) => { setBottom(v); if (linkTB) setTop(v); };
  const setLeftV   = (v) => { setLeft(v);   if (linkLR) setRight(v); };
  const setRightV  = (v) => { setRight(v);  if (linkLR) setLeft(v); };

  const handleAutoToggle = useCallback(async (checked) => {
    setAutoMargins(checked);
    if (!checked || !hasImage) return;
    setDetecting(true);
    try {
      const res = await onDetect();
      if (res) {
        setTop(res.top_mm ?? 5); setBottom(res.bottom_mm ?? 5);
        setLeft(res.left_mm ?? 10); setRight(res.right_mm ?? 10);
      }
    } finally { setDetecting(false); }
  }, [onDetect, hasImage]);

  const handleApply = useCallback(async () => {
    await onApply(top, bottom, left, right);
  }, [onApply, top, bottom, left, right]);

  const Spinner = ({ label, value, onChange }) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <div style={{ fontSize: 10, color: 'var(--color-text-3)', textAlign: 'center' }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 0,
        border: '1px solid var(--color-border)', borderRadius: 5, overflow: 'hidden' }}>
        <button onClick={() => onChange(Math.max(0, parseFloat((value - 0.5).toFixed(1))))}
          style={{ padding: '3px 6px', background: 'var(--color-surface-3)',
            border: 'none', cursor: 'pointer', color: 'var(--color-text-2)', fontSize: 12 }}>−</button>
        <input type="number" value={value} min={0} max={50} step={0.5}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          style={{ width: 44, textAlign: 'center', background: 'none', border: 'none',
            outline: 'none', color: 'var(--color-text)', fontSize: 12, padding: '3px 0' }} />
        <button onClick={() => onChange(parseFloat((value + 0.5).toFixed(1)))}
          style={{ padding: '3px 6px', background: 'var(--color-surface-3)',
            border: 'none', cursor: 'pointer', color: 'var(--color-text-2)', fontSize: 12 }}>+</button>
      </div>
    </div>
  );

  const LinkBtn = ({ linked, onToggle }) => (
    <button onClick={onToggle} title={linked ? 'Unlink' : 'Link'}
      style={{ background: 'none', border: 'none', cursor: 'pointer',
        color: linked ? 'var(--color-cyan)' : 'var(--color-text-3)',
        fontSize: 16, padding: '0 2px', alignSelf: 'flex-end', marginBottom: 2 }}>
      {linked ? '🔗' : '🔓'}
    </button>
  );

  return (
    <>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, cursor: 'pointer' }}>
        <input type="checkbox" checked={autoMargins}
          onChange={e => handleAutoToggle(e.target.checked)}
          disabled={dis} id="chk-auto-margins"
          style={{ accentColor: 'var(--color-cyan)', width: 14, height: 14 }} />
        <span style={{ fontSize: 12, color: 'var(--color-text-2)', fontWeight: 600 }}>Auto Margins</span>
        {detecting && <span className="spinner" style={{ width: 12, height: 12 }} />}
      </label>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <Spinner label="Top"    value={top}    onChange={setTopV} />
        <div />
        <Spinner label="Bottom" value={bottom} onChange={setBottomV} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 4 }}>
        <LinkBtn linked={linkTB} onToggle={() => setLinkTB(l => !l)} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <Spinner label="Left"  value={left}  onChange={setLeftV} />
        <LinkBtn linked={linkLR} onToggle={() => setLinkLR(l => !l)} />
        <Spinner label="Right" value={right} onChange={setRightV} />
      </div>
      <div style={{ fontSize: 10, color: 'var(--color-text-3)', marginBottom: 12 }}>Values in mm</div>

      <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 10, marginTop: 4, marginBottom: 10 }}>
        <div className="field-label" style={{ marginBottom: 8 }}>Alignment</div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, cursor: 'pointer' }}>
          <input type="checkbox" checked={matchSize} onChange={e => setMatchSize(e.target.checked)}
            id="chk-match-size" style={{ accentColor: 'var(--color-cyan)', width: 13, height: 13 }} />
          <span style={{ fontSize: 11, color: 'var(--color-text-2)' }}>Match size with other pages</span>
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{ fontSize: 11, color: 'var(--color-text-3)' }}>Horizontal mode:</span>
          <select value={hMode} onChange={e => setHMode(e.target.value)}
            style={{ fontSize: 11, background: 'var(--color-surface-3)',
              border: '1px solid var(--color-border)', borderRadius: 4,
              color: 'var(--color-text)', padding: '2px 4px' }}>
            <option>Manual</option><option>Auto</option><option>Original</option>
          </select>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', alignItems: 'center', gap: 6, marginBottom: 10 }}>
          <span style={{ fontSize: 11, color: 'var(--color-text-3)' }}>Vertical mode:</span>
          <select value={vMode} onChange={e => setVMode(e.target.value)}
            style={{ fontSize: 11, background: 'var(--color-surface-3)',
              border: '1px solid var(--color-border)', borderRadius: 4,
              color: 'var(--color-text)', padding: '2px 4px' }}>
            <option>Manual</option><option>Auto</option><option>Original</option>
          </select>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4, marginBottom: 10 }}>
          {[0,1,2,3,4,5,6,7,8].map(i => {
            const icons = ['↖','↑','↗','←','·','→','↙','↓','↘'];
            return (
              <button key={i} onClick={() => setAlignPos(i)} style={{
                aspectRatio: '1', padding: 0, borderRadius: 4,
                border: `1.5px solid ${alignPos === i ? 'var(--color-cyan)' : 'var(--color-border)'}`,
                background: alignPos === i ? 'rgba(0,229,255,0.12)' : 'var(--color-surface-3)',
                color: alignPos === i ? 'var(--color-cyan)' : 'var(--color-text-3)',
                cursor: 'pointer', fontSize: 14, fontWeight: 600,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>{icons[i]}</button>
            );
          })}
        </div>
      </div>

      <button className="btn w-full" onClick={handleApply} disabled={dis}
        id="btn-margins-apply"
        style={{
          background: !dis ? 'linear-gradient(135deg,#00e5ff22,#7c4dff22)' : undefined,
          color: 'var(--color-text)', border: '1px solid var(--color-border)',
          boxShadow: !dis ? '0 0 12px rgba(0,229,255,0.15)' : 'none',
        }}>
        {loading ? <span className="spinner" /> : '⊡'} Apply To…
      </button>

      <ApplyAllButton
        onClick={() => onApplyAll?.(top, bottom, left, right)}
        pageCount={pageCount}
        disabled={dis}
        loading={batchLoading}
        progress={batchProgress}
        label={`⊡ Apply Margins to All ${pageCount} Pages`}
      />
    </>
  );
}

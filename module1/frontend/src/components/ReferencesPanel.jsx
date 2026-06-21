import { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Use local worker bundled by Vite instead of external CDN
import workerSrc from 'pdfjs-dist/build/pdf.worker.mjs?url';
pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

export default function ReferencesPanel({ refData, pdfUrl, onClose }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [renderData, setRenderData] = useState(null);

  useEffect(() => {
    let active = true;
    if (!refData || !pdfUrl) return;

    const renderPdf = async () => {
      setLoading(true);
      setError(null);
      try {
        const loadingTask = pdfjsLib.getDocument(pdfUrl);
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(refData.page_num);
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        // Calculate scale to fit container width roughly
        const containerWidth = containerRef.current?.clientWidth || 500;
        const unscaledViewport = page.getViewport({ scale: 1.0 });
        const scale = (containerWidth - 32) / unscaledViewport.width; // padding
        
        const viewport = page.getViewport({ scale });
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
          canvasContext: context,
          viewport: viewport,
        };
        
        if (active) {
            await page.render(renderContext).promise;
            
            // Calculate bbox overlay styles.
            // Module 3 OCR renders PDFs at 300 DPI (see pdf_to_images dpi=300);
            // PDF.js uses 72 DPI points. To convert OCR pixel coords → canvas pixels:
            //   ocrScale = (canvas pixels / pdf-point) * (pdf-points / ocr-pixel)
            //            = scale * (72 / 300)
            const OCR_DPI = 300;
            const PDF_DPI = 72;
            const ocrScale = scale * (PDF_DPI / OCR_DPI);

            let highlightStyle = null;
            if (refData.bbox && refData.bbox.length === 4) {
               const [x0, y0, x1, y1] = refData.bbox;
               highlightStyle = {
                 position: 'absolute',
                 left:   x0 * ocrScale,
                 top:    y0 * ocrScale,
                 width:  (x1 - x0) * ocrScale,
                 height: (y1 - y0) * ocrScale,
               };
            }
            if (active) {
                setRenderData({ scale, highlightStyle, height: viewport.height });
                setLoading(false);

                // Auto-scroll logic — attempt to scroll to highlight
                setTimeout(() => {
                    if (highlightStyle && containerRef.current) {
                        const topPos = highlightStyle.top;
                        // scroll such that highlight is roughly in the middle of viewer
                        containerRef.current.scrollTo({
                            top: Math.max(0, topPos - 150),
                            behavior: 'smooth'
                        });
                    }
                }, 200);
            }
        }
      } catch (err) {
        console.error("PDF Render Error:", err);
        if (active) {
            setError(err.message);
            setLoading(false);
        }
      }
    };

    renderPdf();

    return () => { active = false; };
  }, [refData, pdfUrl]);

  if (!refData) return null;

  return (
    <div className="refs-panel">
      <div className="refs-header">
        <span className="refs-title">
          📎 Visual Reference Viewer
        </span>
        <button
          className="btn btn-ghost btn-icon btn-sm"
          onClick={onClose}
          title="Close viewer"
        >
          ✕
        </button>
      </div>

      <div className="refs-list" ref={containerRef} style={{ position: 'relative', overflowY: 'auto', padding: 16 }}>
        {loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 40, color: 'var(--color-text-3)' }}>
                <span className="spinner" style={{ marginBottom: 12 }} />
                Loading PDF Engine...
            </div>
        )}
        {error && <div style={{ padding: 20, color: 'var(--color-red)' }}>Error: {error}</div>}
        
        <div className="pdf-viewer-container" style={{ position: 'relative', margin: '0 auto', display: loading ? 'none' : 'block' }}>
            <canvas ref={canvasRef} style={{ display: 'block', background: '#fff', borderRadius: 4, boxShadow: 'var(--shadow-card)' }} />
            {renderData?.highlightStyle && (
                <div className="pdf-highlight-box" style={renderData.highlightStyle} />
            )}
        </div>
        
        {!loading && (
            <div style={{ marginTop: 24, padding: '16px', background: 'var(--color-surface-2)', borderRadius: 8, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-cyan)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                  📄 EXACT TEXT — PAGE {refData.page_num}
                </div>
                <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--color-text-2)' }}>
                    "{refData.text}"
                </div>
            </div>
        )}
      </div>
    </div>
  );
}

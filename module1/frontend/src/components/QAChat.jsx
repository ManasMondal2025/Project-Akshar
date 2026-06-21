import { useState, useRef, useEffect } from 'react';
import { exportPdfBbox } from '../api';

/**
 * QAChat — Chat-style interface for RAG-powered document Q&A.
 */
export default function QAChat({ messages, onQuery, onShowRefs, onBack, documentId, ocrData, pages = [], pdfPath = '' }) {
  const [input, setInput]           = useState('');
  const [thinking, setThinking]     = useState(false);
  const [exportingBbox, setExportingBbox] = useState(false);
  const bottomRef   = useRef(null);
  const textareaRef = useRef(null);

  const handleExportJson = () => {
    if (!ocrData) return;
    const blob = new Blob([JSON.stringify(ocrData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extracted_ocr_${documentId || 'data'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportBbox = async () => {
    if (!ocrData?.blocks?.length) return;
    if (!pdfPath) {
      alert('No source PDF available. The document must be processed through the pipeline first.');
      return;
    }
    setExportingBbox(true);
    try {
      const blob = await exportPdfBbox(pdfPath, ocrData.blocks, 300);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `layout_bbox_${documentId || 'doc'}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`BBox PDF export failed: ${err.message}`);
    } finally {
      setExportingBbox(false);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || thinking) return;
    setInput('');
    setThinking(true);
    try { await onQuery(q); }
    finally { setThinking(false); }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-area">
      {/* Header */}
      <div className="qa-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onBack} id="btn-qa-back">← Back</button>
          <span className="qa-title">Document Q&amp;A</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {ocrData && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={handleExportJson}
              title="Download raw OCR data as JSON">
              ⬇ Export JSON
            </button>
          )}
          {ocrData?.blocks?.length > 0 && !!pdfPath && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={handleExportBbox}
              disabled={exportingBbox}
              title="Export PDF with colour-coded layout bounding boxes"
              style={{
                background: exportingBbox ? 'rgba(0,229,255,0.08)' : undefined,
                color: exportingBbox ? 'var(--color-cyan)' : undefined,
                border: '1px solid rgba(0,229,255,0.25)',
                display: 'flex', alignItems: 'center', gap: 5,
              }}
            >
              {exportingBbox
                ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Generating…</>
                : <>📦 Export PDF + BBoxes</>}
            </button>
          )}
          {documentId && (
            <span className="qa-doc-badge">doc: {documentId.slice(0, 16)}…</span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {isEmpty && !thinking && (
          <div className="empty-chat">
            <svg className="empty-icon" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="32" cy="32" r="28" />
              <path d="M20 26h24M20 32h18M20 38h14" strokeLinecap="round" />
            </svg>
            <div className="empty-title">Ask a question</div>
            <div className="empty-subtitle">
              Type a question about your document. The AI will find the most relevant passages.
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12, justifyContent: 'center' }}>
              {['Summarize this document', 'What are the key findings?', 'List all mentioned technologies'].map((q, i) => (
                <button key={i} className="btn btn-ghost btn-sm"
                  onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                  style={{ borderRadius: 20, fontSize: 12 }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} onShowRefs={onShowRefs} />
        ))}

        {thinking && (
          <div className="chat-msg assistant">
            <span className="chat-role">AKSHAR AI</span>
            <div className="chat-thinking">
              <span>Searching document</span>
              <div className="thinking-dots">
                <span className="thinking-dot" /><span className="thinking-dot" /><span className="thinking-dot" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-bar">
        <textarea ref={textareaRef} className="chat-input" id="qa-input"
          placeholder="Ask anything about your document… (Enter to send, Shift+Enter for new line)"
          value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey} rows={1}
        />
        <button className="btn btn-primary" onClick={handleSend}
          disabled={!input.trim() || thinking} id="btn-send" style={{ flexShrink: 0 }}>
          {thinking ? <span className="spinner" /> : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" strokeLinecap="round" />
            </svg>
          )}
          Send
        </button>
      </div>
    </div>
  );
}

function ChatMessage({ message, onShowRefs }) {
  const isUser  = message.role === 'user';
  const hasRefs = !isUser && message.refs && message.refs.length > 0;
  const [refsOpen, setRefsOpen] = useState(false);

  return (
    <div className={`chat-msg ${message.role}`}>
      {isUser ? (
        <>
          <span className="chat-role" style={{ textAlign: 'right' }}>YOU</span>
          <div className="chat-bubble">{message.content}</div>
        </>
      ) : (
        <>
          <span className="chat-role">AKSHAR AI</span>
          <div className="chat-bubble"><MarkdownText text={message.content} /></div>

          {hasRefs && (
            <div style={{ marginTop: 10 }}>
              {/* ── Toggle button ───────────────────────────────────── */}
              <button
                id="btn-toggle-refs"
                onClick={() => setRefsOpen(o => !o)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '5px 12px',
                  borderRadius: 20,
                  border: '1px solid var(--color-border)',
                  background: refsOpen ? 'var(--color-surface-3)' : 'var(--color-surface)',
                  color: refsOpen ? 'var(--color-cyan)' : 'var(--color-text-2)',
                  fontSize: 11,
                  fontWeight: 700,
                  cursor: 'pointer',
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                  transition: 'all 0.2s ease',
                }}
              >
                {refsOpen ? (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                      stroke="currentColor" strokeWidth="2">
                      <path d="M17 11H7M12 6l-5 5 5 5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Hide References
                  </>
                ) : (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                      stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/>
                    </svg>
                    Show References ({message.refs.length})
                  </>
                )}
              </button>

              {/* ── References list (collapsible) ───────────────────── */}
              {refsOpen && (
                <div className="chat-refs-list" style={{
                  marginTop: 8,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  animation: 'fadeIn 0.2s ease',
                }}>
                  <div style={{
                    fontSize: 11, fontWeight: 600,
                    color: 'var(--color-text-3)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    marginBottom: 2,
                  }}>
                    Source References:
                  </div>
                  {message.refs.map((ref, idx) => (
                    <div
                      key={idx}
                      className="chat-ref-item"
                      onClick={() => onShowRefs(ref)}
                      style={{
                        padding: '8px 12px',
                        borderRadius: '8px',
                        background: 'var(--color-surface)',
                        border: '1px solid var(--color-border)',
                        cursor: 'pointer',
                        fontSize: 12,
                        color: 'var(--color-text-2)',
                        transition: 'all 0.2s ease',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 4,
                      }}
                      onMouseOver={e => {
                        e.currentTarget.style.borderColor = 'var(--color-cyan)';
                        e.currentTarget.style.background   = 'var(--color-surface-3)';
                      }}
                      onMouseOut={e => {
                        e.currentTarget.style.borderColor = 'var(--color-border)';
                        e.currentTarget.style.background   = 'var(--color-surface)';
                      }}
                    >
                      <div style={{
                        fontSize: 10, fontWeight: 700,
                        color: 'var(--color-cyan)',
                        fontFamily: 'var(--font-mono)',
                      }}>
                        📄 Page {ref.page_num} · click to view in PDF
                      </div>
                      <div style={{
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}>
                        "{ref.text}"
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MarkdownText({ text }) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\n)/g);
  return (
    <span>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('`') && part.endsWith('`')) return (
          <code key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em',
            background: 'rgba(0,229,255,0.1)', padding: '1px 5px', borderRadius: 4 }}>
            {part.slice(1, -1)}
          </code>
        );
        if (part === '\n') return <br key={i} />;
        return part;
      })}
    </span>
  );
}

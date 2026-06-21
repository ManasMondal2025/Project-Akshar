import { useRef, useState } from 'react';

/**
 * UploadZone — Redesigned Drag and drop / click to upload PDFs or images.
 * Akshar AI styled upload portal with animated spinning arcs and floating emoji.
 */
export default function UploadZone({ onFilesSelect, loading }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragOver(true);
    else setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) onFilesSelect(files);
  };

  const handleChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) onFilesSelect(files);
    e.target.value = '';
  };

  return (
    <div className="az-upload-container">
      <div
        className={`az-upload-box-wrapper ${dragOver ? 'az-dragover' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        id="upload-zone"
        style={{ cursor: loading ? 'wait' : 'pointer' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,image/*"
          multiple
          onChange={handleChange}
          style={{ display: 'none' }}
          id="file-input"
        />

        <div className="az-upload-box-inner">
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
              <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
              <p style={{ color: 'var(--color-text-2)', fontSize: 14 }}>Processing…</p>
            </div>
          ) : (
            <>
              <div className="az-upload-icon-container">
                <div className="az-icon-ring az-icon-ring-top"></div>
                <div className="az-icon-ring az-icon-ring-bottom"></div>
                <div className="az-page-emoji">📄</div>
              </div>

              <h3 className="az-upload-title">Upload Document</h3>
              <p className="az-upload-subtitle">Drag &amp; Drop or click to browse</p>

              <button
                type="button"
                className="az-upload-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  if (!loading) inputRef.current?.click();
                }}
              >
                Select File
              </button>

              <div className="az-supported-formats">SUPPORTS PDF, JPG, PNG, TIFF</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

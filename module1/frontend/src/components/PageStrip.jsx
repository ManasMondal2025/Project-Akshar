/**
 * PageStrip — Horizontal strip of page thumbnails with Add button on the right.
 *
 * Props:
 *  - pages: array of page objects (each has currentPreview, filename)
 *  - activeIndex: currently selected page index
 *  - onSelect: callback(index) to switch active page
 *  - onRemove: callback(index) to remove a page
 *  - onAddMore: callback to trigger adding more images
 */
export default function PageStrip({ pages, activeIndex, onSelect, onRemove, onAddMore }) {
  return (
    <div className="page-strip">
      {/* Scrollable thumbnails */}
      <div className="page-strip-inner">
        {pages.map((page, idx) => (
          <div
            key={page.id}
            className={`page-thumb ${idx === activeIndex ? 'page-thumb-active' : ''}`}
            onClick={() => onSelect(idx)}
            title={page.filename}
          >
            <div className="page-thumb-img-wrapper">
              <img
                src={page.currentPreview}
                alt={`Page ${idx + 1}`}
                className="page-thumb-img"
              />
              {/* Remove button */}
              {pages.length > 1 && (
                <button
                  className="page-thumb-remove"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(idx);
                  }}
                  title="Remove this page"
                >
                  ×
                </button>
              )}
            </div>
            <div className="page-thumb-label">
              <span className="page-thumb-number">{idx + 1}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Add More button — pinned to the right */}
      <div className="page-thumb page-thumb-add" onClick={onAddMore} title="Add more images">
        <div className="page-thumb-img-wrapper page-thumb-add-inner">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </div>
      </div>
    </div>
  );
}

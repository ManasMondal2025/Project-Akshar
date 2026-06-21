import { useState } from 'react';

export default function StPanel({ number, title, children, open, onToggle, colorClass = '' }) {
  const [localOpen, setLocalOpen] = useState(false);
  
  const isControlled = open !== undefined;
  const isOpen = isControlled ? open : localOpen;

  const handleClick = () => {
    if (isControlled) {
      if (onToggle) onToggle(!isOpen);
    } else {
      setLocalOpen(!isOpen);
    }
  };

  return (
    <div className={`st-panel ${colorClass} ${isOpen ? 'open' : ''}`}>
      <div
        className={`st-panel-header${isOpen ? ' open' : ''}`}
        onClick={handleClick}
        role="button"
        aria-expanded={isOpen}
      >
        <span className="st-panel-num">{number}</span>
        <span className="st-panel-title">{title}</span>
        <span className="st-panel-arrow">▶</span>
      </div>
      {isOpen && <div className="st-panel-body">{children}</div>}
    </div>
  );
}

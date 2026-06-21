/**
 * ApplyAllButton — shared "Apply to All Pages" button for all workbench panels.
 *
 * Props:
 *   onClick     — handler to call
 *   pageCount   — total number of pages currently in the workbench
 *   disabled    — hard-disable (e.g. split panel)
 *   loading     — true while batch is running
 *   progress    — current page index (1-based) being processed
 *   title       — tooltip override (for disabled state explanation)
 *   label       — custom button label (optional)
 */
export default function ApplyAllButton({
  onClick,
  pageCount = 1,
  disabled = false,
  loading = false,
  progress = 0,
  title,
  label,
}) {
  const isDisabled = disabled || loading || pageCount < 2;

  const defaultTitle = disabled
    ? title
    : pageCount < 2
    ? 'Upload multiple pages to use Apply to All'
    : title || `Apply same settings to all ${pageCount} pages`;

  const buttonLabel = loading
    ? `Applying… (${progress}/${pageCount})`
    : label || `⟳ Apply to All ${pageCount} Pages`;

  return (
    <button
      className="apply-all-btn"
      onClick={isDisabled ? undefined : onClick}
      disabled={isDisabled}
      title={defaultTitle}
      style={{ width: '100%' }}
    >
      {buttonLabel}
    </button>
  );
}

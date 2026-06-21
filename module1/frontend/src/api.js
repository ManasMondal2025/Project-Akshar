/**
 * PROJECT AKSHAR — Unified API Client
 *
 * Module 1 backend (port 8000): upload, transform, corners, split,
 *   margins, content, export, pipeline, OCR, QA.
 * Module 2 backend (port 8001): dewarp, deskew, enhancement.
 */

const API_BASE    = 'http://localhost:8000';  // module1 — workbench + pipeline
const API_BASE_M2 = 'http://localhost:8001';  // module2 — dewarp + enhance

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

async function apiPostForm(endpoint, formData) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// Calls module2 backend on port 8001 (dewarp + enhance)
async function apiPost_m2(endpoint, body) {
  const res = await fetch(`${API_BASE_M2}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// --------------------------------------------------------------------------
// Module 1 — Image Workbench
// --------------------------------------------------------------------------

/** Upload a single image file. Returns { file_path, preview, width, height } */
export async function uploadImage(file) {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm('/upload', form);
}

/** Apply perspective transform with 4 corner points [[x,y], ...] */
export async function transformImage(imagePath, corners) {
  return apiPost('/transform', { image_path: imagePath, corners });
}

/** Grid-based ScanTailor dewarping (manual mode — uses detected/edited grid) */
export async function dewarpImage(imagePath, strength = 1.0, customGrid = null) {
  const body = { image_path: imagePath, strength };
  if (customGrid && customGrid.topCPs && customGrid.botCPs) {
    body.row_curves = [customGrid.topCPs, customGrid.botCPs];
  } else if (customGrid && customGrid.row_curves) {
    body.row_curves = customGrid.row_curves;
  }
  return apiPost_m2('/dewarp', body);
}


/**
 * ML-based automatic dewarping using the ICCV 2023 neural network.
 * No grid interaction needed — the model detects and corrects warp automatically.
 */
export async function dewarpImageAuto(imagePath) {
  return apiPost_m2('/dewarp/auto', { image_path: imagePath });
}

/**
 * B-spline poly dewarping using the dewarp2 algorithm.
 * Automatically estimates 4 text-density curves and flattens the page
 * without any grid interaction or neural network model required.
 */
export async function dewarpImagePoly(imagePath, customCurves = null) {
  return apiPost_m2('/dewarp/poly', { image_path: imagePath, custom_curves: customCurves });
}

/**
 * Estimate the 4 B-spline poly curves WITHOUT applying dewarping.
 * Returns { detected, curves: [{id, name, color, points:[{x,y},...]},...], width, height }
 * for rendering the 4 poly curves as an overlay on the canvas.
 */
export async function estimatePolyCurves(imagePath) {
  return apiPost_m2('/dewarp/poly/estimate-curves', { image_path: imagePath });
}

/**
 * Analyze the document warp grid WITHOUT applying correction.
 * Returns { detected, row_curves, col_lines, row_count, width, height }
 * for rendering the ScanTailor-style blue grid overlay.
 */
export async function analyzeDewarpGrid(imagePath, nCols = 30) {
  return apiPost_m2('/dewarp/analyze-grid', { image_path: imagePath, n_cols: nCols });
}

/** Auto-detect and correct skew */
export async function deskewAuto(imagePath) {
  return apiPost_m2('/deskew', { image_path: imagePath });
}

/** User-specified rotation correction */
export async function deskewManual(imagePath, angle) {
  return apiPost_m2('/deskew/manual', { image_path: imagePath, angle });
}

/** Otsu binarization with output format */
export async function enhanceOtsu(imagePath, outputFormat = 'bw') {
  return apiPost_m2('/enhance/otsu', { image_path: imagePath, output_format: outputFormat });
}

/** Adaptive Gaussian threshold with output format */
export async function enhanceAdaptive(imagePath, outputFormat = 'bw') {
  return apiPost_m2('/enhance/adaptive', { image_path: imagePath, output_format: outputFormat });
}

/** Export pages to PDF — returns binary blob */
export async function exportPdf(imagePaths) {
  const res = await fetch(`${API_BASE_M2}/export/pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_paths: imagePaths }),
  });
  if (!res.ok) throw new Error(`PDF export failed: ${res.statusText}`);
  return res.blob();
}

/**
 * Export PDF with colour-coded layout bounding boxes overlaid.
 * Uses the same PDF that OCR ran on so bbox coordinates align exactly.
 * @param {string}   pdfPath  - server-side path to the OCR source PDF
 * @param {object[]} blocks   - OCR blocks: [{page, type, bbox:[x1,y1,x2,y2], text}, ...]
 * @param {number}   ocr_dpi  - DPI used when OCR was run (default 300)
 * @returns {Blob} annotated PDF binary
 */
export async function exportPdfBbox(pdfPath, blocks, ocr_dpi = 300) {
  const res = await fetch(`${API_BASE_M2}/export/pdf-bbox`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pdf_path: pdfPath, blocks, ocr_dpi }),
  });
  if (!res.ok) throw new Error(`BBox PDF export failed: ${res.statusText}`);
  return res.blob();
}

// --------------------------------------------------------------------------
// Pipeline — Orchestration
// --------------------------------------------------------------------------

/** Upload a PDF or image to the pipeline router. Returns { file_path, file_type } */
export async function pipelineUpload(file) {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm('/pipeline/upload', form);
}

/** Detect if PDF is digital or scanned. Returns { document_type, recommended_route } */
export async function pipelineDetect(pdfPath) {
  return apiPost('/pipeline/detect', { pdf_path: pdfPath });
}

/** Extract pages from scanned PDF. Returns { image_paths: [...] } */
export async function pipelineExtractPages(pdfPath) {
  return apiPost('/pipeline/extract-pages', { pdf_path: pdfPath });
}

/**
 * Register an already-extracted server-side image into the workbench.
 * Returns { file_path, preview, width, height } — same shape as uploadImage.
 */
export async function pipelineRegisterPage(serverPath, label = '') {
  return apiPost('/pipeline/register-page', { server_path: serverPath, label });
}

/** Assemble processed images into PDF. Returns { pdf_path } */
export async function pipelineConvertToPdf(imagePaths, filename = 'akshar_output.pdf') {
  return apiPost('/pipeline/convert-to-pdf', {
    image_paths: imagePaths,
    output_filename: filename,
  });
}

/** Run Module 3 OCR on a PDF path. Returns { blocks: [...] } */
export async function pipelineRunOcr(pdfPath, sarvamApiKey = '') {
  return apiPost('/pipeline/run-ocr', {
    pdf_path: pdfPath,
    sarvam_api_key: sarvamApiKey,
  });
}

/** Run Sarvam AI API Document Parse on a PDF path. Returns { blocks: [...] } */
export async function pipelineRunOcrSarvam(pdfPath, sarvamApiKey = '') {
  return apiPost('/pipeline/run-ocr-sarvam', {
    pdf_path: pdfPath,
    sarvam_api_key: sarvamApiKey,
  });
}

/** Index OCR blocks into ChromaDB */
export async function pipelineIndex(documentId, metadataBlocks, collectionName = 'project_akshar') {
  return apiPost('/pipeline/index', {
    document_id: documentId,
    metadata_blocks: metadataBlocks,
    collection_name: collectionName,
  });
}

/** Run a RAG query. Returns { answer, references: [{text, page_num}] } */
export async function pipelineQuery(query, documentId, groqApiKey = '', collectionName = 'project_akshar') {
  return apiPost('/pipeline/query', {
    query,
    document_id: documentId,
    collection_name: collectionName,
    groq_api_key: groqApiKey,
  });
}

// --------------------------------------------------------------------------
// ScanTailor Tools — Fix Orientation, Adjust Corners, Split Pages, Margins
// --------------------------------------------------------------------------

/** Rotate image. angle: 90 (CW), -90 (CCW), 180 */
export async function orientRotate(imagePath, angle) {
  return apiPost('/orient/rotate', { image_path: imagePath, angle });
}

/** Suggest best orientation. Returns { suggested_angle, is_landscape } */
export async function orientAuto(imagePath) {
  return apiPost('/orient/auto', { image_path: imagePath });
}

/** Auto-detect document corners. Returns { corners: [[x,y],...], width, height }
 *  @param {'classical'|'cnn'} method  Detection method (default: 'classical')
 */
export async function cornersDetect(imagePath, method = 'classical') {
  return apiPost('/corners/detect', { image_path: imagePath, method });
}

/** Apply perspective warp using 4 corners [TL,TR,BR,BL]. Returns processed image. */
export async function cornersApply(imagePath, corners) {
  return apiPost('/corners/apply', { image_path: imagePath, corners });
}

/** Detect page layout. Returns { layout_type, split_x, split_x_ratio, confidence } */
export async function splitDetect(imagePath) {
  return apiPost('/split/detect', { image_path: imagePath });
}

/**
 * Fast spine detection via projection-profile dip + Hough-line vote.
 * Immediately returns left/right pixel-crop previews (no second apply call needed).
 * Returns { detected, spine_x, spine_x_ratio, left_page, right_page }
 */
export async function splitDetectSpine(imagePath) {
  return apiPost('/split/detect-spine', { image_path: imagePath });
}

/**
 * Split image using the geometry-first PageLayout pipeline.
 * When pageLayout is provided (from splitDetect), uses polygon clipping + perspective warp.
 * Falls back to pixel-crop if pageLayout is absent.
 *
 * @param {string}  imagePath
 * @param {number}  splitX       pixel x position (fallback only)
 * @param {string}  layoutType   'two_pages' | 'single_uncut' | 'single_cut'
 * @param {number}  contentX1    pixel x (fallback only)
 * @param {number}  contentX2    pixel x (fallback only)
 * @param {string}  selectedSide 'left' | 'right' | 'both'
 * @param {object}  pageLayout   PageLayout dict from splitDetect (preferred)
 */
export async function splitApply(
  imagePath, splitX, layoutType = 'two_pages',
  contentX1 = 0, contentX2 = null,
  selectedSide = 'both', pageLayout = null,
) {
  const body = {
    image_path:    imagePath,
    split_x:       splitX,
    layout_type:   layoutType,
    content_x1:    Math.round(contentX1),
    selected_side: selectedSide,
  };
  if (contentX2 !== null) body.content_x2 = Math.round(contentX2);
  if (pageLayout)          body.page_layout = pageLayout;
  return apiPost('/split/apply', body);
}


/** Auto-detect content box. Returns { content_box: {x,y,width,height}, content_rect } */
export async function contentDetect(imagePath, dpi = 300) {
  return apiPost('/content/detect', { image_path: imagePath, dpi });
}

/** Crop image to detected content area. Returns processed image. */
export async function contentApply(imagePath, contentRect = null, dpi = 300) {
  return apiPost('/content/apply', { image_path: imagePath, content_rect: contentRect, dpi });
}

/** Auto-detect margins. Returns { top_mm, bottom_mm, left_mm, right_mm } */
export async function marginsDetect(imagePath) {
  return apiPost('/margins/detect', { image_path: imagePath });
}

/** Crop + pad image with given margins (mm). Returns processed image. */
export async function marginsApply(imagePath, topMm, bottomMm, leftMm, rightMm) {
  return apiPost('/margins/apply', {
    image_path: imagePath,
    top_mm: topMm, bottom_mm: bottomMm,
    left_mm: leftMm, right_mm: rightMm,
  });
}


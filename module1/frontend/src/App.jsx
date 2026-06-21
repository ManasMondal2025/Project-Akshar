import { useState, useCallback, useRef, useEffect } from 'react';
import {
  uploadImage,
  pipelineUpload,
  pipelineDetect,
  pipelineExtractPages,
  pipelineConvertToPdf,
  pipelineRunOcr,
  pipelineRunOcrSarvam,
  pipelineIndex,
  pipelineQuery,
  pipelineRegisterPage,
  transformImage,
  dewarpImage,
  dewarpImageAuto,
  dewarpImagePoly,
  estimatePolyCurves,
  analyzeDewarpGrid,
  deskewAuto,
  deskewManual,
  enhanceOtsu,
  enhanceAdaptive,
  exportPdf,
  // ScanTailor new
  orientRotate,
  cornersDetect,
  cornersApply,
  splitDetect,
  splitDetectSpine,
  splitApply,
  marginsDetect,
  marginsApply,
  contentDetect,
  contentApply,
} from './api';

import CanvasEditor from './components/CanvasEditor';
import UploadZone from './components/UploadZone';
import Controls from './components/Controls';
import Preview from './components/Preview';
import PageStrip from './components/PageStrip';
import QAChat from './components/QAChat';
import ReferencesPanel from './components/ReferencesPanel';

// ScanTailor new panels + overlays
import FixOrientation from './components/FixOrientation';
import AdjustCorners from './components/AdjustCorners';
import SplitPage, { SplitBoundaryOverlay } from './components/SplitPage';
import Deskew, { DeskewArcOverlay } from './components/Deskew';
import Margins from './components/Margins';
import ContentSelection from './components/ContentSelection';
import StPanel from './components/StPanel';


// -----------------------------------------------------------------------
// Stage constants
// -----------------------------------------------------------------------

const STAGE_UPLOAD = 'upload';
const STAGE_WORKBENCH = 'workbench';
const STAGE_QA = 'qa';

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

const initCorners = (w, h) => {
  const m = Math.min(w, h) * 0.05;
  return [
    { x: m, y: m },
    { x: w - m, y: m },
    { x: w - m, y: h - m },
    { x: m, y: h - m },
  ];
};

const createPage = (data, name) => ({
  id: Date.now() + Math.random(),
  filename: name,
  originalPreview: data.preview,
  currentPreview: data.preview,
  originalPath: data.file_path,
  currentPath: data.file_path,
  imageWidth: data.width,
  imageHeight: data.height,
  corners: initCorners(data.width, data.height),
  activeFilter: null,
  history: [{ action: 'Upload', preview: data.preview, path: data.file_path, width: data.width, height: data.height }],
});

// -----------------------------------------------------------------------
// App
// -----------------------------------------------------------------------

export default function App() {
  // Stage routing
  const [stage, setStage] = useState(STAGE_UPLOAD);

  // ── Keep browser history in sync with stage so the browser ← button
  //    navigates between stages instead of leaving the SPA entirely ──────
  const setStageNav = useCallback((newStage) => {
    // Push a history entry so the browser back button can return here
    window.history.pushState({ stage: newStage }, '');
    setStage(newStage);
  }, []);

  useEffect(() => {
    // Seed the initial history entry
    window.history.replaceState({ stage: STAGE_UPLOAD }, '');

    const onPopState = (e) => {
      const s = e.state?.stage;
      if (s === STAGE_UPLOAD || s === STAGE_WORKBENCH || s === STAGE_QA) {
        setStage(s);
      } else {
        // Unknown state — stay on upload rather than leaving the app
        window.history.pushState({ stage: STAGE_UPLOAD }, '');
        setStage(STAGE_UPLOAD);
      }
    };

    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  // Pipeline state
  const [docType, setDocType] = useState(null);   // 'digital' | 'scanned'
  const [documentId, setDocumentId] = useState('');
  const [pdfPath, setPdfPath] = useState('');

  // Workflow choice modal state (shown for non-digital PDFs)
  const [showWorkflowChoice, setShowWorkflowChoice] = useState(false);
  const [showQAModal, setShowQAModal] = useState(false); // QA engine selection
  const qaDropdownRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (qaDropdownRef.current && !qaDropdownRef.current.contains(event.target)) {
        setShowQAModal(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [qaDropdownRef]);

  const [pendingPdfData, setPendingPdfData] = useState(null); // { file_path, file_id, filename }
  const [pipelineStep, setPipelineStep] = useState(null); // for progress display

  // Workbench state
  const [pages, setPages] = useState([]);
  const [activePageIndex, setActivePageIndex] = useState(0);
  const addInputRef = useRef(null);

  // QA state
  const [messages, setMessages] = useState([]);
  const [activeRefs, setActiveRefs] = useState(null);
  const [showRefs, setShowRefs] = useState(false);

  // ── ScanTailor new panel state ────────────────────────────────────────
  // Open states for panels that affect canvas overlays
  const [perspectiveOpen, setPerspectiveOpen] = useState(false);
  const [dewarpOpen, setDewarpOpen] = useState(false);
  const [dewarpGrid, setDewarpGrid] = useState(null); // grid data for dewarp overlay
  const [dewarpMode, setDewarpMode] = useState('manual'); // 'auto' | 'manual' | 'poly'
  const [polyGrid, setPolyGrid] = useState(null); // poly dewarp curve overlay

  // Deskew arc overlay
  const [deskewMode, setDeskewMode] = useState('auto'); // 'auto' | 'manual'
  const [deskewManualAngle, setDeskewManualAngle] = useState(0);
  const [deskewOverlay, setDeskewOverlay] = useState(false);  // show arc on canvas

  // Split page
  const [splitPanelOpen, setSplitPanelOpen] = useState(false); // toggle visibility like perspective
  const [splitXRatio, setSplitXRatio] = useState(null); // null → hidden
  const [splitOverlayData, setSplitOverlayData] = useState(null); // full overlay state from detect
  const [splitBoundary, setSplitBoundary] = useState(null); // 4 × {x,y} ratio corners
  const [splitCutter, setSplitCutter] = useState(null); // {top:{x,y}, bottom:{x,y}} ratios
  const [splitPageLayout, setSplitPageLayout] = useState(null); // raw PageLayout dict from backend

  // Adjust Corners
  const [cornersPanelOpen, setCornersPanelOpen] = useState(false);
  const [cornerDetectMethod, setCornerDetectMethod] = useState('classical'); // 'classical' | 'cnn'

  // Content Selection
  const [contentPanelOpen, setContentPanelOpen] = useState(false);

  // UI state
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [ocrData, setOcrData] = useState(null);

  const activePage = pages[activePageIndex] || null;


  // -----------------------------------------------------------------------
  // Status toast
  // -----------------------------------------------------------------------

  const toast = useCallback((text, type = 'info') => {
    setStatusMsg({ text, type });
    setTimeout(() => setStatusMsg(null), 3500);
  }, []);

  const updatePage = useCallback((index, updates) => {
    setPages(prev => prev.map((p, i) => i === index ? { ...p, ...updates } : p));
  }, []);

  // -----------------------------------------------------------------------
  // Stage 1 — Upload handler
  // -----------------------------------------------------------------------

  const handleUpload = useCallback(async (files) => {
    setUploadLoading(true);
    try {
      const file = files[0]; // pipeline handles first file for type detection
      const ext = file.name.split('.').pop().toLowerCase();

      if (ext === 'pdf') {
        // ── PDF: detect type, then route accordingly ──────────────────────
        toast('Uploading & analysing PDF…', 'info');
        setPipelineStep('upload');
        const uploaded = await pipelineUpload(file);

        setPipelineStep('detect');
        const detection = await pipelineDetect(uploaded.file_path);

        const dType = detection.document_type;
        setDocType(dType);
        setPdfPath(uploaded.file_path);

        const docId = uploaded.file_id;
        setDocumentId(docId);

        if (dType === 'digital') {
          // ── Digital PDF → Module 3 (OCR) → Module 4 (QA) ──────────────
          toast('Digital PDF detected — running OCR…', 'info');
          setPipelineStep('ocr');
          const ocrResult = await pipelineRunOcr(uploaded.file_path, '');
          setOcrData(ocrResult);

          toast('Building vector index…', 'info');
          setPipelineStep('index');
          await pipelineIndex(docId, ocrResult.blocks);

          toast('Document ready — ask your questions!', 'success');
          setPipelineStep(null);
          setMessages([{
            role: 'assistant',
            content: `📄 **Digital PDF indexed** — ${ocrResult.block_count} text blocks extracted from **${file.name}**. Ask me anything about this document!`,
            refs: [],
          }]);
          setStageNav(STAGE_QA);

        } else {
          // ── Scanned PDF → Show Workflow Choice Modal ──
          setPendingPdfData({ file_path: uploaded.file_path, file_id: docId, filename: file.name });
          setPipelineStep(null);
          setShowWorkflowChoice(true);
          toast('Scanned PDF detected — choose your workflow', 'info');
        }

      } else {
        // ── Direct image upload → workbench (skip Module 1) ──────────────
        toast(`Uploading ${files.length} image${files.length > 1 ? 's' : ''}…`, 'info');
        const newPages = [];
        for (const f of files) {
          const data = await uploadImage(f);
          newPages.push(createPage(data, f.name));
        }

        // Add pages directly — images land in the workbench as-is.
        setPages(prev => [...prev, ...newPages]);

        setActivePageIndex(0);
        setDocType('scanned');
        setDocumentId(`img_${Date.now()}`);
        toast(`${files.length} image${files.length > 1 ? 's' : ''} loaded`, 'success');
        setStageNav(STAGE_WORKBENCH);
      }

    } catch (err) {
      console.error(err);
      toast(err.message || 'Upload failed', 'error');
    } finally {
      setUploadLoading(false);
      setPipelineStep(null);
    }
  }, [toast]);

  // -----------------------------------------------------------------------
  // Workflow choice handlers — kept as no-ops (modal no longer shown)
  // -----------------------------------------------------------------------

  // handleDirectQA / handlePreprocessing are restored to allow the user
  // to choose between skipping directly to QA or going through the workbench.
  const handleDirectQA = useCallback(async () => {
    if (!pendingPdfData) return;
    setShowWorkflowChoice(false);
    setUploadLoading(true);
    try {
      const { file_path, file_id, filename } = pendingPdfData;

      toast('Running OCR on PDF…', 'info');
      setPipelineStep('ocr');
      const ocrResult = await pipelineRunOcr(file_path, '');
      setOcrData(ocrResult);

      toast('Building vector index…', 'info');
      setPipelineStep('index');
      await pipelineIndex(file_id, ocrResult.blocks);

      toast('Document ready — ask your questions!', 'success');
      setPipelineStep(null);
      setMessages([{
        role: 'assistant',
        content: `📄 **PDF indexed (direct)** — ${ocrResult.block_count} text blocks extracted from **${filename}**. Ask me anything about this document!`,
        refs: [],
      }]);
      setStageNav(STAGE_QA);

    } catch (err) {
      console.error(err);
      toast(err.message || 'Direct QA failed', 'error');
    } finally {
      setUploadLoading(false);
      setPipelineStep(null);
      setPendingPdfData(null);
    }
  }, [pendingPdfData, toast]);

  const handlePreprocessing = useCallback(async () => {
    if (!pendingPdfData) return;
    setShowWorkflowChoice(false);
    setUploadLoading(true);
    try {
      const { file_path, filename } = pendingPdfData;

      toast('Extracting pages…', 'info');
      setPipelineStep('extract');
      const extracted = await pipelineExtractPages(file_path);
      
      toast(`${extracted.page_count || extracted.image_paths.length} pages extracted — opening workbench…`, 'success');
      setPipelineStep(null);

      const newPages = [];
      for (let i = 0; i < extracted.image_paths.length; i++) {
        const imgPath = extracted.image_paths[i];
        const data = await pipelineRegisterPage(imgPath, `Page ${i + 1}`);
        newPages.push(createPage(data, `Page ${i + 1}`));
      }

      setPages(prev => [...prev, ...newPages]);
      setActivePageIndex(0);
      setStageNav(STAGE_WORKBENCH);

    } catch (err) {
      console.error(err);
      toast(err.message || 'Preprocessing failed', 'error');
    } finally {
      setUploadLoading(false);
      setPipelineStep(null);
      setPendingPdfData(null);
    }
  }, [pendingPdfData, toast]);

  // -----------------------------------------------------------------------
  // Workbench handlers
  // -----------------------------------------------------------------------

  const handleTransform = useCallback(async () => {
    if (!activePage) return;
    setLoading(true);
    try {
      const corners = activePage.corners.map(c => [c.x, c.y]);
      const data = await transformImage(activePage.currentPath, corners);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        corners: initCorners(data.width, data.height),
        activeFilter: 'Perspective Transform',
        history: [...activePage.history,
        { action: 'Transform', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Perspective transform applied', 'success');
    } catch (e) { toast(e.message || 'Transform failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleDewarp = useCallback(async (strength) => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await dewarpImage(activePage.currentPath, strength, dewarpGrid);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: `Grid Dewarp (${strength.toFixed(2)}×)`,
        history: [...activePage.history,
        { action: `Grid Dewarp (${strength.toFixed(2)}×)`, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      setDewarpGrid(null); // clear grid overlay after applying
      toast('Grid dewarp applied', 'success');
    } catch (e) { toast(e.message || 'Dewarp failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── ML-based automatic dewarping (ICCV 2023 neural network) ──────────────
  const handleDewarpAuto = useCallback(async () => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await dewarpImageAuto(activePage.currentPath);
      updatePage(activePageIndex, {
        currentPreview: data.preview,
        currentPath: data.image_path,
        imageWidth: data.width,
        imageHeight: data.height,
        activeFilter: 'Auto Dewarp (ML)',
        history: [...activePage.history,
        { action: 'Auto Dewarp (ML)', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Auto dewarp applied (neural network)', 'success');
    } catch (e) { toast(e.message || 'Auto dewarp failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── B-spline poly dewarping (dewarp2 algorithm) ────────────────────────────
  const handleDewarpPoly = useCallback(async () => {
    if (!activePage) return;
    setLoading(true);
    try {
      // Only pass customCurves if all curves have control_points
      let customCurves = null;
      if (polyGrid?.polyCurves && polyGrid.polyCurves.every(c => c.control_points)) {
        customCurves = polyGrid.polyCurves.map(c => ({
          id: c.id,
          name: c.name,
          color: c.color,
          control_points: c.control_points
        }));
      }
      const data = await dewarpImagePoly(activePage.currentPath, customCurves);
      updatePage(activePageIndex, {
        currentPreview: data.preview,
        currentPath: data.image_path,
        imageWidth: data.width,
        imageHeight: data.height,
        activeFilter: 'Poly Dewarp (B-spline)',
        history: [...activePage.history,
        { action: 'Poly Dewarp (B-spline)', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Poly dewarp applied (B-spline curves)', 'success');
    } catch (e) { toast(e.message || 'Poly dewarp failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── Estimate poly curves for canvas overlay (no dewarping applied) ──────────
  const handleEstimatePolyCurves = useCallback(async () => {
    if (!activePage) return;
    // Toggle off if already showing
    if (polyGrid) { setPolyGrid(null); return; }
    setLoading(true);
    try {
      const data = await estimatePolyCurves(activePage.currentPath);
      if (data.detected && data.curves?.length) {
        // Convert to the format CanvasEditor expects for poly curves overlay
        setPolyGrid({ detected: true, polyCurves: data.curves, width: data.width, height: data.height });
        toast('5 B-spline curves detected — click Apply Poly Dewarp to correct', 'success');
      } else {
        toast('Could not detect curves on this image', 'info');
      }
    } catch (e) { toast(e.message || 'Poly curve estimation failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, polyGrid, toast]);

  // Toggle the ScanTailor-style blue grid overlay
  const handleAnalyzeGrid = useCallback(async () => {
    if (!activePage) return;
    // Toggle off if already showing
    if (dewarpGrid) { setDewarpGrid(null); return; }
    setLoading(true);
    try {
      const data = await analyzeDewarpGrid(activePage.currentPath);
      if (!data.detected) {
        toast('No text rows detected — try on a document image with visible text lines', 'info');
      } else {
        setDewarpGrid(data);
        toast(`Grid: ${data.row_count} text rows detected`, 'success');
      }

    } catch (e) { toast(e.message || 'Grid analysis failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, dewarpGrid, toast]);

  const handleDeskewAuto = useCallback(async () => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await deskewAuto(activePage.currentPath);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: `Auto Deskew (${data.detected_angle?.toFixed(2) || '?'}°)`,
        history: [...activePage.history,
        { action: `Auto Deskew`, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast(`Deskew applied — corrected ${data.detected_angle?.toFixed(2) || '?'}°`, 'success');
    } catch (e) { toast(e.message || 'Deskew failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleDeskewManual = useCallback(async (angle) => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await deskewManual(activePage.currentPath, angle);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: `Manual Deskew (${angle}°)`,
        history: [...activePage.history,
        { action: `Manual Deskew (${angle}°)`, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast(`Manual deskew applied (${angle}°)`, 'success');
    } catch (e) { toast(e.message || 'Manual deskew failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleEnhance = useCallback(async (type, outputFormat = 'bw') => {
    if (!activePage) return;
    setLoading(true);
    const label = outputFormat === 'color' ? 'Color Enhancement' : 'B/W Enhancement';
    try {
      let data;
      if (type === 'otsu') data = await enhanceOtsu(activePage.currentPath, outputFormat);
      else if (type === 'adaptive') data = await enhanceAdaptive(activePage.currentPath, outputFormat);
      else throw new Error(`Unknown enhancement: ${type}`);

      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: label,
        history: [...activePage.history,
        { action: label, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast(`${label} applied`, 'success');
    } catch (e) { toast(e.message || 'Enhancement failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleUndo = useCallback(() => {
    if (!activePage || activePage.history.length <= 1) return;
    const newHist = activePage.history.slice(0, -1);
    const last = newHist[newHist.length - 1];
    updatePage(activePageIndex, {
      currentPreview: last.preview, currentPath: last.path,
      imageWidth: last.width, imageHeight: last.height,
      corners: initCorners(last.width, last.height),
      activeFilter: newHist.length > 1 ? newHist[newHist.length - 1].action : null,
      history: newHist,
    });
    toast('Undid last step', 'info');
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleReset = useCallback(() => {
    if (!activePage) return;
    const first = activePage.history[0];
    updatePage(activePageIndex, {
      currentPreview: first.preview, currentPath: first.path,
      imageWidth: first.width, imageHeight: first.height,
      corners: initCorners(first.width, first.height),
      activeFilter: null, history: [first],
    });
    toast('Reset to original', 'info');
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleRemovePage = useCallback((index) => {
    setPages(prev => prev.filter((_, i) => i !== index));
    setActivePageIndex(p => Math.max(0, index <= p ? p - 1 : p));
    toast('Page removed', 'info');
  }, []);

  // ── ScanTailor: Fix Orientation ──────────────────────────────────────────
  const handleOrientRotate = useCallback(async (angle) => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await orientRotate(activePage.currentPath, angle);
      const label = angle > 0 ? `Rotate ${angle}° CW` : `Rotate ${Math.abs(angle)}° CCW`;
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        corners: initCorners(data.width, data.height),
        activeFilter: label,
        history: [...activePage.history,
        { action: label, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast(label + ' applied', 'success');
      return data;
    } catch (e) { toast(e.message || 'Rotation failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── Apply Rotation to ALL pages ─────────────────────────────────────────
  const handleOrientRotateAll = useCallback(async (angle) => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    const label = angle > 0 ? `Rotate ${angle}° CW` : `Rotate ${Math.abs(angle)}° CCW`;
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === label) continue; // Skip if already applied
        const data = await orientRotate(snap[i].currentPath, angle);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          corners: initCorners(data.width, data.height),
          activeFilter: label,
          history: [...snap[i].history,
            { action: label, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`${label} applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch rotation failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── ScanTailor: Split Page ────────────────────────────────────────────────
  const handleSplitDetect = useCallback(async () => {
    if (!activePage) return null;
    try {
      const res = await splitDetect(activePage.currentPath);
      return res;
    } catch (e) { toast(e.message || 'Split detect failed', 'error'); return null; }
  }, [activePage, toast]);

  const handleSpineDetect = useCallback(async () => {
    if (!activePage) return null;
    try {
      const res = await splitDetectSpine(activePage.currentPath);
      return res;
    } catch (e) { toast(e.message || 'Spine detect failed', 'error'); return null; }
  }, [activePage, toast]);

  // handleSplitApply — geometry-first when pageLayout available, pixel-crop fallback
  const handleSplitApply = useCallback(async ({
    splitX, layoutType, contentX1 = 0, contentX2 = null,
    selectedSide = null, boundary = null, cutter = null,
    pageLayout = null,
  } = {}) => {
    if (!activePage) return;
    setLoading(true);
    const imgW = activePage.imageWidth || 1000;
    const cx2 = contentX2 ?? 1;
    const cx1Px = Math.round((contentX1 ?? 0) * imgW);
    const cx2Px = Math.round(cx2 * imgW);
    const spxPx = Math.round((splitX ?? 0.5) * imgW);
    try {
      const data = await splitApply(
        activePage.currentPath,
        spxPx,
        layoutType,
        cx1Px,
        cx2Px,
        selectedSide ?? 'both',
        pageLayout,          // ← geometry-first path when present
      );

      const clearSplit = () => {
        setSplitXRatio(null); setSplitOverlayData(null);
        setSplitBoundary(null); setSplitCutter(null); setSplitPageLayout(null);
      };

      if (layoutType === 'single_cut' || (layoutType === 'two_pages' && selectedSide === 'left')) {
        const lp = data.left_page;
        if (!lp) { toast('No left page returned', 'error'); return; }
        updatePage(activePageIndex, {
          currentPreview: lp.preview, currentPath: lp.image_path,
          imageWidth: lp.width, imageHeight: lp.height,
          corners: initCorners(lp.width, lp.height),
          activeFilter: layoutType === 'single_cut' ? 'Crop to Content' : 'Split (Left)',
          history: [...activePage.history,
          {
            action: layoutType === 'single_cut' ? 'Crop to Content' : 'Split (Left)',
            preview: lp.preview, path: lp.image_path, width: lp.width, height: lp.height
          }],
        });
        clearSplit();
        toast(layoutType === 'single_cut' ? 'Cropped to content area' : 'Left page extracted', 'success');

      } else if (layoutType === 'two_pages' && selectedSide === 'right') {
        const rp = data.right_page;
        if (!rp) { toast('No right page returned', 'error'); return; }
        updatePage(activePageIndex, {
          currentPreview: rp.preview, currentPath: rp.image_path,
          imageWidth: rp.width, imageHeight: rp.height,
          corners: initCorners(rp.width, rp.height),
          activeFilter: 'Split (Right)',
          history: [...activePage.history,
          {
            action: 'Split (Right)', preview: rp.preview, path: rp.image_path,
            width: rp.width, height: rp.height
          }],
        });
        clearSplit();
        toast('Right page extracted', 'success');

      } else {
        // Full split — both pages
        const lp = data.left_page;
        if (lp) {
          updatePage(activePageIndex, {
            currentPreview: lp.preview, currentPath: lp.image_path,
            imageWidth: lp.width, imageHeight: lp.height,
            corners: initCorners(lp.width, lp.height),
            activeFilter: data.used_geometry ? 'Split ✓ Geometry' : 'Split (Left)',
            history: [...activePage.history,
            {
              action: 'Split (Left)', preview: lp.preview, path: lp.image_path,
              width: lp.width, height: lp.height
            }],
          });
        }
        const rp = data.right_page;
        if (rp) {
          setPages(prev => {
            const arr = [...prev];
            arr.splice(activePageIndex + 1, 0, {
              id: Date.now() + Math.random(),
              filename: (activePage.filename || 'page') + '_right',
              originalPreview: rp.preview, currentPreview: rp.preview,
              originalPath: rp.image_path, currentPath: rp.image_path,
              imageWidth: rp.width, imageHeight: rp.height,
              corners: initCorners(rp.width, rp.height),
              activeFilter: 'Split (Right)',
              history: [{
                action: 'Split (Right)', preview: rp.preview, path: rp.image_path,
                width: rp.width, height: rp.height
              }],
            });
            return arr;
          });
        }
        clearSplit();
        const mode = data.used_geometry ? ' (geometry-first)' : '';
        toast(`Page split${mode} — right half inserted after current page`, 'success');
      }
    } catch (e) { toast(e.message || 'Split failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── ScanTailor: Adjust Corners ──────────────────────────────────────────
  const handleCornersDetect = useCallback(async () => {
    if (!activePage) return false;
    setLoading(true);
    try {
      const data = await cornersDetect(activePage.currentPath, cornerDetectMethod);
      if (data.corners) {
        const formattedCorners = data.corners.map(c => ({ x: c[0], y: c[1] }));
        updatePage(activePageIndex, { corners: formattedCorners });
        toast('Corners detected — drag to adjust', 'success');
        return true;
      }
      return false;
    } catch (e) {
      toast(e.message || 'Corner detection failed', 'error');
      return false;
    } finally {
      setLoading(false);
    }
  }, [activePage, activePageIndex, updatePage, toast, cornerDetectMethod]);

  const handleCornersApply = useCallback(async () => {
    if (!activePage || !activePage.corners) return;
    setLoading(true);
    try {
      const formattedCorners = activePage.corners.map(c => [
        Math.round(c.x !== undefined ? c.x : c[0]),
        Math.round(c.y !== undefined ? c.y : c[1])
      ]);
      const data = await cornersApply(activePage.currentPath, formattedCorners);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        corners: initCorners(data.width, data.height),
        activeFilter: 'Adjust Corners',
        history: [...activePage.history,
        { action: 'Adjust Corners', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Corner warp applied', 'success');
    } catch (e) {
      toast(e.message || 'Corner warp failed', 'error');
    } finally {
      setLoading(false);
    }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleCornersApplyAll = useCallback(async () => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === 'Adjust Corners') continue;

        // Auto detect corners for each page using the currently selected method
        const detectData = await cornersDetect(snap[i].currentPath, cornerDetectMethod);
        if (!detectData.corners) continue;

        // Apply corners
        const applyData = await cornersApply(snap[i].currentPath, detectData.corners);
        updatePage(i, {
          currentPreview: applyData.preview, currentPath: applyData.image_path,
          imageWidth: applyData.width, imageHeight: applyData.height,
          corners: initCorners(applyData.width, applyData.height),
          activeFilter: 'Adjust Corners',
          history: [...snap[i].history,
          { action: 'Adjust Corners', preview: applyData.preview, path: applyData.image_path, width: applyData.width, height: applyData.height }],
        });
      }
      toast(`Corners detected and applied to all ${snap.length} pages`, 'success');
    } catch (e) {
      toast(e.message || 'Batch corner apply failed', 'error');
    } finally {
      setBatchLoading(false); setBatchProgress(0);
    }
  }, [pages, updatePage, toast, cornerDetectMethod]);

  // ── ScanTailor: Deskew (handlers reused from existing; just used by new Deskew.jsx) ──
  const handleDeskewAutoReturn = useCallback(async () => {
    if (!activePage) return null;
    setLoading(true);
    try {
      const data = await deskewAuto(activePage.currentPath);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: `Auto Deskew (${data.detected_angle?.toFixed(2) || '?'}°)`,
        history: [...activePage.history,
        { action: 'Auto Deskew', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast(`Deskew applied — corrected ${data.detected_angle?.toFixed(2) || '?'}°`, 'success');
      return data;
    } catch (e) { toast(e.message || 'Deskew failed', 'error'); return null; }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleDeskewManualApply = useCallback(async (angle) => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await deskewManual(activePage.currentPath, angle);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        activeFilter: `Manual Deskew (${angle}°)`,
        history: [...activePage.history,
        { action: `Manual Deskew (${angle}°)`, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      setDeskewManualAngle(0); // reset preview angle after apply
      toast(`Manual deskew applied (${angle}°)`, 'success');
    } catch (e) { toast(e.message || 'Manual deskew failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── Apply Deskew Auto to ALL pages ───────────────────────────────────────
  const handleDeskewAutoAll = useCallback(async () => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter?.startsWith('Auto Deskew')) continue; // Skip if already applied
        const data = await deskewAuto(snap[i].currentPath);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          activeFilter: `Auto Deskew (${data.detected_angle?.toFixed(2) || '?'}°)`,
          history: [...snap[i].history,
            { action: 'Auto Deskew', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Auto-deskew applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch deskew failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── Apply Manual Deskew to ALL pages ─────────────────────────────────────
  const handleDeskewManualAll = useCallback(async (angle) => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        const label = `Manual Deskew (${angle}°)`;
        if (snap[i].activeFilter === label) continue; // Skip if already applied
        const data = await deskewManual(snap[i].currentPath, angle);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          activeFilter: `Manual Deskew (${angle}°)`,
          history: [...snap[i].history,
            { action: `Manual Deskew (${angle}°)`, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Manual deskew (${angle}°) applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch manual deskew failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── ScanTailor: Margins ───────────────────────────────────────────────
  const handleMarginsDetect = useCallback(async () => {
    if (!activePage) return null;
    try {
      const res = await marginsDetect(activePage.currentPath);
      return res;
    } catch (e) { toast(e.message || 'Margin detection failed', 'error'); return null; }
  }, [activePage, toast]);

  const handleMarginsApply = useCallback(async (top, bottom, left, right) => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await marginsApply(activePage.currentPath, top, bottom, left, right);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        corners: initCorners(data.width, data.height),
        activeFilter: `Margins (T:${top} B:${bottom} L:${left} R:${right}mm)`,
        history: [...activePage.history,
        { action: 'Margins Applied', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Margins applied', 'success');
    } catch (e) { toast(e.message || 'Margins failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── Apply Margins to ALL pages ────────────────────────────────────
  const handleMarginsApplyAll = useCallback(async (top, bottom, left, right) => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        const label = `Margins (T:${top} B:${bottom} L:${left} R:${right}mm)`;
        if (snap[i].activeFilter === label) continue; // Skip if already applied
        const data = await marginsApply(snap[i].currentPath, top, bottom, left, right);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          corners: initCorners(data.width, data.height),
          activeFilter: `Margins (T:${top} B:${bottom} L:${left} R:${right}mm)`,
          history: [...snap[i].history,
            { action: 'Margins Applied', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Margins applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch margins failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── ScanTailor: Select Content ─────────────────────────────────────────
  const handleContentDetect = useCallback(async () => {
    if (!activePage) return null;
    try {
      const res = await contentDetect(activePage.currentPath);
      if (res && res.preview) {
        updatePage(activePageIndex, {
          currentPreview: res.preview,
          contentRect: res.content_rect || null,
        });
      }
      return res;
    } catch (e) { toast(e.message || 'Content detection failed', 'error'); return null; }
  }, [activePage, activePageIndex, updatePage, toast]);

  const handleContentClearPreview = useCallback(() => {
    if (!activePage) return;
    const lastHistory = activePage.history[activePage.history.length - 1];
    if (lastHistory) {
      updatePage(activePageIndex, {
        currentPreview: lastHistory.preview,
        contentRect: null,
      });
    }
  }, [activePage, activePageIndex, updatePage]);

  const handleContentApply = useCallback(async () => {
    if (!activePage) return;
    setLoading(true);
    try {
      const data = await contentApply(activePage.currentPath, activePage.contentRect);
      updatePage(activePageIndex, {
        currentPreview: data.preview, currentPath: data.image_path,
        imageWidth: data.width, imageHeight: data.height,
        corners: initCorners(data.width, data.height),
        activeFilter: 'Select Content',
        contentRect: null, // Clear box after applying
        history: [...activePage.history,
        { action: 'Select Content', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
      });
      toast('Content selection applied', 'success');
    } catch (e) { toast(e.message || 'Content selection failed', 'error'); }
    finally { setLoading(false); }
  }, [activePage, activePageIndex, updatePage, toast]);

  // ── Apply Content Selection to ALL pages ──────────────────────────────
  const handleContentApplyAll = useCallback(async () => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === 'Select Content') continue; // Skip if already applied
        const data = await contentApply(snap[i].currentPath, snap[i].contentRect || null);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          corners: initCorners(data.width, data.height),
          activeFilter: 'Select Content',
          contentRect: null,
          history: [...snap[i].history,
            { action: 'Select Content', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Content selection applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch content selection failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── Apply Auto Dewarp to ALL pages ───────────────────────────────────
  const handleDewarpAutoAll = useCallback(async () => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === 'Auto Dewarp (ML)') continue; // Skip if already applied
        const data = await dewarpImageAuto(snap[i].currentPath);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          activeFilter: 'Auto Dewarp (ML)',
          history: [...snap[i].history,
            { action: 'Auto Dewarp (ML)', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Auto dewarp applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch auto dewarp failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── Apply Poly Dewarp to ALL pages ───────────────────────────────────
  const handleDewarpPolyAll = useCallback(async () => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === 'Poly Dewarp (B-spline)') continue; // Skip if already applied
        // Batch mode runs fully automatic (no custom curves)
        const data = await dewarpImagePoly(snap[i].currentPath, null);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          activeFilter: 'Poly Dewarp (B-spline)',
          history: [...snap[i].history,
            { action: 'Poly Dewarp (B-spline)', preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`Poly dewarp applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch poly dewarp failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  // ── Apply Enhancement to ALL pages ───────────────────────────────────
  const handleEnhanceAll = useCallback(async (type, outputFormat) => {
    const snap = pages;
    setBatchLoading(true); setBatchProgress(0);
    const label = outputFormat === 'color' ? 'Color Enhancement' : 'B/W Enhancement';
    try {
      for (let i = 0; i < snap.length; i++) {
        setBatchProgress(i + 1);
        if (snap[i].activeFilter === label) continue; // Skip if already applied
        let data;
        if (type === 'otsu') data = await enhanceOtsu(snap[i].currentPath, outputFormat);
        else if (type === 'adaptive') data = await enhanceAdaptive(snap[i].currentPath, outputFormat);
        else throw new Error(`Unknown enhancement: ${type}`);
        updatePage(i, {
          currentPreview: data.preview, currentPath: data.image_path,
          imageWidth: data.width, imageHeight: data.height,
          activeFilter: label,
          history: [...snap[i].history,
            { action: label, preview: data.preview, path: data.image_path, width: data.width, height: data.height }],
        });
      }
      toast(`${label} applied to all ${snap.length} pages`, 'success');
    } catch (e) { toast(e.message || 'Batch enhancement failed', 'error'); }
    finally { setBatchLoading(false); setBatchProgress(0); }
  }, [pages, updatePage, toast]);

  const handleExportPdf = useCallback(async () => {
    if (!pages.length) return;
    setLoading(true);
    try {
      const blob = await exportPdf(pages.map(p => p.currentPath));
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'akshar_output.pdf';
      a.click();
      URL.revokeObjectURL(url);
      toast(`PDF exported (${pages.length} pages)`, 'success');
    } catch (e) { toast(e.message || 'PDF export failed', 'error'); }
    finally { setLoading(false); }
  }, [pages, toast]);

  // -----------------------------------------------------------------------
  // Proceed to QA from workbench
  // -----------------------------------------------------------------------

  const handleProceedToQALocal = useCallback(async () => {
    setShowQAModal(false);
    if (!pages.length) return;
    setLoading(true);
    toast('Assembling PDF & indexing…', 'info');
    try {
      // 1. Assemble processed pages into PDF
      const assembled = await pipelineConvertToPdf(
        pages.map(p => p.currentPath),
        'akshar_processed.pdf',
      );

      const docId = documentId || `doc_${Date.now()}`;
      setDocumentId(docId);
      setPdfPath(assembled.pdf_path);

      // 2. Run OCR on assembled PDF
      toast('Running OCR on processed document…', 'info');
      const ocrResult = await pipelineRunOcr(assembled.pdf_path, '');
      setOcrData(ocrResult);

      // 3. Index into ChromaDB
      toast('Building vector index…', 'info');
      await pipelineIndex(docId, ocrResult.blocks);

      toast('Document indexed — ready for Q&A!', 'success');
      setMessages([{
        role: 'assistant',
        content: `✅ **Document processed & indexed** — ${ocrResult.block_count} text blocks from ${pages.length} page${pages.length > 1 ? 's' : ''}. Ask me anything!`,
        refs: [],
      }]);
      setStageNav(STAGE_QA);
    } catch (e) {
      console.error(e);
      toast(e.message || 'Failed to proceed to QA', 'error');
    } finally {
      setLoading(false);
    }
  }, [pages, documentId, toast]);

  const handleProceedToQASarvam = useCallback(async () => {
    if (!pages.length) return;
    setShowQAModal(false);
    setLoading(true);
    toast('Assembling PDF & extracting via Sarvam AI API…', 'info');
    try {
      // 1. Assemble processed pages into PDF
      const assembled = await pipelineConvertToPdf(
        pages.map(p => p.currentPath),
        'akshar_processed.pdf',
      );

      const docId = documentId || `doc_${Date.now()}`;
      setDocumentId(docId);
      setPdfPath(assembled.pdf_path);

      // 2. Run OCR on assembled PDF using Sarvam API
      toast('Parsing document with Sarvam AI…', 'info');
      // Pass the API key explicitly
      const ocrResult = await pipelineRunOcrSarvam(assembled.pdf_path, 'sk_p8ojpatk_S4tt4EfsJQJPgW6evwiWkgMY');
      setOcrData(ocrResult);

      // 3. Index into ChromaDB
      toast('Building vector index…', 'info');
      await pipelineIndex(docId, ocrResult.blocks);

      toast('Document indexed — ready for Q&A!', 'success');
      setMessages([{
        role: 'assistant',
        content: `✅ **Document parsed & indexed via Sarvam API** — ${ocrResult.block_count} text blocks from ${pages.length} page${pages.length > 1 ? 's' : ''}. Ask me anything!`,
        refs: [],
      }]);
      setStageNav(STAGE_QA);
    } catch (e) {
      console.error(e);
      toast(e.message || 'Failed to proceed to QA (Sarvam API)', 'error');
    } finally {
      setLoading(false);
    }
  }, [pages, documentId, toast]);

  // -----------------------------------------------------------------------
  // QA handlers
  // -----------------------------------------------------------------------

  const handleQuery = useCallback(async (query) => {
    if (!query.trim() || !documentId) return;

    // Add user message immediately
    setMessages(prev => [...prev, { role: 'user', content: query }]);

    // Show thinking
    const thinkId = `think_${Date.now()}`;
    setMessages(prev => [...prev, { role: 'thinking', id: thinkId }]);

    try {
      const result = await pipelineQuery(query, documentId, '');

      // Replace thinking with answer
      setMessages(prev =>
        prev
          .filter(m => m.id !== thinkId)
          .concat({
            role: 'assistant',
            content: result.answer,
            refs: result.references || [],
          })
      );
    } catch (e) {
      setMessages(prev =>
        prev
          .filter(m => m.id !== thinkId)
          .concat({
            role: 'assistant',
            content: `❌ ${e.message || 'Query failed. Check your GROQ_API_KEY.'}`,
            refs: [],
          })
      );
    }
  }, [documentId]);

  const handleShowRefs = useCallback((refs) => {
    setActiveRefs(refs);
    setShowRefs(true);
  }, []);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="app">
      {/* ── WORKFLOW CHOICE MODAL ──────────────────────────────────── */}
      {showWorkflowChoice && (
        <div className="wf-modal-overlay" onClick={() => setShowWorkflowChoice(false)}>
          <div className="wf-modal" onClick={(e) => e.stopPropagation()}>
            <div className="wf-modal-header">
              <div className="wf-modal-icon">📋</div>
              <h2 className="wf-modal-title">Choose Your Workflow</h2>
              <p className="wf-modal-subtitle">
                Scanned PDF detected — <strong>{pendingPdfData?.filename}</strong>
              </p>
            </div>

            <div className="wf-options">
              {/* Option 1 — Direct QA */}
              <button
                className="wf-option-card wf-option-qa"
                onClick={handleDirectQA}
                disabled={uploadLoading}
                id="workflow-direct-qa"
              >
                <div className="wf-option-icon">💬</div>
                <div className="wf-option-body">
                  <h3 className="wf-option-title">Go to Question Answering</h3>
                  <p className="wf-option-desc">
                    Skip preprocessing — send PDF directly to OCR &amp; QA pipeline.
                    Best for <strong>readable scanned PDFs</strong>.
                  </p>
                  <div className="wf-option-flow">
                    <span className="wf-flow-badge wf-flow-skip">Module 1</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-skip">Module 1</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-active">Module 3</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-active">Module 4</span>
                  </div>
                </div>
                <div className="wf-option-arrow">→</div>
              </button>

              {/* Option 2 — Full Preprocessing */}
              <button
                className="wf-option-card wf-option-preprocess"
                onClick={handlePreprocessing}
                disabled={uploadLoading}
                id="workflow-preprocessing"
              >
                <div className="wf-option-icon">🔧</div>
                <div className="wf-option-body">
                  <h3 className="wf-option-title">Go to Preprocessing</h3>
                  <p className="wf-option-desc">
                    Full pipeline — extract pages, clean up in workbench, then QA.
                    Best for <strong>noisy or skewed scans</strong>.
                  </p>
                  <div className="wf-option-flow">
                    <span className="wf-flow-badge wf-flow-active">Module 1</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-active">Module 1</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-active">Module 3</span>
                    <span className="wf-flow-arrow">→</span>
                    <span className="wf-flow-badge wf-flow-active">Module 4</span>
                  </div>
                </div>
                <div className="wf-option-arrow">→</div>
              </button>
            </div>

            {uploadLoading && (
              <div className="wf-loading">
                <div className="spinner" style={{ width: 24, height: 24, borderWidth: 2 }} />
                <span>Processing…</span>
              </div>
            )}
          </div>
        </div>
      )}
      {/* Status toast */}
      {statusMsg && (
        <div className={`status-toast status-toast-${statusMsg.type}`}>
          {statusMsg.type === 'success' && '✓ '}
          {statusMsg.type === 'error' && '✕ '}
          {statusMsg.type === 'info' && 'ℹ '}
          {statusMsg.text}
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <svg className="header-logo" viewBox="0 0 36 36" fill="none">
            <defs>
              <linearGradient id="logoG" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#00e5ff" />
                <stop offset="100%" stopColor="#7c4dff" />
              </linearGradient>
            </defs>
            <rect x="2" y="2" width="32" height="32" rx="8" stroke="url(#logoG)" strokeWidth="2.5" />
            <path d="M10 12h16M10 18h10M10 24h13" stroke="url(#logoG)" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <div>
            <div className="header-name">PROJECT AKSHAR</div>
            <div className="header-tagline">AI Document Intelligence Platform</div>
          </div>
        </div>

        {/* Stage Navigation */}
        <div className="stage-nav">
          <StageStep num={1} label="Upload" active={stage === STAGE_UPLOAD} completed={stage !== STAGE_UPLOAD}
            onClick={stage !== STAGE_UPLOAD ? () => setStage(STAGE_UPLOAD) : null} />
          <div className="stage-divider" />
          <StageStep num={2} label="Workbench" active={stage === STAGE_WORKBENCH}
            completed={stage === STAGE_QA}
            onClick={pages.length > 0 && stage === STAGE_QA ? () => setStage(STAGE_WORKBENCH) : null} />
          <div className="stage-divider" />
          <StageStep num={3} label="Q&A" active={stage === STAGE_QA}
            completed={false} />
        </div>

        <div className="header-right">
          {docType && (
            <span className={`badge badge-${docType === 'digital' ? 'digital' : 'scanned'}`}>
              {docType === 'digital' ? '📄 Digital PDF' : '🖼 Scanned'}
            </span>
          )}
        </div>
      </header>

      {/* ── STAGE 1: UPLOAD ─────────────────────────────────────────── */}
      {stage === STAGE_UPLOAD && (
        <div className="upload-stage">
          <div className="upload-stage-inner">

            {/* Logo */}
            <div className="upload-brand">
              Akshar AI<span className="upload-brand-dot">.</span>
            </div>

            {/* Title */}
            <h1 className="upload-hero-title">
              Welcome to <span className="upload-hero-highlight">Akshar AI</span>
            </h1>

            {/* Subtitle */}
            <p className="upload-hero-sub">
              Upload your document (PDF or Image) to start Question Answering.
            </p>

            {/* Upload Zone */}
            <UploadZone onFilesSelect={handleUpload} loading={uploadLoading} />

            {pipelineStep && (
              <div style={{ marginTop: 24 }}>
                <PipelineProgress step={pipelineStep} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── STAGE 2: IMAGE WORKBENCH ─────────────────────────────────── */}
      {stage === STAGE_WORKBENCH && (
        <div className="editor-stage">
          {/* Page strip */}
          {pages.length > 0 && (
            <PageStrip
              pages={pages}
              activeIndex={activePageIndex}
              onSelect={setActivePageIndex}
              onRemove={handleRemovePage}
              onAddMore={() => addInputRef.current?.click()}
            />
          )}

          <div className="editor-layout">
            {/* Left — Canvas */}
            <div className="editor-main">
              <div className="canvas-toolbar">
                <div className="canvas-toolbar-left">
                  <span className="canvas-title">Document Editor</span>
                  {activePage && (
                    <span className="canvas-subtitle">
                      — Page {activePageIndex + 1} of {pages.length}
                      {activePage.activeFilter && ` · ${activePage.activeFilter}`}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => setStageNav(STAGE_UPLOAD)}>← Back</button>
                  <div style={{ position: 'relative' }} ref={qaDropdownRef}>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => setShowQAModal(!showQAModal)}
                      disabled={!pages.length || loading}
                    >
                      {loading ? <span className="spinner" /> : null}
                      Proceed to Q&A →
                    </button>
                    {showQAModal && (
                      <div style={{
                        position: 'absolute',
                        top: '100%',
                        right: 0,
                        marginTop: 8,
                        backgroundColor: 'var(--color-bg)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 8,
                        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                        width: 250,
                        zIndex: 100,
                        padding: 12,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8
                      }}>
                        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-text-2)', marginBottom: 4 }}>Select QA Engine</div>
                        <button
                          className="btn btn-outline btn-sm"
                          style={{ justifyContent: 'flex-start', textAlign: 'left', padding: '8px 12px', height: 'auto' }}
                          onClick={handleProceedToQASarvam}
                        >
                          <div>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Sarvam ai API</div>
                            <div style={{ fontSize: 11, color: 'var(--color-text-3)', fontWeight: 400, whiteSpace: 'normal' }}>
                              Cloud Document Parse API
                            </div>
                          </div>
                        </button>
                        <button
                          className="btn btn-outline btn-sm"
                          style={{ justifyContent: 'flex-start', textAlign: 'left', padding: '8px 12px', height: 'auto' }}
                          onClick={handleProceedToQALocal}
                        >
                          <div>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>LP & POCR</div>
                            <div style={{ fontSize: 11, color: 'var(--color-text-3)', fontWeight: 400, whiteSpace: 'normal' }}>
                              Local Pipeline
                            </div>
                          </div>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="canvas-wrapper">
                {activePage ? (
                  <div
                    className={`canvas-rotate-wrapper${deskewOverlay && deskewMode === 'manual' ? ' deskew-active' : ''}`}
                    style={deskewOverlay && deskewMode === 'manual'
                      ? { transform: `rotate(${deskewManualAngle}deg)` }
                      : {}}
                  >
                    <CanvasEditor
                      imageSrc={activePage.currentPreview}
                      imageWidth={activePage.imageWidth}
                      imageHeight={activePage.imageHeight}
                      corners={(perspectiveOpen || cornersPanelOpen) ? activePage.corners : null}
                      onCornersChange={corners => updatePage(activePageIndex, { corners })}
                      isWarpMode={false}
                      dewarpGrid={dewarpOpen ? (dewarpMode === 'poly' ? polyGrid : dewarpGrid) : null}
                      onDewarpGridChange={dewarpMode === 'poly' ? setPolyGrid : setDewarpGrid}
                      contentRect={contentPanelOpen ? (activePage.contentRect || null) : null}
                      onContentRectChange={contentRect => updatePage(activePageIndex, { contentRect })}
                    />
                  </div>
                ) : (
                  <div style={{ color: 'var(--color-text-3)', fontSize: 14 }}>No pages loaded</div>
                )}

                {/* Deskew arc overlay */}
                {deskewOverlay && deskewMode === 'manual' && (
                  <DeskewArcOverlay
                    angle={deskewManualAngle}
                    onAngleChange={setDeskewManualAngle}
                  />
                )}

                {/* SplitBoundaryOverlay — only visible while the Split Pages panel is open */}
                {splitPanelOpen && splitXRatio != null && (
                  <SplitBoundaryOverlay
                    boundary={splitBoundary}
                    onBoundaryChange={(bnd) => {
                      setSplitBoundary(bnd);
                      setSplitOverlayData(prev => prev ? { ...prev, boundary: bnd } : prev);
                    }}
                    cutter={splitCutter}
                    onCutterChange={(ct) => {
                      setSplitCutter(ct);
                      const avgX = (ct.top.x + ct.bottom.x) / 2;
                      setSplitXRatio(avgX);
                      setSplitOverlayData(prev => prev ? { ...prev, cutter: ct, splitXRatio: avgX } : prev);
                    }}
                    xRatio={splitXRatio}
                    onXRatioChange={(r) => {
                      setSplitXRatio(r);
                      setSplitOverlayData(prev => prev ? { ...prev, splitXRatio: r } : prev);
                    }}
                    layoutType={splitOverlayData?.layoutType ?? 'two_pages'}
                    selectedSide={splitOverlayData?.selectedSide ?? null}
                    showSplitLine={
                      splitOverlayData?.splitMode === 'manual' ||
                      splitOverlayData?.layoutType === 'two_pages' ||
                      splitOverlayData?.layoutType === 'single_cut'
                    }
                    onSideSelect={(side) =>
                      setSplitOverlayData(prev => prev ? { ...prev, selectedSide: side } : prev)
                    }
                  />
                )}

                {loading && (
                  <div className="loader-overlay" style={{ borderRadius: 0 }}>
                    <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
                  </div>
                )}
              </div>

              {/* Before/After preview */}
              {activePage && (
                <Preview
                  originalSrc={activePage.originalPreview}
                  processedSrc={activePage.activeFilter ? activePage.currentPreview : null}
                  activeFilter={activePage.activeFilter}
                />
              )}
            </div>

            {/* Right — Controls sidebar */}
            <div className="editor-sidebar">
              <input ref={addInputRef} type="file" accept="image/*" multiple
                style={{ display: 'none' }}
                onChange={async (e) => {
                  const files = Array.from(e.target.files || []);
                  if (files.length) await handleUpload(files);
                  e.target.value = '';
                }} />

              {activePage && (
                <div className="st-panels-block">
                  {/* Panel 1 — Fix Orientation */}
                  <StPanel number={1} title="Fix Orientation" colorClass="panel-red">
                    <FixOrientation
                      onRotate={handleOrientRotate}
                      onRotateAll={handleOrientRotateAll}
                      loading={loading}
                      batchLoading={batchLoading}
                      batchProgress={batchProgress}
                      hasImage={!!activePage.currentPath}
                      pageCount={pages.length}
                    />
                  </StPanel>

                  {/* Panel 2 — Adjust Corners */}
                  <StPanel number={2} title="Adjust Corners" colorClass="panel-purple"
                    open={cornersPanelOpen}
                    onToggle={(isOpen) => {
                      setCornersPanelOpen(isOpen);
                      if (!isOpen) {
                        // Reset corners to default when panel closes
                        if (activePage) {
                          updatePage(activePageIndex, {
                            corners: initCorners(activePage.imageWidth, activePage.imageHeight),
                          });
                        }
                      }
                    }}
                  >
                    <AdjustCorners
                      onDetect={handleCornersDetect}
                      onApply={handleCornersApply}
                      onApplyAll={handleCornersApplyAll}
                      loading={loading}
                      batchLoading={batchLoading}
                      batchProgress={batchProgress}
                      hasImage={!!activePage.currentPath}
                      pageCount={pages.length}
                      cornersActive={cornersPanelOpen}
                      onCornersActiveChange={setCornersPanelOpen}
                      method={cornerDetectMethod}
                      onMethodChange={setCornerDetectMethod}
                    />
                  </StPanel>

                  {/* Panel 3 — Split Pages */}
                  <StPanel number={3} title="Split Pages" colorClass="panel-orange"
                    open={splitPanelOpen}
                    onToggle={(isOpen) => {
                      setSplitPanelOpen(isOpen);
                      if (!isOpen) {
                        // Hide overlay when panel is collapsed (same as perspective)
                        setSplitXRatio(null);
                        setSplitOverlayData(null);
                        setSplitBoundary(null);
                        setSplitCutter(null);
                        setSplitPageLayout(null);
                      }
                    }}
                  >
                    <SplitPage
                      onDetect={handleSplitDetect}
                      onSpineDetect={handleSpineDetect}
                      onApply={handleSplitApply}
                      loading={loading}
                      hasImage={!!activePage.currentPath}
                      panelOpen={splitPanelOpen}
                      splitXRatio={splitXRatio}
                      onSplitXChange={(r) => {
                        setSplitXRatio(r);
                      }}
                      imageWidth={activePage.imageWidth}
                      pageCount={pages.length}
                      onDetectResult={(result) => {
                        setSplitOverlayData(result);
                        setSplitBoundary(result.boundary ?? null);
                        setSplitCutter(result.cutter ?? null);
                        setSplitPageLayout(result.pageLayout ?? null);
                        const lt = result.layoutType;
                        setSplitXRatio(lt !== 'single_uncut' ? result.splitXRatio : null);
                      }}
                    />
                  </StPanel>

                  {/* Panel 4 — Deskew */}
                  <StPanel number={4} title="Deskew" colorClass="panel-yellow">
                    <Deskew
                      angle={deskewManualAngle}
                      onDeskewAuto={handleDeskewAutoReturn}
                      onDeskewManual={handleDeskewManualApply}
                      onDeskewAutoAll={handleDeskewAutoAll}
                      onDeskewManualAll={handleDeskewManualAll}
                      onModeChange={(mode) => {
                        setDeskewMode(mode);
                        setDeskewOverlay(mode === 'manual');
                        if (mode !== 'manual') setDeskewManualAngle(0);
                      }}
                      onAngleChange={setDeskewManualAngle}
                      loading={loading}
                      batchLoading={batchLoading}
                      batchProgress={batchProgress}
                      hasImage={!!activePage.currentPath}
                      pageCount={pages.length}
                    />
                  </StPanel>

                  {/* Panel 5 — Select Content */}
                  <StPanel number={5} title="Select Content" colorClass="panel-teal"
                    open={contentPanelOpen}
                    onToggle={(isOpen) => {
                      setContentPanelOpen(isOpen);
                      if (isOpen && activePage && !activePage.contentRect) {
                        // Default bounding box with premium 10% margins
                        updatePage(activePageIndex, {
                          contentRect: [
                            Math.round(activePage.imageWidth * 0.1),
                            Math.round(activePage.imageHeight * 0.1),
                            Math.round(activePage.imageWidth * 0.9),
                            Math.round(activePage.imageHeight * 0.9)
                          ]
                        });
                      }
                      if (!isOpen && activePage) {
                        handleContentClearPreview();
                      }
                    }}
                  >
                    <ContentSelection
                      onDetect={handleContentDetect}
                      onApply={handleContentApply}
                      onApplyAll={handleContentApplyAll}
                      onClearPreview={handleContentClearPreview}
                      loading={loading}
                      batchLoading={batchLoading}
                      batchProgress={batchProgress}
                      hasImage={!!activePage.currentPath}
                      pageCount={pages.length}
                    />
                  </StPanel>

                  {/* Panel 6 — Margins */}
                  <StPanel number={6} title="Margins" colorClass="panel-green">
                    <Margins
                      onDetect={handleMarginsDetect}
                      onApply={handleMarginsApply}
                      onApplyAll={handleMarginsApplyAll}
                      loading={loading}
                      batchLoading={batchLoading}
                      batchProgress={batchProgress}
                      hasImage={!!activePage.currentPath}
                      pageCount={pages.length}
                    />
                  </StPanel>
                </div>
              )}

              {activePage && (
                <Controls
                  onTransform={handleTransform}
                  onDewarp={handleDewarp}
                  onDewarpAuto={handleDewarpAuto}
                  onDewarpPoly={handleDewarpPoly}
                  onDewarpAutoAll={handleDewarpAutoAll}
                  onDewarpPolyAll={handleDewarpPolyAll}
                  onEnhance={handleEnhance}
                  onEnhanceAll={handleEnhanceAll}
                  onReset={handleReset}
                  onUndo={handleUndo}
                  onExportPdf={handleExportPdf}
                  onAnalyzeGrid={handleAnalyzeGrid}
                  dewarpGridActive={dewarpMode === 'poly' ? !!polyGrid : !!dewarpGrid}
                  onEstimatePolyCurves={handleEstimatePolyCurves}
                  polyGridActive={!!polyGrid}
                  dewarpMode={dewarpMode}
                  onDewarpModeChange={setDewarpMode}
                  loading={loading}
                  batchLoading={batchLoading}
                  batchProgress={batchProgress}
                  hasImage={!!activePage.currentPath}
                  hasTransformed={!!activePage.activeFilter}
                  canUndo={activePage.history.length > 1}
                  pageCount={pages.length}
                  perspectiveOpen={perspectiveOpen}
                  onPerspectiveToggle={setPerspectiveOpen}
                  dewarpOpen={dewarpOpen}
                  onDewarpToggle={setDewarpOpen}
                />
              )}



              {/* History */}
              {activePage && activePage.history.length > 1 && (
                <div className="history-panel">
                  <div className="controls-header">
                    <span style={{ fontSize: 13 }}>⟳</span>
                    <span className="controls-title">History</span>
                  </div>
                  <div className="history-list">
                    {activePage.history.map((item, idx) => (
                      <div key={idx}
                        className={`history-item ${idx === activePage.history.length - 1 ? 'active' : ''}`}>
                        <span className="history-dot" />
                        <span>{item.action}</span>
                        {idx === activePage.history.length - 1 && idx > 0 && (
                          <span className="history-badge">now</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── STAGE 3: Q&A ────────────────────────────────────────────── */}
      {stage === STAGE_QA && (
        <div className={`qa-stage ${!showRefs ? 'no-refs' : ''}`}>
          <QAChat
            messages={messages}
            onQuery={handleQuery}
            onShowRefs={handleShowRefs}
            onBack={() => setStageNav(pages.length > 0 ? STAGE_WORKBENCH : STAGE_UPLOAD)}
            documentId={documentId}
            ocrData={ocrData}
            pages={pages}
            pdfPath={pdfPath}
          />
          {showRefs && (
            <ReferencesPanel
              refData={activeRefs}
              pdfUrl={pdfPath ? `http://localhost:8000/static/${pdfPath.includes('processed') ? 'processed' : 'uploads'}/${pdfPath.split(/[/\\]/).pop()}` : null}
              onClose={() => setShowRefs(false)}
            />
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------
// Stage Step badge (inline component)
// -----------------------------------------------------------------------

function StageStep({ num, label, active, completed, onClick }) {
  return (
    <div
      className={`stage-step ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${onClick ? 'clickable' : ''}`}
      onClick={onClick || undefined}
    >
      <span className="stage-num">{completed ? '✓' : num}</span>
      {label}
    </div>
  );
}

// -----------------------------------------------------------------------
// Pipeline progress indicator (inline)
// -----------------------------------------------------------------------

const STEPS = [
  { id: 'upload', label: 'Uploading file' },
  { id: 'detect', label: 'Detecting document type' },
  { id: 'ocr', label: 'Running OCR extraction' },
  { id: 'extract', label: 'Extracting PDF pages' },
  { id: 'index', label: 'Building vector index' },
];

function PipelineProgress({ step }) {
  const currentIdx = STEPS.findIndex(s => s.id === step);
  return (
    <div className="pipeline-steps">
      {STEPS.filter(s => s.id !== 'extract' || step === 'extract').map((s, i) => (
        <div key={s.id}
          className={`pipeline-step ${i === currentIdx ? 'active' : i < currentIdx ? 'done' : ''}`}>
          <span className="step-icon">
            {i < currentIdx ? '✓' : i === currentIdx ? <span className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} /> : i + 1}
          </span>
          {s.label}
        </div>
      ))}
    </div>
  );
}

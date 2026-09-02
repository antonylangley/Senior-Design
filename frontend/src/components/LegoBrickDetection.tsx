import { useRef, useState } from 'react'
import { AlertTriangle, Bug, Check, Clipboard, Download, ImageUp, LoaderCircle } from 'lucide-react'
import { api } from '../services/api'
import type { LegoDetectResponse } from '../types/lego'
import { LegoOverlay } from './LegoOverlay'
import { formatLegoSummary, serializeLegoResult } from '../utils/legoExport'

const readImage = (file: File): Promise<string> => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(String(reader.result))
  reader.onerror = () => reject(new Error('Could not read the selected image.'))
  reader.readAsDataURL(file)
})

export function LegoBrickDetection() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [imageData, setImageData] = useState<string | null>(null)
  const [result, setResult] = useState<LegoDetectResponse | null>(null)
  const [rows, setRows] = useState(10)
  const [columns, setColumns] = useState(10)
  const [debug, setDebug] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exportFeedback, setExportFeedback] = useState<{ message: string; success: boolean } | null>(null)

  const copyText = async (text: string, label: string) => {
    setExportFeedback(null)
    try {
      if (!navigator.clipboard) throw new Error('Clipboard access is unavailable.')
      await navigator.clipboard.writeText(text)
      setExportFeedback({ message: `${label} copied to clipboard.`, success: true })
    } catch {
      setExportFeedback({ message: `Failed to copy ${label.toLowerCase()}.`, success: false })
    }
  }

  const downloadJson = () => {
    if (!result) return
    const url = URL.createObjectURL(new Blob([serializeLegoResult(result)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `lego-detection-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    link.click()
    URL.revokeObjectURL(url)
    setExportFeedback({ message: 'JSON download started.', success: true })
  }

  const processFile = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Choose a JPEG, PNG, or other browser-supported image.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await readImage(file)
      setImageData(data)
      setResult(await api.detectLegoBricks(data, rows, columns, debug))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'LEGO detection failed.')
    } finally {
      setLoading(false)
    }
  }

  const rerun = async () => {
    if (!imageData) return
    setLoading(true)
    setError(null)
    try {
      setResult(await api.detectLegoBricks(imageData, rows, columns, debug))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'LEGO detection failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="lego-workspace" aria-label="LEGO brick detection workspace">
      <section className="panel lego-main-panel">
        <div className="section-heading"><ImageUp size={20} aria-hidden="true" /><h2>LEGO Brick Detection</h2></div>
        <p className="muted lego-intro">Upload a separated, studs-up brick layout photographed from above. Geometry is measured locally with OpenCV.</p>

        <input ref={inputRef} className="visually-hidden" type="file" accept="image/*" onChange={(event) => { const file = event.target.files?.[0]; if (file) void processFile(file) }} />
        <button
          type="button"
          className={`upload-dropzone${dragging ? ' dragging' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void processFile(file) }}
        >
          <ImageUp size={30} aria-hidden="true" />
          <strong>Drop an image here or choose a file</strong>
          <span>Original files stay local except for this backend request; LEGO detection does not call OpenAI.</span>
        </button>

        <div className="lego-controls">
          <label>Grid rows<input type="number" min="1" max="50" value={rows} onChange={(event) => setRows(Math.max(1, Math.min(50, Number(event.target.value))))} /></label>
          <label>Grid columns<input type="number" min="1" max="50" value={columns} onChange={(event) => setColumns(Math.max(1, Math.min(50, Number(event.target.value))))} /></label>
          <label className="toggle-control"><input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} /><Bug size={17} aria-hidden="true" /> Debug</label>
          <button className="secondary-button" type="button" disabled={!imageData || loading} onClick={() => void rerun()}>Apply settings</button>
        </div>

        {loading ? <p className="loading-line"><LoaderCircle className="spinner" size={20} aria-hidden="true" /> Detecting bricks…</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
        {result?.warning ? <p className="warning-text">{result.warning}</p> : null}
        {result ? <LegoOverlay imageData={result.processed_image_data} result={result} /> : null}

        {result ? (
          <p className={`rectification-status${result.rectification.active ? ' active' : ''}`}>
            Rectification: {result.rectification.active ? `Active (${result.rectification.method})` : 'Inactive'}
          </p>
        ) : null}

        {debug && result?.debug ? (
          <section className="debug-grid" aria-label="Debug images">
            <figure><img src={imageData ?? ''} alt="Raw uploaded frame" /><figcaption>Raw image</figcaption></figure>
            {result.debug.rectified_view ? <figure><img src={result.debug.rectified_view} alt="Rectified workspace view" /><figcaption>Rectified view</figcaption></figure> : null}
            <figure><img src={result.debug.segmentation_mask} alt="Binary segmentation mask" /><figcaption>Segmentation mask</figcaption></figure>
            <figure><img src={result.debug.components} alt="Raw and final oriented boundaries" /><figcaption>Raw box (cyan) and final boundary (green)</figcaption></figure>
            <figure><img src={result.debug.studs} alt="Detected stud centers" /><figcaption>Detected studs</figcaption></figure>
            <figure><img src={result.debug.pose_refinement} alt="Canonical LEGO pose refinement diagnostics" /><figcaption>Pose fit: predicted yellow, matched green, rejected red</figcaption></figure>
          </section>
        ) : null}
      </section>

      <aside className="panel lego-results-panel">
        <div className="section-heading"><h2>Results</h2></div>
        <p className="detected-total">Detected bricks: <strong>{result?.bricks.length ?? 0}</strong></p>
        <p className="coordinate-note">Grid coordinates below are zero-based.</p>
        <div className="export-actions">
          <button type="button" className="secondary-button" disabled={!result} onClick={() => { if (result) void copyText(serializeLegoResult(result), 'JSON') }}><Clipboard size={16} aria-hidden="true" />Copy JSON</button>
          <button type="button" className="secondary-button" disabled={!result} onClick={() => { if (result) void copyText(formatLegoSummary(result), 'Summary') }}><Clipboard size={16} aria-hidden="true" />Copy Summary</button>
          <button type="button" className="secondary-button" disabled={!result} onClick={downloadJson}><Download size={16} aria-hidden="true" />Download JSON</button>
        </div>
        {exportFeedback ? (
          <p className={`export-feedback${exportFeedback.success ? '' : ' failed'}`} role="status">
            {exportFeedback.success ? <Check size={15} aria-hidden="true" /> : <AlertTriangle size={15} aria-hidden="true" />}
            {exportFeedback.message}
          </p>
        ) : null}
        {result?.bricks.map((brick) => (
          <article className="brick-result" key={brick.id}>
            <h3>Brick #{brick.id}</h3>
            <dl>
              <div><dt>Color</dt><dd>{brick.color}</dd></div>
              <div><dt>Dimensions</dt><dd>{brick.dimensions ? `${brick.dimensions.studs_x}x${brick.dimensions.studs_y}` : 'Unresolved'}</dd></div>
              <div><dt>Stud count</dt><dd>{brick.stud_count}</dd></div>
              <div><dt>Center</dt><dd>({brick.center_px.x.toFixed(1)}, {brick.center_px.y.toFixed(1)})</dd></div>
              <div><dt>Normalized</dt><dd>({brick.center_normalized.x.toFixed(3)}, {brick.center_normalized.y.toFixed(3)})</dd></div>
              <div><dt>Yaw</dt><dd>{brick.angle_degrees.toFixed(1)}°</dd></div>
              <div><dt>Grid</dt><dd>[{brick.grid_position.row},{brick.grid_position.column}]</dd></div>
              <div><dt>Confidence</dt><dd>{Math.round(brick.confidence * 100)}%</dd></div>
              <div><dt>Size confidence</dt><dd>{Math.round(brick.dimension_confidence * 100)}%</dd></div>
              <div><dt>Pose</dt><dd>{brick.pose_source === 'lego_model_fit' ? 'Model fit' : 'Contour fallback'}</dd></div>
              <div><dt>Pose confidence</dt><dd>{Math.round(brick.pose_confidence * 100)}%</dd></div>
            </dl>
          </article>
        ))}
        {!result ? <p className="muted">Results will appear after an image is processed.</p> : null}
      </aside>
    </section>
  )
}

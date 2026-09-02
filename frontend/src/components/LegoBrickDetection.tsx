import { useRef, useState } from 'react'
import { Bug, ImageUp, LoaderCircle } from 'lucide-react'
import { api } from '../services/api'
import type { LegoDetectResponse } from '../types/lego'
import { LegoOverlay } from './LegoOverlay'

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
        {imageData && result ? <LegoOverlay imageData={imageData} result={result} /> : null}

        {debug && result?.debug ? (
          <section className="debug-grid" aria-label="Debug images">
            <figure><img src={imageData ?? ''} alt="Raw uploaded frame" /><figcaption>Raw image</figcaption></figure>
            <figure><img src={result.debug.segmentation_mask} alt="Binary segmentation mask" /><figcaption>Segmentation mask</figcaption></figure>
            <figure><img src={result.debug.components} alt="Detected components and oriented rectangles" /><figcaption>Components and pose</figcaption></figure>
            <figure><img src={result.debug.studs} alt="Detected stud centers" /><figcaption>Detected studs</figcaption></figure>
          </section>
        ) : null}
      </section>

      <aside className="panel lego-results-panel">
        <div className="section-heading"><h2>Results</h2></div>
        <p className="detected-total">Detected bricks: <strong>{result?.bricks.length ?? 0}</strong></p>
        <p className="coordinate-note">Grid coordinates below are zero-based.</p>
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
            </dl>
          </article>
        ))}
        {!result ? <p className="muted">Results will appear after an image is processed.</p> : null}
      </aside>
    </section>
  )
}

import { Check, Pencil, Save } from 'lucide-react'
import { useState } from 'react'
import type { RobotIntent, SketchRecognitionResult } from '../types/intent'
import { buildDrawingIntent, cleanLabel, formatConfidence } from '../utils/intent'

interface RecognitionReviewProps {
  recognition: SketchRecognitionResult
  onVerified: (intent: RobotIntent) => void
}

export function RecognitionReview({ recognition, onVerified }: RecognitionReviewProps) {
  const [editing, setEditing] = useState(false)
  const [editedLabel, setEditedLabel] = useState(recognition.primary_label)
  const [editError, setEditError] = useState<string | null>(null)

  const verify = (label: string) => {
    const cleaned = cleanLabel(label)
    if (!cleaned) {
      setEditError('Enter an object label.')
      return
    }
    setEditError(null)
    onVerified(buildDrawingIntent(recognition, cleaned))
  }

  return (
    <section className="recognition-block" aria-label="Recognition result">
      <p className="label-small">Recognition</p>
      <p className="guess-line">
        I think you drew a: <strong>{recognition.primary_label.toUpperCase()}</strong>
      </p>
      <p className="confidence-line">Confidence: {formatConfidence(recognition.confidence)}</p>
      <p className="evidence-line">{recognition.reasoning_summary}</p>

      {editing ? (
        <div className="edit-row">
          <label htmlFor="correct-label">Correct label</label>
          <div className="inline-control">
            <input
              id="correct-label"
              value={editedLabel}
              onChange={(event) => setEditedLabel(event.target.value)}
            />
            <button type="button" className="secondary-button" onClick={() => verify(editedLabel)}>
              <Save size={18} aria-hidden="true" />
              Save
            </button>
          </div>
          {editError ? <p className="error-text">{editError}</p> : null}
        </div>
      ) : (
        <div className="action-row">
          <button type="button" className="primary-button" onClick={() => verify(recognition.primary_label)}>
            <Check size={20} aria-hidden="true" />
            Correct
          </button>
          <button type="button" className="secondary-button" onClick={() => setEditing(true)}>
            <Pencil size={18} aria-hidden="true" />
            Edit
          </button>
        </div>
      )}
    </section>
  )
}

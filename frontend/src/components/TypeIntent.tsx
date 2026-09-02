import { SendHorizonal } from 'lucide-react'
import type { FormEvent } from 'react'
import { useState } from 'react'
import { api } from '../services/api'
import type { RobotIntent } from '../types/intent'

interface TypeIntentProps {
  onVerifiedIntent: (intent: RobotIntent) => void
}

export function TypeIntent({ onVerifiedIntent }: TypeIntentProps) {
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed) {
      setError('Enter an assembly intent.')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await api.normalizeTextIntent(trimmed)
      onVerifiedIntent(response.intent)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Text intent normalization failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel type-panel" aria-label="Typed intent">
      <div className="section-heading">
        <h2>Typed Input</h2>
      </div>
      <form onSubmit={submit} className="type-form">
        <label htmlFor="typed-intent">Intent</label>
        <textarea
          id="typed-intent"
          rows={5}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Build me a duck"
        />
        <button type="submit" className="primary-button" disabled={loading}>
          <SendHorizonal size={18} aria-hidden="true" />
          {loading ? 'Normalizing' : 'Use Intent'}
        </button>
        {error ? <p className="error-text">{error}</p> : null}
      </form>
    </section>
  )
}

import { CheckCircle2 } from 'lucide-react'
import type { RobotIntent } from '../types/intent'

interface VerifiedIntentProps {
  intent: RobotIntent | null
}

export function VerifiedIntent({ intent }: VerifiedIntentProps) {
  return (
    <aside className="panel verified-panel" aria-label="Verified intent">
      <div className="section-heading">
        <CheckCircle2 size={20} aria-hidden="true" />
        <h2>Verified Intent</h2>
      </div>

      {intent ? (
        <>
          <p className="intent-line">{intent.normalized_intent}</p>
          <pre className="payload-preview">{JSON.stringify(intent, null, 2)}</pre>
        </>
      ) : (
        <p className="muted">No verified intent yet.</p>
      )}
    </aside>
  )
}

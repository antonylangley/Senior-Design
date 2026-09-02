import { useState } from 'react'
import { CameraScanner } from './components/CameraScanner'
import { ModeTabs } from './components/ModeTabs'
import { LegoBrickDetection } from './components/LegoBrickDetection'
import { TypeIntent } from './components/TypeIntent'
import { VerifiedIntent } from './components/VerifiedIntent'
import type { InputMode, RobotIntent } from './types/intent'

function App() {
  const [mode, setMode] = useState<InputMode>('draw')
  const [verifiedIntent, setVerifiedIntent] = useState<RobotIntent | null>(null)

  const handleModeChange = (nextMode: InputMode) => {
    setMode(nextMode)
    setVerifiedIntent(null)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Autonomous assembly stage 1</p>
          <h1>Robot Intent Interface</h1>
        </div>
        <ModeTabs activeMode={mode} onModeChange={handleModeChange} />
      </header>

      {mode === 'lego' ? <LegoBrickDetection /> : (
        <section className="workspace-grid" aria-label="Intent input workspace">
          {mode === 'draw' ? <CameraScanner onVerifiedIntent={setVerifiedIntent} /> : <TypeIntent onVerifiedIntent={setVerifiedIntent} />}
          <VerifiedIntent intent={verifiedIntent} />
        </section>
      )}
    </main>
  )
}

export default App

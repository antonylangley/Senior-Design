import { Camera, Keyboard } from 'lucide-react'
import type { InputMode } from '../types/intent'

interface ModeTabsProps {
  activeMode: InputMode
  onModeChange: (mode: InputMode) => void
}

export function ModeTabs({ activeMode, onModeChange }: ModeTabsProps) {
  return (
    <div className="mode-tabs" role="tablist" aria-label="Input mode">
      <button
        type="button"
        role="tab"
        aria-selected={activeMode === 'draw'}
        className={activeMode === 'draw' ? 'active' : ''}
        onClick={() => onModeChange('draw')}
      >
        <Camera size={18} aria-hidden="true" />
        Draw
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={activeMode === 'type'}
        className={activeMode === 'type' ? 'active' : ''}
        onClick={() => onModeChange('type')}
      >
        <Keyboard size={18} aria-hidden="true" />
        Type
      </button>
    </div>
  )
}

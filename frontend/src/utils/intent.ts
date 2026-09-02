import type { RobotIntent, SketchRecognitionResult } from '../types/intent'

const leadingPhrases = [
  'a drawing of ',
  'drawing of ',
  'a sketch of ',
  'sketch of ',
  'a picture of ',
  'picture of ',
  'the ',
  'an ',
  'a ',
]

export function cleanLabel(value: string): string {
  let label = value.trim().toLowerCase().replace(/\s+/g, ' ')
  label = label.replace(/^[\s.,!?;:"'`()[\]{}]+|[\s.,!?;:"'`()[\]{}]+$/g, '')
  for (const phrase of leadingPhrases) {
    if (label.startsWith(phrase)) {
      label = label.slice(phrase.length).trim()
      break
    }
  }
  return label
}

function articleFor(label: string): 'a' | 'an' {
  const first = label.trim()[0]
  return first && 'aeiou'.includes(first) ? 'an' : 'a'
}

export function buildNormalizedIntent(label: string): string {
  const cleaned = cleanLabel(label)
  return `Build ${articleFor(cleaned)} ${cleaned}`
}

export function buildDrawingIntent(
  recognition: SketchRecognitionResult,
  verifiedLabel: string,
): RobotIntent {
  const label = cleanLabel(verifiedLabel)
  return {
    input_type: 'drawing',
    raw_ai_label: recognition.primary_label,
    verified_label: label,
    normalized_intent: buildNormalizedIntent(label),
    human_verified: true,
    confidence: recognition.confidence,
  }
}

export function formatConfidence(confidence: number | null | undefined): string {
  if (confidence === null || confidence === undefined) {
    return 'n/a'
  }
  return `${Math.round(confidence * 100)}%`
}

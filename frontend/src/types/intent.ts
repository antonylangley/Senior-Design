export type InputMode = 'draw' | 'type' | 'lego'

export interface Point {
  x: number
  y: number
}

export interface RecognitionAlternative {
  label: string
  confidence: number
}

export interface SketchRecognitionResult {
  primary_label: string
  normalized_intent: string
  confidence: number
  alternatives: RecognitionAlternative[]
  reasoning_summary: string
}

export interface RobotIntent {
  input_type: 'drawing' | 'text'
  raw_ai_label?: string | null
  verified_label: string
  normalized_intent: string
  human_verified: boolean
  confidence?: number | null
}

export interface DetectResponse {
  paper_detected: boolean
  confidence: number
  corners: Point[] | null
  image_width: number
  image_height: number
  warning: string | null
}

export interface ScanResponse {
  paper_detected: boolean
  used_full_frame: boolean
  confidence: number
  corners: Point[] | null
  processed_image_data: string
  recognition: SketchRecognitionResult | null
  recognition_error: string | null
  warning: string | null
}

export interface TextIntentResponse {
  intent: RobotIntent
}

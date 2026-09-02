import type {
  DetectResponse,
  ScanResponse,
  TextIntentResponse,
} from '../types/intent'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function postJson<TResponse>(
  path: string,
  body: Record<string, unknown>,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) {
        message = payload.detail
      }
    } catch {
      // Keep the HTTP status as the fallback message.
    }
    throw new Error(message)
  }

  return (await response.json()) as TResponse
}

export const api = {
  detectPaper(imageData: string): Promise<DetectResponse> {
    return postJson('/api/detect', { image_data: imageData })
  },

  scanDrawing(
    imageData: string,
    useFullFrameOnFailure: boolean,
  ): Promise<ScanResponse> {
    return postJson('/api/scan', {
      image_data: imageData,
      use_full_frame_on_failure: useFullFrameOnFailure,
    })
  },

  normalizeTextIntent(text: string): Promise<TextIntentResponse> {
    return postJson('/api/text-intent', { text })
  },
}

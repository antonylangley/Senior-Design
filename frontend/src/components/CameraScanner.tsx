import { AlertTriangle, Camera, FileImage, ScanLine } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { DetectResponse, RobotIntent, ScanResponse } from '../types/intent'
import { formatConfidence } from '../utils/intent'
import { RecognitionReview } from './RecognitionReview'

interface CameraScannerProps {
  onVerifiedIntent: (intent: RobotIntent) => void
}

interface CapturedFrame {
  imageData: string
}

type CameraState = 'starting' | 'ready' | 'denied' | 'unavailable' | 'error'

function cameraErrorMessage(error: unknown): { state: CameraState; message: string } {
  if (!navigator.mediaDevices?.getUserMedia) {
    return { state: 'unavailable', message: 'This browser does not expose webcam capture.' }
  }

  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return { state: 'denied', message: 'Webcam permission was denied.' }
    }
    if (error.name === 'NotFoundError' || error.name === 'OverconstrainedError') {
      return { state: 'unavailable', message: 'No compatible webcam was found.' }
    }
  }

  return { state: 'error', message: 'The webcam could not be started.' }
}

export function CameraScanner({ onVerifiedIntent }: CameraScannerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const detectionInFlight = useRef(false)
  const [cameraState, setCameraState] = useState<CameraState>('starting')
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [detection, setDetection] = useState<DetectResponse | null>(null)
  const [scanResponse, setScanResponse] = useState<ScanResponse | null>(null)
  const [scanSource, setScanSource] = useState<string | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraState('unavailable')
        setCameraError('This browser does not expose webcam capture.')
        return
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setCameraState('ready')
        setCameraError(null)
      } catch (error) {
        const result = cameraErrorMessage(error)
        setCameraState(result.state)
        setCameraError(result.message)
      }
    }

    void startCamera()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }, [])

  const captureFrame = useCallback((maxSide: number): CapturedFrame | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return null
    }

    const longest = Math.max(video.videoWidth, video.videoHeight)
    const scale = Math.min(1, maxSide / longest)
    const width = Math.round(video.videoWidth * scale)
    const height = Math.round(video.videoHeight * scale)
    canvas.width = width
    canvas.height = height

    const context = canvas.getContext('2d')
    if (!context) {
      return null
    }

    context.drawImage(video, 0, 0, width, height)
    return { imageData: canvas.toDataURL('image/jpeg', 0.92) }
  }, [])

  useEffect(() => {
    if (cameraState !== 'ready') {
      return undefined
    }

    const pollDetection = async () => {
      if (detectionInFlight.current) {
        return
      }

      const frame = captureFrame(900)
      if (!frame) {
        return
      }

      detectionInFlight.current = true
      try {
        const result = await api.detectPaper(frame.imageData)
        setDetection(result)
      } catch {
        setDetection(null)
      } finally {
        detectionInFlight.current = false
      }
    }

    void pollDetection()
    const intervalId = window.setInterval(() => {
      void pollDetection()
    }, 1200)

    return () => window.clearInterval(intervalId)
  }, [cameraState, captureFrame])

  const runScan = async (useFullFrameOnFailure: boolean) => {
    setScanError(null)
    setScanning(true)
    try {
      const imageData = useFullFrameOnFailure && scanSource ? scanSource : captureFrame(1600)?.imageData
      if (!imageData) {
        throw new Error('Could not capture a webcam frame.')
      }

      setScanSource(imageData)
      const response = await api.scanDrawing(imageData, useFullFrameOnFailure)
      setScanResponse(response)
      if (response.recognition_error) {
        setScanError(response.recognition_error)
      }
    } catch (error) {
      setScanError(error instanceof Error ? error.message : 'Image upload failed.')
    } finally {
      setScanning(false)
    }
  }

  const statusText = detection?.paper_detected
    ? `Paper detected (${formatConfidence(detection.confidence)})`
    : 'Paper boundary not locked'

  return (
    <section className="panel camera-panel" aria-label="Drawing scanner">
      <div className="section-heading">
        <Camera size={20} aria-hidden="true" />
        <h2>Live Camera</h2>
      </div>

      <div className="camera-stage">
        <video ref={videoRef} playsInline muted />
        {detection?.paper_detected && detection.corners ? (
          <svg
            className="paper-overlay"
            viewBox={`0 0 ${detection.image_width} ${detection.image_height}`}
            preserveAspectRatio="xMidYMid meet"
            aria-hidden="true"
          >
            <polygon points={detection.corners.map((point) => `${point.x},${point.y}`).join(' ')} />
          </svg>
        ) : null}
        {cameraState !== 'ready' ? (
          <div className="camera-placeholder">
            {cameraError ?? 'Starting camera'}
          </div>
        ) : null}
      </div>

      <div className="scan-toolbar">
        <span className={detection?.paper_detected ? 'status-pill ok' : 'status-pill'}>
          {statusText}
        </span>
        <button
          type="button"
          className="primary-button scan-button"
          onClick={() => void runScan(false)}
          disabled={cameraState !== 'ready' || scanning}
        >
          <ScanLine size={20} aria-hidden="true" />
          {scanning ? 'Scanning' : 'Scan Drawing'}
        </button>
      </div>

      <canvas ref={canvasRef} className="hidden-canvas" />

      {scanResponse ? (
        <section className="captured-block" aria-label="Captured input">
          <div className="section-heading small">
            <FileImage size={18} aria-hidden="true" />
            <h3>Captured Input</h3>
          </div>
          <img src={scanResponse.processed_image_data} alt="Processed captured drawing" />
          {scanResponse.warning ? (
            <p className="warning-text">
              <AlertTriangle size={17} aria-hidden="true" />
              {scanResponse.warning}
            </p>
          ) : null}
          {!scanResponse.paper_detected && !scanResponse.used_full_frame ? (
            <button
              type="button"
              className="secondary-button"
              onClick={() => void runScan(true)}
              disabled={scanning}
            >
              <FileImage size={18} aria-hidden="true" />
              Submit Full Frame
            </button>
          ) : null}
        </section>
      ) : null}

      {scanError ? <p className="error-text">{scanError}</p> : null}

      {scanResponse?.recognition ? (
        <RecognitionReview recognition={scanResponse.recognition} onVerified={onVerifiedIntent} />
      ) : null}
    </section>
  )
}

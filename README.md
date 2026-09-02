# Robot Intent Interface MVP

This prototype implements the first stage of an autonomous robotic assembly pipeline:

`hand-drawn sketch -> camera -> paper extraction -> AI identification -> human verification -> normalized English intent`

It does not generate LEGO models, plan assembly, control robots, or do motion planning.

It also includes an independent **LEGO Brick Detection** prototype for measuring separated,
studs-up bricks in controlled top-down images. This perception output is not yet connected to
robot control.

## Architecture

- `frontend/`: React, TypeScript, and Vite browser UI.
- `backend/`: FastAPI service with OpenCV image processing and OpenAI vision recognition.
- `backend/app/vision/`: reusable paper detection, corner ordering, perspective warp, and scan enhancement.
- `backend/app/vision/lego.py`: deterministic segmentation, pose, stud-layout, grid, and color logic.
- `backend/app/vision/lego_model.py`: nominal LEGO geometry, robust canonical-lattice fitting,
  ideal physical boundary generation, and scene-scale consistency analysis.
- `backend/app/ai/`: OpenAI Responses API integration.
- `backend/app/models/`: domain models, including `RobotIntent`.
- `backend/app/services/`: scan processing and typed intent normalization.

The frontend sends still frames to the backend. The backend sends only the processed paper scan to the AI model when paper detection succeeds. If paper detection fails, the user can explicitly submit the full frame.

## Installation

From the project root:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install --upgrade pip
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt

cd frontend
npm install
```

On macOS:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt

cd frontend
npm install
```

## Environment Variables

Copy `.env.example` to `.env` in the project root and provide:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_VISION_MODEL=gpt-5.6
BACKEND_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`OPENAI_VISION_MODEL` can be changed to any Responses API model available to your account that supports image input.

## Run Backend

From the project root:

```powershell
backend\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

On macOS:

```bash
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Run Frontend

In another terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## Fresh Clone On MacBook

```bash
git clone https://github.com/antonylangley/Senior-Design.git
cd Senior-Design
cp .env.example .env
```

Edit `.env` and provide your own `OPENAI_API_KEY`.

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt

cd frontend
npm install
npm run dev
```

In a second terminal from the repository root:

```bash
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

To run checks:

```bash
cd backend
.venv/bin/python -m pytest
cd ../frontend
npm run lint
npm test
npm run build
```

## Webcam Demo

1. Open the frontend URL in a browser.
2. Allow webcam access.
3. Place a white sheet of paper in view.
4. The live preview polls the backend for a paper outline.
5. Press `Scan Drawing`.
6. Review the processed scan and recognition.
7. Select `Correct` or `Edit`.
8. The verified `RobotIntent` payload appears in the right panel.

## API Endpoints

- `GET /health`
- `POST /api/detect`: detect paper corners for live preview overlay.
- `POST /api/scan`: detect/warp/enhance the drawing and run recognition only for a scan action.
- `POST /api/recognize`: recognize an already prepared image.
- `POST /api/text-intent`: normalize typed input into the shared intent shape.
- `POST /api/lego/detect`: detect LEGO instances and return pixel geometry, zero-based grid
  positions, stud centers, dimensions when resolved, color, confidence, and optional debug images.

### LEGO perception conventions

- Grid rows and columns are configurable from 1 to 50 and API positions are zero-based.
- A brick is assigned using the center of its oriented rectangle; the grid never splits detection.
- `angle_degrees` is the brick rectangle's long-axis yaw, clockwise in image coordinates and
  normalized to `0 <= angle < 180` because a rectangular axis is directionless.
- Dimension inference scores the supported 2x2, 2x3, and 2x4 sizes using the oriented-box aspect
  ratio, stud count, and stud lattice. `dimension_confidence` and `dimension_source` expose how
  strongly the signals agreed.
- Pixel measurements are produced locally with OpenCV. The LEGO endpoint does not call OpenAI.
- Enable Debug in the UI to inspect the raw image, segmentation mask, components/pose, and studs.

### Canonical LEGO pose refinement

The pose refinement uses the supplied nominal constants directly: 8.0 mm stud pitch, approximately
4.8 mm stud diameter, 9.6 mm body height, and footprints of 15.8x15.8 mm (2x2), 23.8x15.8 mm
(2x3), and 31.8x15.8 mm (2x4). Stud lattices are centered at LEGO-local `(0, 0)` with the long
axis along local `+X`.

Segmentation and `minAreaRect` still produce a rough candidate pose. When enough detected studs
robustly match the selected canonical lattice, a similarity transform estimates image translation,
yaw, and pixels per nominal millimeter. The canonical origin becomes the final center and the
nominal footprint transformed into image space becomes the final green polygon. Missing studs are
allowed, extra candidates are rejected by one-to-one reprojection matching, and least-squares
refinement reports an actual RMS reprojection error. Weak fits explicitly retain the contour center,
yaw, and rectangle as `contour_fallback`.

Reliable per-brick scales contribute to a median/MAD scene consistency estimate. A second pass uses
that scale only when multiple fits agree sufficiently (with slightly different requirements for a
rectified image). `scene_scale_px_per_mm` is an image-space model-fit diagnostic—not calibrated
workcell scale and not a robot/world transformation. The nominal constants supplied to this
prototype must later be validated against the actual physical bricks, camera, and workcell during
robot calibration.

The request may optionally include four workspace corners for a planar perspective warp:

```json
{
  "rectification": {
    "workspace_corners": [
      {"x": 100, "y": 80}, {"x": 1180, "y": 100},
      {"x": 1160, "y": 880}, {"x": 120, "y": 900}
    ],
    "output_width": 1080,
    "output_height": 800
  }
}
```

Without this object, preprocessing is unchanged. The rectification module reserves configuration
boundaries for future camera intrinsics, distortion coefficients, homography persistence, and
pixels-to-millimeters calibration without pretending those values are calibrated today.

## Current Limitations

- Paper detection is tuned for a light printer sheet on a darker desk with reasonable lighting.
- Pencil drawings with very low contrast may need marker or better lighting.
- The full-frame fallback may include desk clutter, which can reduce recognition confidence.
- The app stores no history and has no database.
- Speech input, LEGO design generation, manufacturability checks, assembly graphs, task planning, and robot execution are intentionally out of scope.
- LEGO V1 assumes separated, rectangular, standard-height, studs-up bricks on a simple contrasting
  background under even light. Touching/overlapping or stacked bricks, severe shadows, upside-down
  or sideways pieces, plates and unusual geometry, heavy perspective, and precise millimeter/world
  localization are not reliable yet. Dimension inference is deliberately reported as unresolved
  when the detected stud centers do not form a complete rectangular lattice.

## Later Pipeline Connection

The next layer should consume the verified `RobotIntent` object:

```json
{
  "input_type": "drawing",
  "raw_ai_label": "swan",
  "verified_label": "duck",
  "normalized_intent": "Build a duck",
  "human_verified": true,
  "confidence": 0.64
}
```

That object is the boundary between intent recognition and future LEGO/design/robot planning stages.

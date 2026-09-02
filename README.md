# Robot Intent Interface MVP

This prototype implements the first stage of an autonomous robotic assembly pipeline:

`hand-drawn sketch -> camera -> paper extraction -> AI identification -> human verification -> normalized English intent`

It does not generate LEGO models, plan assembly, control robots, or do motion planning.

## Architecture

- `frontend/`: React, TypeScript, and Vite browser UI.
- `backend/`: FastAPI service with OpenCV image processing and OpenAI vision recognition.
- `backend/app/vision/`: reusable paper detection, corner ordering, perspective warp, and scan enhancement.
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
backend/.venv/bin/python -m pytest
cd frontend
npm run lint
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

## Current Limitations

- Paper detection is tuned for a light printer sheet on a darker desk with reasonable lighting.
- Pencil drawings with very low contrast may need marker or better lighting.
- The full-frame fallback may include desk clutter, which can reduce recognition confidence.
- The app stores no history and has no database.
- Speech input, LEGO design generation, manufacturability checks, assembly graphs, task planning, and robot execution are intentionally out of scope.

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

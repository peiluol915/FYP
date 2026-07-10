# Occluded Face Recognition Frontend

React/Vite interface for the occluded face reconstruction and recognition
backend.

## Scripts

```powershell
npm install
npm run dev
npm run build
npm run lint
```

The app calls `POST /analyze` on local backend ports in this order: `8011`,
`8001`, `8000`. Start the backend before uploading an image:

```powershell
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8011
```

The UI displays the occluded input, detected mask, raw DEGAN output,
mask-restricted reconstruction, top gallery matches, confidence scores, and
runtime diagnostics returned by the backend.

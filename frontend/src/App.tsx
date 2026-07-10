import { useMemo, useState } from 'react';
import { Activity, Eye, Image as ImageIcon, LoaderCircle, ScanFace, Upload } from 'lucide-react';

type RecognitionMatch = {
  name: string;
  similarity: number;
  rank_score?: number;
  confidence: number;
  matched_image_path?: string | null;
};

type AnalyzeResponse = {
  status: string;
  detected_face: boolean;
  is_occluded: boolean;
  occlusion_region: string;
  occlusion_ratio: number;
  predicted_name: string | null;
  similarity: number;
  confidence: number;
  recognition_accepted?: boolean;
  recognition_rejection_reason?: string;
  recognition_similarity_gap?: number;
  recognition_similarity_threshold?: number;
  recognition_margin_threshold?: number;
  reconstructed_predicted_name?: string | null;
  reconstructed_similarity?: number;
  reconstructed_recognition_accepted?: boolean;
  reconstructed_recognition_rejection_reason?: string;
  reconstructed_recognition_similarity_gap?: number;
  fps: number;
  recognition_mode: string;
  recognition_model_name?: string;
  recognition_top_matches?: RecognitionMatch[];
  reconstructed_top_matches?: RecognitionMatch[];
  reconstruction_model_loaded?: boolean;
  refiner_model_loaded?: boolean;
  refiner_requested?: boolean;
  refiner_used?: boolean;
  refiner_applied?: boolean;
  refiner_status?: string;
  recognition_model_loaded?: boolean;
  backend_version?: string;
  occlusion_mask_source?: string;
  display_occlusion_ratio?: number;
  display_outside_max_delta?: number;
  degan_weights_modified?: string;
  refiner_weights_modified?: string;
  input_image_base64: string;
  occlusion_mask_base64?: string;
  raw_reconstructed_image_base64?: string;
  reconstructed_image_base64: string;
  reconstructed_model_view_base64?: string;
  original_image_base64?: string;
  reconstructed_original_image_base64?: string;
};

const BACKEND_PORTS = [8011, 8001, 8000] as const;
const COMPATIBLE_BACKEND_MARKERS = [
  'rectangular-structure-sharp-lowblur-seam3d',
  'robust-skin-tone-panel',
  'conservative-panel-seam-pad',
  'conservative-panel-remnant-cleanup',
  'conservative-panel-mask',
  'direct-panel-mask-with-strap-cleanup',
  'direct-panel-mask-region-only',
  'no-reference-local-skin-tone-highres-display-fix',
  'no-reference-local-skin-tone-rank-score-fix',
  'no-reference-local-skin-tone',
  'no-reference-refiner-status',
  'no-reference-rectangular-panel-gate',
  'no-reference-strong-detail-rescue',
  'no-reference-confidence-gated-detail-rescue',
  'no-reference-detail-rescue',
  'model-only-reconstruction',
  'best-checkpoint-input-recognition',
  'optional-mtr-unet',
  'aligned-reconstruction-mask-fix',
  'aligned-reconstruction-panel-repair',
  'pure-degan-mtr-reconstruction',
  'pure-degan-mtr-refiner-guard',
  'pure-degan-mtr-panel-mask-fix',
  'aligned-landmark-mask-fix',
  'pure-model-reconstruction-blend',
  'training-matched-mask-inference',
  'training-style-reconstruction-view',
  'reconstruction-debug-views',
  'training-blend-output',
];

function backendBaseUrl(port: number) {
  return `http://127.0.0.1:${port}`;
}

function toDataUrl(base64: string | undefined) {
  return base64 ? `data:image/png;base64,${base64}` : null;
}

function metricLabel(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return value.toFixed(digits);
}

function statusLabel(value: string | null | undefined) {
  if (!value) {
    return '--';
  }
  return value
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function recognitionDecisionLabel(accepted: boolean | undefined, reason: string | undefined) {
  if (accepted === undefined) {
    return '--';
  }
  return accepted ? 'High confidence' : `Ambiguous: ${statusLabel(reason)}`;
}

function closedSetPredictionLabel(
  name: string | null | undefined,
  similarity: number | null | undefined,
  accepted: boolean | undefined,
  reason: string | undefined,
) {
  const decision = recognitionDecisionLabel(accepted, reason);
  const suffix = decision !== '--' ? ` | ${decision}` : '';
  return `Closed-set Top-1 | ${name ?? '--'} (${metricLabel(similarity, 4)})${suffix}`;
}

const EMPTY_MATCH: RecognitionMatch = {
  name: '--',
  similarity: Number.NaN,
  rank_score: Number.NaN,
  confidence: Number.NaN,
  matched_image_path: null,
};

async function analyzeWithAvailableBackend(file: File, useRefiner: boolean) {
  let lastError: Error | null = null;

  for (const port of BACKEND_PORTS) {
    const apiBaseUrl = backendBaseUrl(port);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiBaseUrl}/analyze?use_refiner=${useRefiner}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend on port ${port} returned ${response.status}`);
      }

      const payload = (await response.json()) as AnalyzeResponse;
      const backendVersion = payload.backend_version ?? '';
      const isCompatible = COMPATIBLE_BACKEND_MARKERS.some((marker) => backendVersion.includes(marker));
      if (isCompatible || port === BACKEND_PORTS[BACKEND_PORTS.length - 1]) {
        return { payload, apiBaseUrl };
      }

      lastError = new Error(`Backend on port ${port} is still the older version.`);
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(`Unable to reach backend on port ${port}.`);
    }
  }

  throw lastError ?? new Error('Unable to reach a compatible backend.');
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useRefiner, setUseRefiner] = useState(false);
  const [activeApiBaseUrl, setActiveApiBaseUrl] = useState(backendBaseUrl(BACKEND_PORTS[0]));
  const inputMatches = result?.recognition_top_matches ?? [];
  const reconstructedMatches = result?.reconstructed_top_matches ?? [];

  const imageCards = useMemo(
    () => [
      {
        title: 'Input Top-1 Candidate',
        subtitle: result?.original_image_base64
          ? closedSetPredictionLabel(
              result.predicted_name,
              result.similarity,
              result.recognition_accepted,
              result.recognition_rejection_reason,
            )
          : 'Gallery match from the occluded input probe',
        image: toDataUrl(result?.original_image_base64),
      },
      {
        title: 'Reconstructed Top-1 Candidate',
        subtitle: result?.reconstructed_original_image_base64
          ? closedSetPredictionLabel(
              result.reconstructed_predicted_name,
              result.reconstructed_similarity,
              result.reconstructed_recognition_accepted,
              result.reconstructed_recognition_rejection_reason,
            )
          : 'Gallery match from the reconstructed probe',
        image: toDataUrl(result?.reconstructed_original_image_base64),
      },
      {
        title: 'Occluded Input',
        subtitle: 'The uploaded masked or partially occluded face',
        image: toDataUrl(result?.input_image_base64) ?? localPreview,
      },
      {
        title: 'Detected Mask',
        subtitle: 'Binary occlusion region used by the reconstruction pipeline',
        image: toDataUrl(result?.occlusion_mask_base64),
      },
      {
        title: 'Raw DE-GAN',
        subtitle: 'Direct model output before final post-processing/blending',
        image: toDataUrl(result?.raw_reconstructed_image_base64),
      },
      {
        title: 'Region-Only Reconstruction',
        subtitle: 'Only the detected occlusion panel is replaced',
        image: toDataUrl(result?.reconstructed_image_base64),
      },
    ],
    [localPreview, result],
  );

  async function handleUpload(file: File) {
    setSelectedFile(file);
    setLocalPreview(URL.createObjectURL(file));
    setError(null);
    setProcessing(true);
    setResult(null);

    try {
      const { payload, apiBaseUrl } = await analyzeWithAvailableBackend(file, useRefiner);
      setActiveApiBaseUrl(apiBaseUrl);
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to analyze the uploaded image.');
    } finally {
      setProcessing(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="hero-grid">
        <section className="hero-copy">
          <p className="eyebrow">Hybrid OFR Viewer</p>
          <h1>Compare original, occluded, and reconstructed faces in one place.</h1>
          <p className="lead">
            Upload an image from your occluded dataset and the site will call the local backend, reconstruct the face,
            predict the person name, and show the matched gallery image only as a comparison.
          </p>

          <div className="metric-strip">
            <div className="metric-card">
              <span className="metric-label">Detected Face</span>
              <span className="metric-value">{result ? (result.detected_face ? 'Yes' : 'No') : '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Input Prediction</span>
              <span className="metric-value metric-name">{result?.predicted_name ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Reconstructed Prediction</span>
              <span className="metric-value metric-name">{result?.reconstructed_predicted_name ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Input Similarity</span>
              <span className="metric-value">{metricLabel(result?.similarity, 4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Reconstructed Similarity</span>
              <span className="metric-value">{metricLabel(result?.reconstructed_similarity, 4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Input Confidence</span>
              <span className="metric-value">{metricLabel(result?.confidence, 4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Input Decision</span>
              <span className="metric-value metric-name">
                {recognitionDecisionLabel(result?.recognition_accepted, result?.recognition_rejection_reason)}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Reconstructed Decision</span>
              <span className="metric-value metric-name">
                {recognitionDecisionLabel(
                  result?.reconstructed_recognition_accepted,
                  result?.reconstructed_recognition_rejection_reason,
                )}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Top Match Gap</span>
              <span className="metric-value">{metricLabel(result?.recognition_similarity_gap, 4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Identifier</span>
              <span className="metric-value metric-name">{result?.recognition_model_name ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Speed</span>
              <span className="metric-value">{result ? `${metricLabel(result.fps)} FPS` : '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Recognition Mode</span>
              <span className="metric-value metric-name">{result?.recognition_mode ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Occlusion</span>
              <span className="metric-value metric-name">
                {result ? `${result.occlusion_region} (${metricLabel(result.occlusion_ratio, 4)})` : '--'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Reconstruction Weights</span>
              <span className="metric-value">
                {result ? (result.reconstruction_model_loaded === false ? 'No' : 'Yes') : '--'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Backend Version</span>
              <span className="metric-value metric-name">{result?.backend_version ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Mask Source</span>
              <span className="metric-value metric-name">{result?.occlusion_mask_source ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Outside Mask Change</span>
              <span className="metric-value">
                {result ? `${result.display_outside_max_delta ?? '--'} px` : '--'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">DEGAN Weights</span>
              <span className="metric-value metric-name">{result?.degan_weights_modified ?? '--'}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">MTR-UNet Refiner</span>
              <span className="metric-value metric-name">
                {result
                  ? `${result.refiner_model_loaded ? 'Loaded' : 'Missing'} / ${
                      result.refiner_applied ? 'Applied' : statusLabel(result.refiner_status)
                    }`
                  : '--'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Refiner Weights</span>
              <span className="metric-value metric-name">{result?.refiner_weights_modified ?? '--'}</span>
            </div>
          </div>
        </section>

        <section className="panel upload-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Input</p>
              <h2>Analyze an occluded face</h2>
            </div>
            <Activity className={`panel-icon ${processing ? 'is-spinning' : ''}`} />
          </div>

          <label className="upload-dropzone">
            <input
              type="file"
              accept="image/*"
              onClick={(event) => {
                event.currentTarget.value = '';
              }}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  void handleUpload(file);
                }
              }}
            />
            <Upload size={24} />
            <div>
              <p>{selectedFile ? selectedFile.name : 'Choose an occluded image'}</p>
              <span>Best for files from `processed_lfw_partial`, `processed_lfw`, or `processed_lfw_heavy`.</span>
            </div>
          </label>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={useRefiner}
              onChange={(event) => {
                setUseRefiner(event.target.checked);
              }}
            />
            <span>Use MTR-UNet refiner when weights are available</span>
          </label>

          <div className="status-box">
            {processing ? (
              <>
                <LoaderCircle size={18} className="is-spinning" />
                <span>Running reconstruction and identity analysis against the local backend.</span>
              </>
            ) : error ? (
              <span className="error-text">{error}</span>
            ) : (
              <span>Backend endpoint: `{activeApiBaseUrl}/analyze`</span>
            )}
          </div>

          <div className="score-grid">
            <div className="score-box">
              <ScanFace size={18} />
              <div>
                <span className="score-label">Input Similarity</span>
                <strong>{metricLabel(result?.similarity, 4)}</strong>
              </div>
            </div>
            <div className="score-box">
              <Eye size={18} />
              <div>
                <span className="score-label">Reconstructed Similarity</span>
                <strong>{metricLabel(result?.reconstructed_similarity, 4)}</strong>
              </div>
            </div>
          </div>

          <div className="match-grid">
            <div className="match-list">
              <span className="score-label">Top Input Matches</span>
              {(inputMatches.length ? inputMatches : [EMPTY_MATCH]).slice(0, 5).map((match, index) => (
                <div key={`input-${match.name}-${index}`} className="match-row">
                  <span>{index + 1}. {match.name}</span>
                  <strong>{metricLabel(match.rank_score ?? match.similarity, 4)}</strong>
                </div>
              ))}
            </div>
            <div className="match-list">
              <span className="score-label">Top Reconstructed Matches</span>
              {(reconstructedMatches.length ? reconstructedMatches : [EMPTY_MATCH])
                .slice(0, 5)
                .map((match, index) => (
                  <div key={`reconstructed-${match.name}-${index}`} className="match-row">
                    <span>{index + 1}. {match.name}</span>
                    <strong>{metricLabel(match.rank_score ?? match.similarity, 4)}</strong>
                  </div>
                ))}
            </div>
          </div>
        </section>
      </div>

      <section className="panel gallery-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Viewer</p>
            <h2>Side-by-side image comparison</h2>
          </div>
          <ImageIcon className="panel-icon" />
        </div>

        <div className="image-grid">
          {imageCards.map((card) => (
            <article key={card.title} className="image-card">
              <div className="image-card-copy">
                <h3>{card.title}</h3>
                <p>{card.subtitle}</p>
              </div>
              <div className="image-frame">
                {card.image ? (
                  <img src={card.image} alt={card.title} />
                ) : (
                  <div className="image-placeholder">
                    <ImageIcon size={28} />
                    <span>No image loaded yet</span>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

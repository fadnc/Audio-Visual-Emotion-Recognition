const fileInput = document.querySelector('#video-file');
const dropzone = document.querySelector('#dropzone');
const analyzeButton = document.querySelector('#analyze');
const modelSelect = document.querySelector('#model-select');
const message = document.querySelector('#message');
const preview = document.querySelector('#preview');
const selectedClip = document.querySelector('#selected-clip');
const uploadView = document.querySelector('#upload-view');
const cameraView = document.querySelector('#camera-view');
const cameraPreview = document.querySelector('#camera-preview');
const cameraState = document.querySelector('#camera-state');
const cameraStart = document.querySelector('#camera-start');
const cameraStop = document.querySelector('#camera-stop');
let selectedFile = null;
let previewUrl = null;
let cameraStream = null;
let recorder = null;
let capturing = false;
let captureTimer = null;

const titleCase = value => value.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
const percent = value => `${Math.round(value * 100)}%`;
const MODEL_LABELS = {
  fusion_attention_face: 'Face-focused audio + video',
  video_face: 'Face-focused video',
  fusion_attention_tuned: 'Fusion + tuned attention',
  fusion_attention_landmarks: 'Fusion + face landmarks',
};
const modelName = model => model.split('/').pop();
const modelLabel = model => MODEL_LABELS[modelName(model)] ?? titleCase(modelName(model));

async function loadModels() {
  try {
    const response = await fetch('/api/models');
    const data = await response.json();
    const status = document.querySelector('.status');
    document.querySelector('#device-label').textContent = `READY · ${data.device.toUpperCase()}`;
    status.classList.add('online');
    modelSelect.replaceChildren();
    if (!data.models.length) {
      modelSelect.add(new Option('No trained checkpoints found', ''));
      message.textContent = 'Train a model and save its checkpoint under checkpoints/ to enable inference.';
      return;
    }
    const preferredOrder = ['fusion_attention_face', 'video_face', 'video', 'fusion_attention_tuned', 'fusion_attention_landmarks', 'fusion_attention', 'fusion_late', 'audio'];
    const rank = model => {
      const index = preferredOrder.indexOf(modelName(model));
      return index === -1 ? preferredOrder.length : index;
    };
    const availableModels = [...data.models].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    for (const model of availableModels) modelSelect.add(new Option(modelLabel(model), model));
    updateButton();
  } catch {
    document.querySelector('#device-label').textContent = 'SERVER UNAVAILABLE';
    message.textContent = 'Could not reach the local inference server.';
  }
}

function updateButton() { analyzeButton.disabled = !selectedFile || !modelSelect.value; }

function showResult(data) {
  document.querySelector('#empty-state').hidden = true;
  document.querySelector('#result-content').hidden = false;
  document.querySelector('#result-model').textContent = modelLabel(data.model);
  document.querySelector('#prediction').textContent = titleCase(data.prediction);
  document.querySelector('#confidence').textContent = percent(data.confidence);
  document.querySelector('#confidence-bar').style.width = percent(data.confidence);
  const geometrySection = document.querySelector('#geometry-section');
  const geometryList = document.querySelector('#geometry-list');
  geometryList.replaceChildren();
  geometrySection.hidden = !data.facial_geometry;
  if (data.facial_geometry) {
    const labels = {
      eye_openness: 'Eye openness', brow_eye_gap: 'Brow to eye gap',
      mouth_opening: 'Mouth opening', left_lip_corner_lift: 'Left lip corner lift',
      right_lip_corner_lift: 'Right lip corner lift', forehead_height: 'Forehead offset',
      face_width: 'Face width', face_height: 'Face height',
    };
    for (const [key, label] of Object.entries(labels)) {
      const row = document.createElement('div'); row.className = 'geometry-value';
      const name = document.createElement('span'); name.textContent = label;
      const value = document.createElement('strong'); value.textContent = data.facial_geometry[key].toFixed(2);
      row.append(name, value); geometryList.append(row);
    }
  }
  const scores = document.querySelector('#scores');
  scores.replaceChildren();
  for (const item of data.scores) {
    const row = document.createElement('div'); row.className = 'score';
    const name = document.createElement('span'); name.textContent = titleCase(item.emotion);
    const track = document.createElement('span'); track.className = 'score-track';
    const fill = document.createElement('i'); fill.style.width = percent(item.probability); track.append(fill);
    const value = document.createElement('span'); value.className = 'score-value'; value.textContent = percent(item.probability);
    row.append(name, track, value); scores.append(row);
  }
}

async function predictFile(file, live = false) {
  if (!modelSelect.value) throw new Error('Choose a trained model first.');
  const body = new FormData();
  body.append('model', modelSelect.value);
  body.append('file', file);
  if (!live) {
    analyzeButton.disabled = true;
    analyzeButton.querySelector('span').textContent = 'Analyzing';
  }
  message.textContent = live ? 'Analyzing live capture…' : 'Preparing the clip and running inference…';
  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Inference failed.');
    showResult(data);
    const coverage = data.face_coverage == null ? '' : ` Face found in ${percent(data.face_coverage)} of sampled frames.`;
    message.textContent = `${live ? 'Live update' : 'Completed'} locally on ${data.device}.${coverage}`;
    return true;
  } catch (error) {
    message.textContent = error.message;
    return false;
  } finally {
    if (!live) {
      analyzeButton.querySelector('span').textContent = 'Analyze clip';
      updateButton();
    }
  }
}

function setInputMode(mode) {
  const isCamera = mode === 'camera';
  if (!isCamera && capturing) stopCamera();
  uploadView.hidden = isCamera;
  cameraView.hidden = !isCamera;
  analyzeButton.hidden = isCamera;
  document.querySelector('.controls').classList.toggle('camera-active', isCamera);
  const uploadMode = document.querySelector('#upload-mode');
  const cameraMode = document.querySelector('#camera-mode');
  uploadMode.classList.toggle('active', !isCamera);
  cameraMode.classList.toggle('active', isCamera);
  uploadMode.setAttribute('aria-pressed', String(!isCamera));
  cameraMode.setAttribute('aria-pressed', String(isCamera));
}

function captureWindow() {
  if (!capturing || !cameraStream || !modelSelect.value) return;
  const chunks = [];
  try {
    const options = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
      ? { mimeType: 'video/webm;codecs=vp8,opus' } : {};
    recorder = new MediaRecorder(cameraStream, options);
  } catch (error) {
    cameraState.textContent = `Recorder unavailable: ${error.message}`;
    stopCamera();
    return;
  }
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  recorder.onstop = async () => {
    if (chunks.length) {
      const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
      if (blob.size) await predictFile(new File([blob], 'live-capture.webm', { type: blob.type }), true);
    }
    if (capturing) {
      cameraState.textContent = 'Capturing next window…';
      captureWindow();
    }
  };
  recorder.start();
  cameraState.textContent = 'Capturing 3 second window…';
  captureTimer = setTimeout(() => {
    if (recorder && recorder.state === 'recording') recorder.stop();
  }, 3000);
}

async function startCamera() {
  if (!modelSelect.value) { cameraState.textContent = 'Choose a model first.'; return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    cameraState.textContent = 'This browser does not support camera capture.'; return;
  }
  cameraStart.disabled = true;
  cameraState.textContent = 'Requesting camera and microphone…';
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: true,
    });
    cameraPreview.srcObject = cameraStream;
    capturing = true;
    cameraStop.disabled = false;
    cameraState.textContent = 'Camera ready.';
    captureWindow();
  } catch (error) {
    cameraState.textContent = error.name === 'NotAllowedError'
      ? 'Camera or microphone permission was denied.' : `Could not start camera: ${error.message}`;
  } finally {
    cameraStart.disabled = Boolean(cameraStream);
  }
}

function stopCamera() {
  capturing = false;
  clearTimeout(captureTimer);
  if (recorder && recorder.state === 'recording') recorder.stop();
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraStart.disabled = false;
  cameraStop.disabled = true;
  cameraState.textContent = 'Camera off';
}

function setFile(file) {
  if (!file) return;
  const fileName = file.name.toLowerCase();
  const supportedType = file.type === 'video/mp4' || file.type === 'video/webm';
  if (!supportedType && !fileName.endsWith('.mp4') && !fileName.endsWith('.webm')) {
    message.textContent = 'Please choose an MP4 or WebM video.';
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    message.textContent = 'This clip is over the 200 MB upload limit.';
    return;
  }
  selectedFile = file;
  document.querySelector('#file-title').textContent = file.name;
  document.querySelector('#file-hint').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · ready to analyze`;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  selectedClip.hidden = false;
  dropzone.hidden = true;
  document.querySelector('#empty-state').hidden = false;
  document.querySelector('#result-content').hidden = true;
  document.querySelector('#result-model').textContent = 'AWAITING INPUT';
  message.textContent = 'Clip ready. Choose a model and analyze.';
  updateButton();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  preview.pause();
  preview.removeAttribute('src');
  preview.load();
  selectedClip.hidden = true;
  dropzone.hidden = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  document.querySelector('#file-title').textContent = 'Choose a video clip';
  document.querySelector('#file-hint').textContent = 'Drop an MP4 or WebM here or browse your files';
  document.querySelector('#empty-state').hidden = false;
  document.querySelector('#result-content').hidden = true;
  document.querySelector('#result-model').textContent = 'AWAITING INPUT';
  message.textContent = 'Select a clip and a trained checkpoint to begin.';
  updateButton();
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
document.querySelector('#replace-file').addEventListener('click', () => {
  fileInput.value = '';
  fileInput.click();
});
document.querySelector('#remove-file').addEventListener('click', clearFile);
modelSelect.addEventListener('change', updateButton);
for (const eventName of ['dragenter', 'dragover']) dropzone.addEventListener(eventName, event => {
  event.preventDefault(); dropzone.classList.add('dragging');
});
for (const eventName of ['dragleave', 'drop']) dropzone.addEventListener(eventName, event => {
  event.preventDefault(); dropzone.classList.remove('dragging');
});
dropzone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));

analyzeButton.addEventListener('click', () => selectedFile && predictFile(selectedFile));
document.querySelector('.input-modes').addEventListener('click', event => {
  const button = event.target.closest('.mode-button');
  if (button?.id === 'upload-mode') setInputMode('upload');
  if (button?.id === 'camera-mode') setInputMode('camera');
});
cameraStart.addEventListener('click', startCamera);
cameraStop.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);

loadModels();

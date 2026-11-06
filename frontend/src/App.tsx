import { useRef, useState } from 'react'
import { ShieldCheck, ScanFace, Upload, Image as ImageIcon, Activity, FileImage, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react'

type Result = {
  prediction: 'Real' | 'Fake'
  confidence: number
  probabilities: { Real: number; Fake: number }
  original: string
  heatmap: string
  overlay: string
  explanation_note: string
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function selectFile(next?: File) {
    if (!next) return
    if (!next.type.startsWith('image/')) { setError('Please choose a valid image file.'); return }
    if (next.size > 10 * 1024 * 1024) { setError('Maximum upload size is 10 MB.'); return }
    setFile(next)
    setPreview(URL.createObjectURL(next))
    setResult(null)
    setError('')
  }

  async function analyze() {
    if (!file) return
    setLoading(true); setError(''); setResult(null)
    try {
      const body = new FormData()
      body.append('image', file)
      const response = await fetch(`${API_URL}/api/predict`, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Analysis failed.')
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to connect to the DeepGuard API.')
    } finally { setLoading(false) }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-icon"><ShieldCheck size={22}/></div><div><b>DEEPGUARD</b><span>AI MEDIA INTELLIGENCE</span></div></div>
        <div className="side-label">WORKSPACE</div>
        <div className="nav-item active"><ScanFace size={18}/> Deepfake analysis</div>
        <div className="nav-item"><Activity size={18}/> Model insights <span className="soon">SOON</span></div>
        <div className="nav-item"><FileImage size={18}/> Analysis history <span className="soon">SOON</span></div>
        <div className="sidebar-bottom"><div className="status-dot"/> Local inference API<div className="version">BUILD 1.0.0</div></div>
      </aside>

      <main className="main">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <b>Deepfake analysis</b></div><div className="secure"><ShieldCheck size={15}/> PRIVATE SESSION</div></header>
        <section className="hero">
          <div className="eyebrow"><span/> AI-POWERED MEDIA AUTHENTICITY</div>
          <h1>See beyond the <em>image.</em></h1>
          <p>Analyze facial imagery for potential manipulation with Vision Transformer classification and visual explanations.</p>
        </section>

        <section className="analysis-grid">
          <div className="panel upload-panel">
            <div className="panel-heading"><div><span className="step">01</span><h2>Image input</h2></div><span className="muted">JPG · PNG · WEBP</span></div>
            <div className={`dropzone ${preview ? 'has-image' : ''}`} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();selectFile(e.dataTransfer.files[0])}} onClick={()=>inputRef.current?.click()}>
              <input ref={inputRef} type="file" accept="image/*" hidden onChange={e=>selectFile(e.target.files?.[0])}/>
              {preview ? <img src={preview} alt="Selected upload"/> : <><div className="upload-icon"><Upload size={24}/></div><b>Drop an image to analyze</b><span>or <u>browse files</u> from your device</span><small>Maximum file size: 10 MB</small></>}
            </div>
            {file && <div className="file-row"><ImageIcon size={16}/><span>{file.name}</span><small>{(file.size/1024/1024).toFixed(2)} MB</small></div>}
            <button className="analyze-btn" disabled={!file || loading} onClick={analyze}>{loading ? <><Loader2 className="spin" size={17}/> Analyzing image...</> : <><ScanFace size={18}/> Analyze image <span>→</span></>}</button>
            {error && <div className="error"><AlertTriangle size={16}/>{error}</div>}
            <div className="privacy"><ShieldCheck size={15}/> Image is processed by your local API. Avoid uploading sensitive images.</div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading"><div><span className="step">02</span><h2>Detection result</h2></div><span className="muted">MODEL OUTPUT</span></div>
            {!result ? <div className="empty-result"><div className="empty-orbit"><ScanFace size={34}/></div><b>Awaiting image analysis</b><span>Your prediction and explainability report will appear here.</span><div className="empty-line"><i/></div></div> :
              <div className="result-content">
                <div className={`verdict ${result.prediction.toLowerCase()}`}><div className="verdict-icon">{result.prediction === 'Real' ? <CheckCircle2/> : <AlertTriangle/>}</div><div><small>MODEL CLASSIFICATION</small><strong>{result.prediction}</strong></div><span>{result.confidence.toFixed(2)}% confidence</span></div>
                <div className="confidence-head"><b>Class probabilities</b><small>Model confidence · not calibrated authenticity probability</small></div>
                {(['Fake','Real'] as const).map(k=><div className="prob-row" key={k}><div><span>{k}</span><b>{result.probabilities[k].toFixed(2)}%</b></div><div className="track"><i className={k.toLowerCase()} style={{width:`${result.probabilities[k]}%`}}/></div></div>)}
                <div className="xai-title"><span>03</span><div><b>Explainable AI</b><small>Grad-CAM++ visual explanation</small></div></div>
                <div className="image-results">
                  <figure><img src={result.original} alt="Original image"/><figcaption>INPUT</figcaption></figure>
                  <figure><img src={result.heatmap} alt="Grad-CAM++ heatmap"/><figcaption>HEATMAP</figcaption></figure>
                  <figure><img src={result.overlay} alt="Grad-CAM++ overlay"/><figcaption>OVERLAY</figcaption></figure>
                </div>
                <div className="disclaimer"><AlertTriangle size={15}/><span>{result.explanation_note}</span></div>
              </div>
            }
          </div>
        </section>
        <footer><span>DEEPGUARD AI <i>·</i> TRANSFORMER-BASED DETECTION</span><span>EXPLAINABILITY IS NOT FORENSIC PROOF</span></footer>
      </main>
    </div>
  )
}

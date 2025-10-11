import React, { useEffect, useState } from 'react'
import api from '../lib/api'

const severityColor = (stage) => {
  if(!stage) return 'text-gray-700'
  const s = stage.toLowerCase()
  if(s.includes('no')) return 'text-green-600'
  if(s.includes('mild')) return 'text-yellow-600'
  if(s.includes('moderate')) return 'text-orange-600'
  return 'text-red-600'
}

export default function Results(){
  const [result, setResult] = useState(null)
  const [opacity, setOpacity] = useState(0.6)
  const [showHeatmap, setShowHeatmap] = useState(true)

  useEffect(()=>{
    api.get('/api/results/').then(r=>{
      const rows = r.data.results || []
      if(rows.length) setResult(rows[0])
    }).catch(()=>{})
  },[])

  if(!result) return (
    <div className="max-w-2xl mx-auto bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4">Latest Result</h2>
      <p>No results yet — try uploading a fundus image.</p>
    </div>
  )

  return (
    <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4">Latest Result</h2>
      <div className="flex gap-6">
        <div className="w-1/2">
          <div className="relative rounded overflow-hidden bg-black">
            <img src={result.image_url} alt="fundus" className="w-full h-auto block" />
            {result.heatmap_url && showHeatmap && (
              <img
                src={result.heatmap_url}
                alt="heatmap"
                className="absolute top-0 left-0 w-full h-full object-cover"
                style={{ opacity, mixBlendMode: 'multiply', pointerEvents: 'none' }}
              />
            )}
          </div>
        </div>
        <div className="w-1/2">
          <h3 className={`text-lg font-semibold ${severityColor(result.predicted_stage)}`}>Stage: {result.predicted_stage}</h3>
          <p className="mb-2">Confidence: <span className="font-medium">{Math.round(result.confidence*100)}%</span></p>
          <div className="mb-2">
            <label className="block text-sm text-gray-600">Heatmap opacity: {Math.round(opacity*100)}%</label>
            <input type="range" min="0" max="1" step="0.01" value={opacity} onChange={e=>setOpacity(Number(e.target.value))} className="w-full" />
          </div>
          <div className="flex items-center gap-3 mb-3">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showHeatmap} onChange={e=>setShowHeatmap(e.target.checked)} /> Show heatmap
            </label>
            {result.heatmap_url && (
              <button className="ml-auto bg-blue-500 text-white px-3 py-1 rounded text-sm" onClick={async ()=>{ const r=await fetch(result.heatmap_url); const b=await r.blob(); const url=URL.createObjectURL(b); const a=document.createElement('a'); a.href=url; a.download='heatmap.png'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }}>Download heatmap</button>
            )}
            {result.image_url && (
              <button className="bg-gray-700 text-white px-3 py-1 rounded text-sm" onClick={async ()=>{ const r=await fetch(result.image_url); const b=await r.blob(); const url=URL.createObjectURL(b); const a=document.createElement('a'); a.href=url; a.download='fundus.png'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }}>Download image</button>
            )}
          </div>
          <p className="text-sm text-gray-500">Timestamp: {new Date(result.timestamp).toLocaleString()}</p>
        </div>
      </div>
    </div>
  )
}

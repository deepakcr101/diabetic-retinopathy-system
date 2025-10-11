import React, { useState } from 'react'
import api from '../lib/api'

export default function Upload(){
  const [file,setFile] = useState(null)
  const [loading,setLoading] = useState(false)
  const [result,setResult] = useState(null)

  const submit = async (e)=>{
    e.preventDefault()
    if(!file) return alert('Select file')
    setLoading(true)
    const form = new FormData()
    form.append('image', file)
    try{
      const res = await api.post('/api/predict/', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(res.data)
    }catch(err){
      alert('Upload failed')
    }finally{ setLoading(false) }
  }

  return (
    <div className="max-w-2xl mx-auto bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4">Upload Fundus Image</h2>
      <form onSubmit={submit} className="flex flex-col gap-3">
        <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])} />
        <button disabled={loading} className="bg-green-600 text-white py-2 rounded">{loading? 'Analyzing...' : 'Upload & Analyze'}</button>
      </form>

      {result && (
        <div className="mt-4">
          <h3 className="font-semibold">Result</h3>
          <p>Stage: {result.predicted_stage}</p>
          <p>Confidence: {Math.round(result.confidence*100)}%</p>
          {result.heatmap_url && <img src={result.heatmap_url} alt="heatmap" className="mt-2 max-w-full" />}
        </div>
      )}
    </div>
  )
}

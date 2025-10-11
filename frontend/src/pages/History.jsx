import React, { useEffect, useState } from 'react'
import api from '../lib/api'

export default function History(){
  const [rows,setRows] = useState([])
  useEffect(()=>{
    api.get('/api/results/').then(r=> setRows(r.data.results || [])).catch(()=>{})
  },[])

  return (
    <div className="max-w-4xl mx-auto bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4">History</h2>
      <table className="w-full table-auto">
        <thead>
          <tr>
            <th>Date</th>
            <th>Stage</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r=> (
            <tr key={r.id} className="border-t">
              <td>{new Date(r.timestamp).toLocaleString()}</td>
              <td>{r.predicted_stage}</td>
              <td>{Math.round(r.confidence*100)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

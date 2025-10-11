import React, { useState } from 'react'
import api from '../lib/api'

export default function Login(){
  const [username,setUsername] = useState('')
  const [password,setPassword] = useState('')

  const submit = async (e) =>{
    e.preventDefault()
    try{
  const res = await api.post('/api/token/', { username,password })
      localStorage.setItem('access', res.data.access)
      localStorage.setItem('refresh', res.data.refresh)
      alert('Logged in (demo).')
    }catch(err){
      alert('Login failed')
    }
  }

  return (
    <div className="max-w-md mx-auto bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4">Login</h2>
      <form onSubmit={submit} className="flex flex-col gap-3">
        <input className="border p-2" placeholder="Username" value={username} onChange={e=>setUsername(e.target.value)} />
        <input type="password" className="border p-2" placeholder="Password" value={password} onChange={e=>setPassword(e.target.value)} />
        <label className="flex items-center gap-2"><input type="checkbox"/> Remember me</label>
        <button className="bg-blue-600 text-white py-2 rounded">Sign in</button>
      </form>
    </div>
  )
}

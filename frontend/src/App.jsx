import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import Login from './pages/Login'
import Upload from './pages/Upload'
import Results from './pages/Results'
import History from './pages/History'

export default function App(){
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow p-4">
        <div className="container mx-auto flex gap-4">
          <Link to="/" className="font-semibold">Home</Link>
          <Link to="/upload">Upload</Link>
          <Link to="/results">Results</Link>
          <Link to="/history">History</Link>
        </div>
      </nav>
      <main className="container mx-auto p-4">
        <Routes>
          <Route path="/" element={<Login/>} />
          <Route path="/upload" element={<Upload/>} />
          <Route path="/results" element={<Results/>} />
          <Route path="/history" element={<History/>} />
        </Routes>
      </main>
    </div>
  )
}

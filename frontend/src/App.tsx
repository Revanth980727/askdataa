import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout } from '@components/layout/Layout'
import { Dashboard } from '@pages/Dashboard'
import { Connections } from '@pages/Connections'
import { Ask } from '@pages/Ask'
import { Runs } from '@pages/Runs'
import { Models } from '@pages/Models'
import { Datasets } from '@pages/Datasets'
import { Admin } from '@pages/Admin'
import { Settings } from '@pages/Settings'
import { Help } from '@pages/Help'
import { ConnectionProvider } from '@components/providers/ConnectionProvider'
import { ThemeProvider } from '@components/providers/ThemeProvider'

function App() {
  return (
    <ThemeProvider>
      <ConnectionProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/connections" element={<Connections />} />
            <Route path="/ask" element={<Ask />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/models" element={<Models />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/help" element={<Help />} />
          </Routes>
        </Layout>
      </ConnectionProvider>
    </ThemeProvider>
  )
}

export default App

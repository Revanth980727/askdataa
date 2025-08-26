import React, { useState } from 'react'
import { TopBar } from './TopBar'
import { Sidebar } from './Sidebar'
import { HealthDrawer } from './HealthDrawer'

interface LayoutProps {
  children: React.ReactNode
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [healthDrawerOpen, setHealthDrawerOpen] = useState(false)

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Top Bar */}
      <TopBar 
        onHealthClick={() => setHealthDrawerOpen(true)}
        onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      
      <div className="flex">
        {/* Sidebar */}
        <Sidebar 
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
        
        {/* Main Content */}
        <main className={`flex-1 transition-all duration-300 ${
          sidebarCollapsed ? 'ml-16' : 'ml-64'
        }`}>
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
      
      {/* Health Drawer */}
      <HealthDrawer 
        open={healthDrawerOpen}
        onClose={() => setHealthDrawerOpen(false)}
      />
    </div>
  )
}

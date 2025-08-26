import React, { useState } from 'react'
import { 
  Menu, 
  Settings, 
  HelpCircle, 
  Sun, 
  Moon, 
  Trash2,
  ChevronDown,
  Database,
  Zap
} from 'lucide-react'
import { useConnectionStore } from '@store/connectionStore'
import { useThemeStore } from '@store/themeStore'
import { ConnectionPicker } from '@components/connection/ConnectionPicker'
import { StatusIndicator } from '@components/ui/StatusIndicator'

interface TopBarProps {
  onHealthClick: () => void
  onSidebarToggle: () => void
}

export const TopBar: React.FC<TopBarProps> = ({ onHealthClick, onSidebarToggle }) => {
  const { activeConnection, clearSessionMemory } = useConnectionStore()
  const { theme, toggleTheme } = useThemeStore()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const handleConnectionChange = (connectionId: string | null) => {
    if (activeConnection?.id !== connectionId) {
      clearSessionMemory()
    }
  }

  const getStatusColor = () => {
    // TODO: Implement actual health check logic
    return 'success' // Placeholder
  }

  return (
    <header className="bg-white border-b border-neutral-200 px-6 py-3 flex items-center justify-between">
      {/* Left side - App name and sidebar toggle */}
      <div className="flex items-center space-x-4">
        <button
          onClick={onSidebarToggle}
          className="p-2 hover:bg-neutral-100 rounded-md transition-colors"
        >
          <Menu className="h-5 w-5 text-neutral-600" />
        </button>
        
        <div className="flex items-center space-x-2">
          <Database className="h-6 w-6 text-primary-600" />
          <h1 className="text-xl font-semibold text-neutral-900">AskData</h1>
        </div>
      </div>

      {/* Center - Active connection pill */}
      <div className="flex-1 flex justify-center">
        <ConnectionPicker
          value={activeConnection?.id || null}
          onChange={handleConnectionChange}
        />
      </div>

      {/* Right side - Status and user menu */}
      <div className="flex items-center space-x-4">
        {/* Run status light */}
        <button
          onClick={onHealthClick}
          className="flex items-center space-x-2 p-2 hover:bg-neutral-100 rounded-md transition-colors"
        >
          <StatusIndicator status={getStatusColor()} />
          <span className="text-sm text-neutral-600">System Status</span>
        </button>

        {/* User menu */}
        <div className="relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="flex items-center space-x-2 p-2 hover:bg-neutral-100 rounded-md transition-colors"
          >
            <Settings className="h-5 w-5 text-neutral-600" />
            <ChevronDown className="h-4 w-4 text-neutral-600" />
          </button>

          {showUserMenu && (
            <div className="absolute right-0 top-full mt-2 w-56 bg-white border border-neutral-200 rounded-lg shadow-strong z-50">
              <div className="py-2">
                <button
                  onClick={toggleTheme}
                  className="w-full px-4 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-50 flex items-center space-x-2"
                >
                  {theme === 'dark' ? (
                    <>
                      <Sun className="h-4 w-4" />
                      <span>Light Mode</span>
                    </>
                  ) : (
                    <>
                      <Moon className="h-4 w-4" />
                      <span>Dark Mode</span>
                    </>
                  )}
                </button>
                
                <button className="w-full px-4 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-50 flex items-center space-x-2">
                  <HelpCircle className="h-4 w-4" />
                  <span>About</span>
                </button>
                
                <button className="w-full px-4 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-50 flex items-center space-x-2">
                  <Zap className="h-4 w-4" />
                  <span>Shortcuts</span>
                </button>
                
                <div className="border-t border-neutral-200 my-1" />
                
                <button className="w-full px-4 py-2 text-left text-sm text-danger-600 hover:bg-neutral-50 flex items-center space-x-2">
                  <Trash2 className="h-4 w-4" />
                  <span>Clear Local Data</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

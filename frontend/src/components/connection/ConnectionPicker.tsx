import React, { useState } from 'react'
import { ChevronDown, Database, Globe } from 'lucide-react'
import { useConnectionStore } from '@store/connectionStore'
import { cn } from '@utils/cn'

interface ConnectionPickerProps {
  value: string | null
  onChange: (connectionId: string | null) => void
}

const engineIcons = {
  postgresql: '🐘',
  mysql: '🐬',
  snowflake: '❄️',
  bigquery: '☁️',
  redshift: '🔴',
  other: '💾',
}

export const ConnectionPicker: React.FC<ConnectionPickerProps> = ({ value, onChange }) => {
  const { connections, getConnection } = useConnectionStore()
  const [isOpen, setIsOpen] = useState(false)
  
  const activeConnection = value ? getConnection(value) : null

  const handleConnectionSelect = (connectionId: string | null) => {
    onChange(connectionId)
    setIsOpen(false)
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center space-x-2 px-4 py-2 bg-white border border-neutral-300 rounded-full hover:bg-neutral-50 transition-colors",
          isOpen && "ring-2 ring-primary-500 ring-offset-2"
        )}
      >
        {activeConnection ? (
          <>
            <span className="text-lg">{engineIcons[activeConnection.engine]}</span>
            <span className="font-medium text-neutral-900">{activeConnection.alias}</span>
            <span className="text-sm text-neutral-500">({activeConnection.engine})</span>
          </>
        ) : (
          <>
            <Globe className="h-4 w-4 text-neutral-500" />
            <span className="font-medium text-neutral-700">Auto</span>
          </>
        )}
        <ChevronDown className={cn(
          "h-4 w-4 text-neutral-500 transition-transform",
          isOpen && "rotate-180"
        )} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 w-80 bg-white border border-neutral-200 rounded-lg shadow-strong z-50">
          <div className="p-3 border-b border-neutral-200">
            <h3 className="font-medium text-neutral-900">Select Connection</h3>
            <p className="text-sm text-neutral-600">Choose a database connection or use Auto routing</p>
          </div>
          
          <div className="max-h-64 overflow-y-auto">
            {/* Auto option */}
            <button
              onClick={() => handleConnectionSelect(null)}
              className={cn(
                "w-full p-3 text-left hover:bg-neutral-50 transition-colors border-b border-neutral-100",
                !value && "bg-primary-50 text-primary-700"
              )}
            >
              <div className="flex items-center space-x-3">
                <Globe className="h-5 w-5 text-neutral-500" />
                <div>
                  <div className="font-medium">Auto</div>
                  <div className="text-sm text-neutral-600">Let the system choose the best connection</div>
                </div>
              </div>
            </button>

            {/* Connection options */}
            {connections.map((connection) => (
              <button
                key={connection.id}
                onClick={() => handleConnectionSelect(connection.id)}
                className={cn(
                  "w-full p-3 text-left hover:bg-neutral-50 transition-colors border-b border-neutral-100",
                  value === connection.id && "bg-primary-50 text-primary-700"
                )}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{engineIcons[connection.engine]}</span>
                  <div className="flex-1">
                    <div className="font-medium">{connection.alias}</div>
                    <div className="text-sm text-neutral-600">
                      {connection.host}:{connection.port} • {connection.database}
                    </div>
                  </div>
                  <div className={cn(
                    "w-2 h-2 rounded-full",
                    connection.status === 'connected' && "bg-success-500",
                    connection.status === 'disconnected' && "bg-neutral-400",
                    connection.status === 'error' && "bg-danger-500"
                  )} />
                </div>
              </button>
            ))}
          </div>

          {connections.length === 0 && (
            <div className="p-4 text-center text-neutral-500">
              <Database className="h-8 w-8 mx-auto mb-2 text-neutral-400" />
              <p>No connections yet</p>
              <p className="text-sm">Add a database connection to get started</p>
            </div>
          )}
        </div>
      )}

      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  )
}

import React from 'react'
import { X, Activity, Server, Database, Brain } from 'lucide-react'
import { cn } from '@utils/cn'

interface HealthDrawerProps {
  open: boolean
  onClose: () => void
}

export const HealthDrawer: React.FC<HealthDrawerProps> = ({ open, onClose }) => {
  // Mock health data - replace with actual API calls
  const healthData = {
    orchestrator: { status: 'healthy', responseTime: '45ms', lastPing: '2s ago' },
    connectionRegistry: { status: 'healthy', responseTime: '23ms', lastPing: '1s ago' },
    introspect: { status: 'healthy', responseTime: '67ms', lastPing: '3s ago' },
    indexer: { status: 'healthy', responseTime: '89ms', lastPing: '2s ago' },
    tableRetriever: { status: 'healthy', responseTime: '34ms', lastPing: '1s ago' },
    microProfiler: { status: 'healthy', responseTime: '56ms', lastPing: '2s ago' },
    columnPruner: { status: 'healthy', responseTime: '78ms', lastPing: '1s ago' },
    joinGraph: { status: 'healthy', responseTime: '45ms', lastPing: '2s ago' },
    metricResolver: { status: 'healthy', responseTime: '67ms', lastPing: '1s ago' },
    sqlGenerator: { status: 'healthy', responseTime: '123ms', lastPing: '3s ago' },
    sqlValidator: { status: 'healthy', responseTime: '89ms', lastPing: '2s ago' },
    queryExecutor: { status: 'healthy', responseTime: '156ms', lastPing: '1s ago' },
    resultExplainer: { status: 'healthy', responseTime: '234ms', lastPing: '2s ago' },
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-success-600 bg-success-50'
      case 'warning':
        return 'text-warning-600 bg-warning-50'
      case 'error':
        return 'text-danger-600 bg-danger-50'
      default:
        return 'text-neutral-600 bg-neutral-50'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '🟢'
      case 'warning':
        return '🟡'
      case 'error':
        return '🔴'
      default:
        return '⚪'
    }
  }

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-50"
          onClick={onClose}
        />
      )}
      
      {/* Drawer */}
      <div className={cn(
        "fixed right-0 top-0 h-full w-96 bg-white shadow-strong transform transition-transform duration-300 ease-in-out z-50",
        open ? "translate-x-0" : "translate-x-full"
      )}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200">
          <div className="flex items-center space-x-3">
            <Activity className="h-6 w-6 text-primary-600" />
            <h2 className="text-lg font-semibold text-neutral-900">System Health</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 rounded-md transition-colors"
          >
            <X className="h-5 w-5 text-neutral-600" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Overall Status */}
          <div className="bg-success-50 border border-success-200 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-success-500 rounded-full"></div>
              <span className="font-medium text-success-800">All Systems Operational</span>
            </div>
            <p className="text-sm text-success-700 mt-1">
              All 13 services are responding normally
            </p>
          </div>

          {/* Service Health Table */}
          <div>
            <h3 className="font-medium text-neutral-900 mb-3">Service Status</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {Object.entries(healthData).map(([service, data]) => (
                <div key={service} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <span className="text-lg">{getStatusIcon(data.status)}</span>
                    <div>
                      <p className="text-sm font-medium text-neutral-900 capitalize">
                        {service.replace(/([A-Z])/g, ' $1').trim()}
                      </p>
                      <p className="text-xs text-neutral-600">
                        Last ping: {data.lastPing}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={cn(
                      "inline-block px-2 py-1 rounded-full text-xs font-medium",
                      getStatusColor(data.status)
                    )}>
                      {data.status}
                    </span>
                    <p className="text-xs text-neutral-600 mt-1">
                      {data.responseTime}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* System Metrics */}
          <div>
            <h3 className="font-medium text-neutral-900 mb-3">System Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-neutral-50 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Database className="h-4 w-4 text-neutral-600" />
                  <span className="text-sm text-neutral-600">Active Connections</span>
                </div>
                <p className="text-lg font-semibold text-neutral-900">3</p>
              </div>
              
              <div className="bg-neutral-50 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Brain className="h-4 w-4 text-neutral-600" />
                  <span className="text-sm text-neutral-600">Queue Length</span>
                </div>
                <p className="text-lg font-semibold text-neutral-900">0</p>
              </div>
            </div>
          </div>

          {/* Last Errors */}
          <div>
            <h3 className="font-medium text-neutral-900 mb-3">Recent Errors</h3>
            <div className="text-center py-4 text-neutral-500">
              <p className="text-sm">No errors in the last 24 hours</p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

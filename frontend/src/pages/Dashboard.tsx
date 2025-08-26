import React from 'react'
import { Link } from 'react-router-dom'
import { 
  Plus, 
  MessageSquare, 
  Play, 
  Database, 
  Brain, 
  BarChart3,
  AlertTriangle,
  Clock,
  CheckCircle,
  XCircle
} from 'lucide-react'
import { useConnectionStore } from '@store/connectionStore'
import { PageHeader } from '@components/ui/PageHeader'
import { Card } from '@components/ui/Card'
import { Button } from '@components/ui/Button'
import { StatusIndicator } from '@components/ui/StatusIndicator'

export const Dashboard: React.FC = () => {
  const { connections, activeConnection, getActiveConnection } = useConnectionStore()
  const activeConnectionData = getActiveConnection()

  // Mock data - replace with actual API calls
  const recentRuns = [
    { id: 'run_001', question: 'Show me total sales by month', status: 'completed', time: '2.3s', rows: 12 },
    { id: 'run_002', question: 'Which customers have the highest order values?', status: 'running', time: '1.8s', rows: 0 },
    { id: 'run_003', question: 'Revenue trends over the last quarter', status: 'failed', time: '0.5s', rows: 0 },
  ]

  const alerts = [
    { type: 'warning', message: 'Schema drift detected in orders table', link: '/connections' },
    { type: 'info', message: 'New embeddings available for customer data', link: '/admin' },
  ]

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-success-500" />
      case 'running':
        return <Clock className="h-4 w-4 text-warning-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-danger-500" />
      default:
        return <Clock className="h-4 w-4 text-neutral-500" />
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Quick overview of your AskData system and recent activity"
      />

      {/* Quick Start Section */}
      <Card>
        <div className="card-header">
          <h2 className="text-lg font-semibold text-neutral-900">Quick Start</h2>
          <p className="text-sm text-neutral-600">Get started with AskData in a few clicks</p>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button
              variant="primary"
              size="lg"
              className="h-24 flex-col space-y-2"
              asChild
            >
              <Link to="/connections">
                <Plus className="h-6 w-6" />
                <span>Connect a Database</span>
              </Link>
            </Button>
            
            <Button
              variant="outline"
              size="lg"
              className="h-24 flex-col space-y-2"
              asChild
            >
              <Link to="/ask">
                <MessageSquare className="h-6 w-6" />
                <span>Ask a Question</span>
              </Link>
            </Button>
            
            <Button
              variant="outline"
              size="lg"
              className="h-24 flex-col space-y-2"
              asChild
            >
              <Link to="/runs">
                <Play className="h-6 w-6" />
                <span>View Runs</span>
              </Link>
            </Button>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Connection */}
        <Card>
          <div className="card-header">
            <h2 className="text-lg font-semibold text-neutral-900">Active Connection</h2>
          </div>
          <div className="card-content">
            {activeConnectionData ? (
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <Database className="h-8 w-8 text-primary-600" />
                  <div>
                    <h3 className="font-medium text-neutral-900">{activeConnectionData.alias}</h3>
                    <p className="text-sm text-neutral-600">{activeConnectionData.engine}</p>
                  </div>
                  <StatusIndicator 
                    status={activeConnectionData.status === 'connected' ? 'success' : 'danger'} 
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-neutral-600">Schemas:</span>
                    <span className="ml-2 font-medium">{activeConnectionData.schemas.length}</span>
                  </div>
                  <div>
                    <span className="text-neutral-600">Last Ingest:</span>
                    <span className="ml-2 font-medium">
                      {activeConnectionData.lastIngest ? 
                        new Date(activeConnectionData.lastIngest).toLocaleDateString() : 
                        'Never'
                      }
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral-600">Embeddings:</span>
                    <span className="ml-2 font-medium">
                      {activeConnectionData.embVersion || 'None'}
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral-600">Schema Rev:</span>
                    <span className="ml-2 font-medium">
                      {activeConnectionData.schemaRev || 'None'}
                    </span>
                  </div>
                </div>
                
                <Button variant="outline" size="sm" asChild>
                  <Link to="/connections">Manage Connection</Link>
                </Button>
              </div>
            ) : (
              <div className="text-center py-8">
                <Database className="h-12 w-12 mx-auto text-neutral-400 mb-4" />
                <p className="text-neutral-600 mb-4">No active connection</p>
                <Button variant="primary" asChild>
                  <Link to="/connections">Add Connection</Link>
                </Button>
              </div>
            )}
          </div>
        </Card>

        {/* Recent Runs */}
        <Card>
          <div className="card-header">
            <h2 className="text-lg font-semibold text-neutral-900">Recent Runs</h2>
            <Link to="/runs" className="text-sm text-primary-600 hover:text-primary-700">
              View all runs
            </Link>
          </div>
          <div className="card-content">
            <div className="space-y-3">
              {recentRuns.map((run) => (
                <div key={run.id} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-neutral-900 truncate">
                      {run.question}
                    </p>
                    <div className="flex items-center space-x-4 text-xs text-neutral-600 mt-1">
                      <span>{run.time}</span>
                      <span>{run.rows} rows</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(run.status)}
                    <span className="text-xs text-neutral-600 capitalize">{run.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training and Adapters */}
        <Card>
          <div className="card-header">
            <h2 className="text-lg font-semibold text-neutral-900">Training & Adapters</h2>
          </div>
          <div className="card-content">
            {activeConnectionData?.adapterStatus === 'none' ? (
              <div className="text-center py-6">
                <Brain className="h-12 w-12 mx-auto text-neutral-400 mb-4" />
                <p className="text-neutral-600 mb-4">No adapter trained for this connection</p>
                <div className="space-x-2">
                  <Button variant="outline" size="sm" asChild>
                    <Link to="/datasets">Build Dataset</Link>
                  </Button>
                  <Button variant="outline" size="sm" asChild>
                    <Link to="/models">Train Adapter</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600">Adapter Status:</span>
                  <span className="text-sm font-medium capitalize">{activeConnectionData.adapterStatus}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600">Version:</span>
                  <span className="text-sm font-medium">v1.2.0</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600">A/B Status:</span>
                  <span className="text-sm font-medium">Active (30%)</span>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Index and Profiler Status */}
        <Card>
          <div className="card-header">
            <h2 className="text-lg font-semibold text-neutral-900">Index & Profiler Status</h2>
          </div>
          <div className="card-content">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-neutral-600">Tables Indexed:</span>
                  <span className="ml-2 font-medium">24</span>
                </div>
                <div>
                  <span className="text-neutral-600">Columns Indexed:</span>
                  <span className="ml-2 font-medium">156</span>
                </div>
                <div>
                  <span className="text-neutral-600">Profiler Cache:</span>
                  <span className="ml-2 font-medium">18 tables</span>
                </div>
                <div>
                  <span className="text-neutral-600">Cache TTL:</span>
                  <span className="ml-2 font-medium">24h</span>
                </div>
              </div>
              
              <div className="flex space-x-2">
                <Button variant="outline" size="sm">Re-introspect</Button>
                <Button variant="outline" size="sm">Re-embed</Button>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Card>
          <div className="card-header">
            <h2 className="text-lg font-semibold text-neutral-900">Alerts</h2>
          </div>
          <div className="card-content">
            <div className="space-y-3">
              {alerts.map((alert, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-warning-50 border border-warning-200 rounded-lg">
                  <AlertTriangle className="h-5 w-5 text-warning-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-sm text-warning-800">{alert.message}</p>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link to={alert.link}>View</Link>
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

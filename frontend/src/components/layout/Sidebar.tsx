import React from 'react'
import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard,
  Database,
  MessageSquare,
  Play,
  Brain,
  BarChart3,
  Settings,
  HelpCircle,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { cn } from '@utils/cn'

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

const navigationItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard', description: 'Overview and quick actions' },
  { path: '/connections', icon: Database, label: 'Connections', description: 'Manage database connections' },
  { path: '/ask', icon: MessageSquare, label: 'Ask', description: 'Ask questions in natural language' },
  { path: '/runs', icon: Play, label: 'Runs', description: 'View and manage query runs' },
  { path: '/models', icon: Brain, label: 'Models', description: 'Manage AI models and adapters' },
  { path: '/datasets', icon: BarChart3, label: 'Datasets', description: 'Build and manage training datasets' },
  { path: '/admin', icon: Settings, label: 'Admin', description: 'System administration and maintenance' },
  { path: '/settings', icon: Settings, label: 'Settings', description: 'Application configuration' },
  { path: '/help', icon: HelpCircle, label: 'Help', description: 'Documentation and support' },
]

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle }) => {
  return (
    <aside className={cn(
      "fixed left-0 top-16 h-full bg-white border-r border-neutral-200 transition-all duration-300 z-40",
      collapsed ? "w-16" : "w-64"
    )}>
      {/* Toggle button */}
      <div className="flex justify-end p-2 border-b border-neutral-200">
        <button
          onClick={onToggle}
          className="p-1 hover:bg-neutral-100 rounded transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4 text-neutral-600" />
          ) : (
            <ChevronLeft className="h-4 w-4 text-neutral-600" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="p-2">
        <ul className="space-y-1">
          {navigationItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) => cn(
                  "flex items-center space-x-3 px-3 py-2 rounded-md transition-colors group relative",
                  isActive 
                    ? "bg-primary-50 text-primary-700 border-r-2 border-primary-600" 
                    : "text-neutral-700 hover:bg-neutral-50 hover:text-neutral-900"
                )}
                title={collapsed ? item.description : undefined}
              >
                <item.icon className={cn(
                  "h-5 w-5 flex-shrink-0",
                  collapsed ? "mx-auto" : ""
                )} />
                
                {!collapsed && (
                  <>
                    <span className="font-medium">{item.label}</span>
                    
                    {/* Tooltip for collapsed state */}
                    {collapsed && (
                      <div className="absolute left-full ml-2 px-2 py-1 bg-neutral-900 text-white text-sm rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                        {item.label}
                        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 w-0 h-0 border-l-4 border-l-neutral-900 border-t-2 border-t-transparent border-b-2 border-b-transparent"></div>
                      </div>
                    )}
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Connection status indicator */}
      {!collapsed && (
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
            <div className="flex items-center space-x-2 text-sm text-neutral-600">
              <div className="w-2 h-2 bg-success-500 rounded-full"></div>
              <span>All systems healthy</span>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}

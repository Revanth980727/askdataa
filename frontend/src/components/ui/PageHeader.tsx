import React from 'react'

interface PageHeaderProps {
  title: string
  description?: string
  children?: React.ReactNode
}

export const PageHeader: React.FC<PageHeaderProps> = ({ title, description, children }) => {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900">{title}</h1>
        {description && (
          <p className="mt-1 text-neutral-600">{description}</p>
        )}
      </div>
      {children && (
        <div className="flex items-center space-x-2">
          {children}
        </div>
      )}
    </div>
  )
}

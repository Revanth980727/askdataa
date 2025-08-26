import React from 'react'
import { cn } from '@utils/cn'

type StatusType = 'success' | 'warning' | 'danger' | 'neutral'

interface StatusIndicatorProps {
  status: StatusType
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ 
  status, 
  size = 'md',
  animated = false 
}) => {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  }

  const statusClasses = {
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    neutral: 'bg-neutral-400'
  }

  return (
    <div
      className={cn(
        'rounded-full',
        sizeClasses[size],
        statusClasses[status],
        animated && status === 'warning' && 'animate-pulse',
        animated && status === 'danger' && 'animate-pulse-slow'
      )}
    />
  )
}

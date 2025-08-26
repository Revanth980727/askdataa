import React from 'react'
import { cn } from '@utils/cn'

interface CardProps {
  children: React.ReactNode
  className?: string
}

export const Card: React.FC<CardProps> = ({ children, className }) => {
  return (
    <div className={cn('card', className)}>
      {children}
    </div>
  )
}

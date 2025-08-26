import React from 'react'
import { cn } from '@utils/cn'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  asChild?: boolean
}

export const Button: React.FC<ButtonProps> = ({ 
  className, 
  variant = 'primary', 
  size = 'md',
  asChild = false,
  children,
  ...props 
}) => {
  const baseClasses = 'btn'
  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    outline: 'btn-outline',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
  }
  const sizeClasses = {
    sm: 'btn-sm',
    md: 'btn-md',
    lg: 'btn-lg',
  }

  const classes = cn(
    baseClasses,
    variantClasses[variant],
    sizeClasses[size],
    className
  )

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      className: cn(classes, children.props.className),
      ...props,
    })
  }

  return (
    <button className={classes} {...props}>
      {children}
    </button>
  )
}

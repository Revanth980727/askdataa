import React, { createContext, useContext, useEffect } from 'react'
import { useThemeStore } from '@store/themeStore'

interface ThemeContextType {
  // Add any theme-specific context here if needed
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useThemeContext = () => {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useThemeContext must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: React.ReactNode
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const { theme, setTheme } = useThemeStore()

  // Apply theme to document on mount and theme change
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  return (
    <ThemeContext.Provider value={{}}>
      {children}
    </ThemeContext.Provider>
  )
}

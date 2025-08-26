import React, { createContext, useContext, useEffect } from 'react'
import { useConnectionStore } from '@store/connectionStore'

interface ConnectionContextType {
  // Add any connection-specific context here if needed
}

const ConnectionContext = createContext<ConnectionContextType | undefined>(undefined)

export const useConnectionContext = () => {
  const context = useContext(ConnectionContext)
  if (context === undefined) {
    throw new Error('useConnectionContext must be used within a ConnectionProvider')
  }
  return context
}

interface ConnectionProviderProps {
  children: React.ReactNode
}

export const ConnectionProvider: React.FC<ConnectionProviderProps> = ({ children }) => {
  const { connections, activeConnectionId } = useConnectionStore()

  // Initialize connection store if needed
  useEffect(() => {
    // Any connection initialization logic can go here
  }, [])

  return (
    <ConnectionContext.Provider value={{}}>
      {children}
    </ConnectionContext.Provider>
  )
}

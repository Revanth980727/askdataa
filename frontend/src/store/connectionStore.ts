import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface DatabaseConnection {
  id: string
  alias: string
  engine: 'postgresql' | 'mysql' | 'snowflake' | 'bigquery' | 'redshift' | 'other'
  host: string
  port: number
  database: string
  username: string
  schemas: string[]
  status: 'connected' | 'disconnected' | 'error'
  lastIngest: string | null
  embVersion: string | null
  schemaRev: string | null
  adapterStatus: 'none' | 'live' | 'staged'
  createdAt: string
  updatedAt: string
}

export interface ConnectionState {
  connections: DatabaseConnection[]
  activeConnectionId: string | null
  sessionId: string | null
  sessionMemory: Record<string, any>
  
  // Actions
  addConnection: (connection: Omit<DatabaseConnection, 'id' | 'createdAt' | 'updatedAt'>) => void
  updateConnection: (id: string, updates: Partial<DatabaseConnection>) => void
  deleteConnection: (id: string) => void
  setActiveConnection: (id: string | null) => void
  clearSessionMemory: () => void
  updateSessionMemory: (key: string, value: any) => void
  getConnection: (id: string) => DatabaseConnection | undefined
  getActiveConnection: () => DatabaseConnection | undefined
}

export const useConnectionStore = create<ConnectionState>()(
  persist(
    (set, get) => ({
      connections: [],
      activeConnectionId: null,
      sessionId: null,
      sessionMemory: {},

      addConnection: (connection) => {
        const newConnection: DatabaseConnection = {
          ...connection,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }
        
        set((state) => ({
          connections: [...state.connections, newConnection],
        }))
      },

      updateConnection: (id, updates) => {
        set((state) => ({
          connections: state.connections.map((conn) =>
            conn.id === id
              ? { ...conn, ...updates, updatedAt: new Date().toISOString() }
              : conn
          ),
        }))
      },

      deleteConnection: (id) => {
        set((state) => ({
          connections: state.connections.filter((conn) => conn.id !== id),
          activeConnectionId: state.activeConnectionId === id ? null : state.activeConnectionId,
        }))
      },

      setActiveConnection: (id) => {
        set({ activeConnectionId: id })
      },

      clearSessionMemory: () => {
        set({ sessionMemory: {} })
      },

      updateSessionMemory: (key, value) => {
        set((state) => ({
          sessionMemory: { ...state.sessionMemory, [key]: value },
        }))
      },

      getConnection: (id) => {
        return get().connections.find((conn) => conn.id === id)
      },

      getActiveConnection: () => {
        const { activeConnectionId, connections } = get()
        return activeConnectionId ? connections.find((conn) => conn.id === activeConnectionId) : undefined
      },
    }),
    {
      name: 'askdata-connections',
      partialize: (state) => ({
        connections: state.connections,
        activeConnectionId: state.activeConnectionId,
      }),
    }
  )
)

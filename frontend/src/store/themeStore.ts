import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'light' | 'dark'

export interface ThemeState {
  theme: Theme
  
  // Actions
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'light',

      setTheme: (theme) => {
        set({ theme })
        // Apply theme to document
        document.documentElement.classList.toggle('dark', theme === 'dark')
      },

      toggleTheme: () => {
        const newTheme = get().theme === 'light' ? 'dark' : 'light'
        get().setTheme(newTheme)
      },
    }),
    {
      name: 'askdata-theme',
    }
  )
)

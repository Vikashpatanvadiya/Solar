import { defineConfig } from 'vite'

export default defineConfig({
  root: 'site',
  server: {
    open: true,
    // Forward the AI forecast API to the FastAPI backend (see backend/app/main.py)
    proxy: {
      '/api': 'http://127.0.0.1:8000'
    }
  }
})

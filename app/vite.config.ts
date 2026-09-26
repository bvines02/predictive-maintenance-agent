/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// /api is proxied to the FastAPI server (uvicorn src.api:app, port 8000) so
// the LLM step works in dev without CORS. Everything else is static.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  build: { chunkSizeWarningLimit: 700 }, // recharts; the fixture JSON is fetched, not bundled
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
  test: { environment: 'node' },
})

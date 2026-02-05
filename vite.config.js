import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Allows access from outside localhost (required for Codespaces)
    port: 5173,      // Your existing port configuration
    hmr: {
      clientPort: 5173 // Ensures hot reload works in Codespaces
    }
  }
})

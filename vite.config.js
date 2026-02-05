import { randomFillSync } from 'node:crypto';
// This shims the missing crypto method before Vite starts
if (typeof globalThis.crypto === 'undefined') {
  globalThis.crypto = {
    getRandomValues: (buffer) => randomFillSync(buffer)
  };
}

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true }
})

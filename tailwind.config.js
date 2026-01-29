/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        'plexie': {
          'dark': '#0a0f1a',
          'card': '#111827',
          'border': '#1f2937',
          'green': '#10b981',
        }
      }
    },
  },
  plugins: [],
}

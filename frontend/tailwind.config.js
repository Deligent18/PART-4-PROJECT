/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      colors: {
        'risk-high': '#ef4444',
        'risk-low': '#22c55e',
        'risk-medium': '#f59e0b',
      }
    },
  },
  plugins: [],
}

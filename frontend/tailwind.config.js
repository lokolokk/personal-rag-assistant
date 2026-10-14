/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#09090b',
        surface: {
          DEFAULT: '#141417',
          elevated: '#18181c',
          hover: '#1e1e24',
          subtle: '#121215',
        },
        border: {
          DEFAULT: '#27272a',
          subtle: '#1f1f23',
          focus: '#6366f1',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#4f46e5',
          subtle: 'rgba(99, 102, 241, 0.12)',
          text: '#818cf8',
        },
        brand: {
          50: '#eef2ff',
          100: '#e0e7ff',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};

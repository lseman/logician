/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Space Grotesk", "IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.05), 0 20px 50px rgba(2,8,14,0.3)",
      },
      colors: {
        ink: "#0d131a",
        sand: "#ece7db",
        tide: "#67b7c8",
        ember: "#d2a35d",
        coral: "#cf7668",
      },
    },
  },
  plugins: [],
};

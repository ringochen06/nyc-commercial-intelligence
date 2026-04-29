import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        muted: "#64748b",
        accent: {
          DEFAULT: "#3b82f6",
          soft: "#dbeafe",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "SF Pro Display",
          "Inter",
          "Segoe UI",
          "Roboto",
          "system-ui",
          "sans-serif",
        ],
      },
      letterSpacing: {
        tightish: "-0.018em",
      },
      borderRadius: {
        glass: "1.5rem",
      },
      boxShadow: {
        glass:
          "inset 0 1px 0 rgba(255,255,255,0.85), inset 0 0 0 1px rgba(255,255,255,0.3), 0 1px 1px rgba(15,23,42,0.04), 0 12px 36px -12px rgba(15,23,42,0.12)",
      },
      backdropBlur: {
        xxl: "32px",
      },
    },
  },
  plugins: [],
};

export default config;

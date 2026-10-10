import { ImageResponse } from "next/og";

export const alt = "face-cloak: invisible armour against face-swap deepfakes";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          padding: 80,
          background: "radial-gradient(circle at 20% 10%, #2a1d6e 0%, #05060a 55%)",
          color: "#e8ecf5",
          fontFamily: "sans-serif",
        }}
      >
        <svg width="220" height="220" viewBox="0 0 64 64" style={{ marginRight: 56 }}>
          <path d="M32 7 12 14v15c0 13 8.500 23 20 28 11.500-5 20-15 20-28V14z" fill="none" stroke="#7c5cff" strokeWidth="3.500" strokeLinejoin="round" />
          <circle cx="32" cy="27" r="7" fill="#22d3ee" />
          <path d="M19.500 46c2.500-7 7-10 12.500-10s10 3 12.500 10" fill="none" stroke="#22d3ee" strokeWidth="3.500" strokeLinecap="round" />
        </svg>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 28, color: "#22d3ee", letterSpacing: 4, textTransform: "uppercase" }}>Multimedia Forensics</div>
          <div style={{ fontSize: 88, fontWeight: 800, lineHeight: 1.05, marginTop: 12 }}>face-cloak</div>
          <div style={{ fontSize: 36, color: "#98a2b8", marginTop: 20, maxWidth: 700 }}>
            Adversarial image cloaking against face-swap attacks
          </div>
        </div>
      </div>
    ),
    size,
  );
}

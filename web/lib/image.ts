import { MAX_INPUT_BYTES, MAX_UPLOAD_SIDE, UPLOAD_JPEG_QUALITY } from "./config";

export interface PreparedImage {
  blob: Blob;
  previewUrl: string;
  width: number;
  height: number;
  name: string;
}

export class ImageError extends Error {}

const MAX_SOURCE_PIXELS = 150_000_000;

interface Sniffed {
  mime: "image/jpeg" | "image/png" | "image/webp" | "image/heic" | null;
  /** raw (pre-EXIF-orientation) dimensions, if the header could be parsed */
  w: number | null;
  h: number | null;
}

/** Identify the format from magic bytes and, where cheap, read the pixel size from the header. */
function sniff(b: Uint8Array): Sniffed {
  const u16 = (i: number) => ((b[i] ?? 0) << 8) | (b[i + 1] ?? 0);
  const u32 = (i: number) => (u16(i) * 65536 + u16(i + 2)) >>> 0;
  const ascii = (i: number, n: number) => String.fromCharCode(...b.subarray(i, i + n));
  if (b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) {
        i++;
        continue;
      }
      const m = b[i + 1] ?? 0;
      if (m === 0xff) {
        i++;
        continue;
      }
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
        i += 2;
        continue;
      }
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { mime: "image/jpeg", h: u16(i + 5), w: u16(i + 7) };
      i += 2 + u16(i + 2);
    }
    return { mime: "image/jpeg", w: null, h: null };
  }
  if (b.length >= 24 && b[0] === 0x89 && ascii(1, 3) === "PNG") return { mime: "image/png", w: u32(16), h: u32(20) };
  if (b.length >= 30 && ascii(0, 4) === "RIFF" && ascii(8, 4) === "WEBP") {
    const t = ascii(12, 4);
    if (t === "VP8X") return { mime: "image/webp", w: 1 + ((b[24] ?? 0) | ((b[25] ?? 0) << 8) | ((b[26] ?? 0) << 16)), h: 1 + ((b[27] ?? 0) | ((b[28] ?? 0) << 8) | ((b[29] ?? 0) << 16)) };
    if (t === "VP8 ") return { mime: "image/webp", w: ((b[26] ?? 0) | ((b[27] ?? 0) << 8)) & 0x3fff, h: ((b[28] ?? 0) | ((b[29] ?? 0) << 8)) & 0x3fff };
    if (t === "VP8L") {
      const v = ((b[21] ?? 0) | ((b[22] ?? 0) << 8) | ((b[23] ?? 0) << 16) | ((b[24] ?? 0) << 24)) >>> 0;
      return { mime: "image/webp", w: (v & 0x3fff) + 1, h: ((v >>> 14) & 0x3fff) + 1 };
    }
    return { mime: "image/webp", w: null, h: null };
  }
  if (b.length >= 12 && ascii(4, 4) === "ftyp" && /^(heic|heix|hevc|mif1|msf1|heim|heis)/.test(ascii(8, 4))) {
    return { mime: "image/heic", w: null, h: null };
  }
  return { mime: null, w: null, h: null };
}

const HEIC_MSG = "HEIC photos are not supported by browsers. Export the photo as JPEG first.";
const TYPE_MSG = "Please choose a JPEG, PNG or WebP photo.";

/**
 * Decode, downscale to <= MAX_UPLOAD_SIDE on the long side and re-encode as JPEG.
 * Drawing to a canvas drops all metadata (EXIF, GPS) and applies the EXIF orientation.
 */
export async function prepareImage(file: File): Promise<PreparedImage> {
  if (file.size === 0) throw new ImageError("That file is empty.");
  if (file.size > MAX_INPUT_BYTES) throw new ImageError("That file is larger than 25 MB. Choose a smaller photo.");

  // Trust the bytes, not the (possibly empty or wrong) file.type.
  const head = new Uint8Array(await file.slice(0, 512 * 1024).arrayBuffer());
  const sn = sniff(head);
  if (sn.mime === "image/heic" || /heic|heif/i.test(file.type)) throw new ImageError(HEIC_MSG);
  if (!sn.mime) throw new ImageError(TYPE_MSG);
  if (sn.w && sn.h && sn.w * sn.h > MAX_SOURCE_PIXELS) throw new ImageError("That image has too many pixels. Choose a smaller photo.");

  // Ask the browser to decode at reduced size so a 100 MP photo is never held in memory at full resolution.
  // Only one dimension is given, so the aspect ratio is kept whatever the EXIF orientation is.
  const opts: ImageBitmapOptions = { imageOrientation: "from-image" };
  if (sn.w && sn.h && Math.max(sn.w, sn.h) > MAX_UPLOAD_SIDE) {
    const scale = MAX_UPLOAD_SIDE / Math.max(sn.w, sn.h);
    if (sn.w >= sn.h) opts.resizeWidth = Math.max(1, Math.round(sn.w * scale));
    else opts.resizeHeight = Math.max(1, Math.round(sn.h * scale));
    opts.resizeQuality = "high";
  }

  let source: ImageBitmap | HTMLImageElement;
  try {
    source = await createImageBitmap(file, opts);
  } catch {
    source = await new Promise<HTMLImageElement>((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new ImageError("The browser could not read that image. Try another file."));
      };
      img.src = url;
    });
  }
  try {
    const sw = "naturalWidth" in source ? source.naturalWidth : source.width;
    const sh = "naturalHeight" in source ? source.naturalHeight : source.height;
    if (!sw || !sh) throw new ImageError("The image has no pixels.");
    // The minimum-size check uses the real size when known (the decoded bitmap may already be downscaled).
    const realMin = sn.w && sn.h ? Math.min(sn.w, sn.h) : Math.min(sw, sh);
    if (realMin < 64) throw new ImageError("The image is too small (minimum 64 px).");
    const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(sw, sh));
    const w = Math.max(1, Math.round(sw * scale));
    const h = Math.max(1, Math.round(sh * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new ImageError("Canvas is not available in this browser.");
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#fff"; // flatten transparency
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(source, 0, 0, w, h);
    const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", UPLOAD_JPEG_QUALITY));
    if (!blob) throw new ImageError("Could not re-encode the image.");
    return { blob, previewUrl: URL.createObjectURL(blob), width: w, height: h, name: file.name };
  } finally {
    if ("close" in source) source.close();
  }
}

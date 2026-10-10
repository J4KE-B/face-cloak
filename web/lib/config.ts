export const REPO_URL = "https://github.com/J4KE-B/face-cloak";
export const REPORT_URL = `${REPO_URL}/tree/main/report`;
export const SITE_URL = "https://face-cloak.vercel.app";

export const KAGGLE_NOTEBOOK_GITHUB_URL = `${REPO_URL}/blob/main/notebooks/kaggle_demo_server.ipynb`;

/**
 * Kaggle's "Open in Kaggle" importer: copies the notebook from GitHub into the signed-in user's account as a new
 * PRIVATE notebook. Keep it private - its outputs contain the live tunnel URL.
 */
export const KAGGLE_DEMO_NOTEBOOK_URL = `https://kaggle.com/kernels/welcome?src=${KAGGLE_NOTEBOOK_GITHUB_URL}`;

export const KAGGLE_NOTEBOOK_RAW_URL =
  "https://raw.githubusercontent.com/J4KE-B/face-cloak/main/notebooks/kaggle_demo_server.ipynb";
export const LOCAL_COMMAND = "./server/run_local.sh";

/** Client-side downscale before upload. */
export const MAX_UPLOAD_SIDE = 1024;
export const UPLOAD_JPEG_QUALITY = 0.92;
/** Reject absurd inputs before even decoding them. */
export const MAX_INPUT_BYTES = 25 * 1024 * 1024;

export const POLL_INTERVAL_MS = 1500;
export const HEALTH_INTERVAL_MS = 10_000;

export const STORAGE_KEY = "facecloak.backend.v1";

export const TEAM = [
  { name: "Jacob Antony Jeejo", role: "Method, backend, demo" },
  { name: "Manasa Manoj", role: "Team member" },
  { name: "Praneetha Reddy", role: "Team member" },
];

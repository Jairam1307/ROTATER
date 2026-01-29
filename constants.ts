export const APP_NAME = "ROTATER";

/**
 * NASA API Key
 * Dynamically resolved from the environment.
 * Fallback value provided for immediate deployment.
 */
export const NASA_API_KEY = (typeof process !== 'undefined' && process.env?.NASA_API_KEY) || "f1FdQ25eAZKWE8bdqhxgLnFyFNVhha77KHNAdxKx";

export const COLORS = {
  primary: '#00f0ff', // Neon Cyan
  secondary: '#0066ff', // Deep Blue
  alert: '#ff3366', // Red/Pink
  warning: '#ff9900', // Orange
  success: '#00ff99', // Green
  darkBg: '#050a14',
  panel: 'rgba(5, 15, 30, 0.85)'
};

export const MOCK_LOCATIONS = [
  { name: "New York", lat: 40.7128, lon: -74.0060 },
  { name: "London", lat: 51.5074, lon: -0.1278 },
  { name: "Tokyo", lat: 35.6762, lon: 139.6503 },
  { name: "Mumbai", lat: 19.0760, lon: 72.8777 },
  { name: "Sydney", lat: -33.8688, lon: 151.2093 }
];

import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from '@playwright/test';

const fakeVideoPath =
  process.env.PLAYWRIGHT_FAKE_VIDEO ||
  path.resolve(process.cwd(), 'test-results', 'fake_site_cam.y4m');

const chromeArgs = [
  '--use-fake-ui-for-media-stream',
  '--use-fake-device-for-media-stream',
];
if (fs.existsSync(fakeVideoPath)) {
  chromeArgs.push(`--use-file-for-fake-video-capture=${fakeVideoPath}`);
}

export default defineConfig({
  testDir: './tests',
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:5173',
    headless: true,
    viewport: { width: 1440, height: 900 },
    launchOptions: {
      executablePath:
        process.env.PLAYWRIGHT_CHROME_PATH ||
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      args: chromeArgs,
    },
  },
});

import path from 'node:path';
import { expect, test } from '@playwright/test';

const SCREENSHOT_DIR =
  process.env.PLAYWRIGHT_SCREENSHOT_DIR || 'test-results/screenshots';

test.describe('SafeGuard E2E Verification', () => {
  test('exercises overview, image analysis, gallery, live webcam, analytics, monitoring, and settings against live ONNX backend', async ({
    page,
  }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.message);
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // 1. Overview Page
    await expect(page.getByRole('heading', { name: 'Operational Overview' })).toBeVisible();
    await expect(page.getByText('ONNX Model Ready')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('Version: EXP-E-yolo11s-onnx')).toBeVisible();
    await expect(page.getByText('Provider: CPUExecutionProvider')).toBeVisible();
    await expect(page.getByText('opset 17 / 640px')).toBeVisible();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '01_overview.png'),
      fullPage: true,
    });

    // 2. Image Analysis Page
    await page.getByRole('button', { name: 'Image Analysis' }).click();
    await expect(page.getByRole('heading', { name: 'Image Analysis' })).toBeVisible();

    const samplePath = path.resolve(
      process.cwd(),
      'public/examples/ul_ultralytics_ppe__val__0c28da30faaa.jpg',
    );
    await page.getByTestId('image-upload-input').setInputFiles(samplePath);
    await page.getByRole('button', { name: 'Run Inspection' }).click();

    await expect(page.getByText('Detections Returned')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText('Server Inference / Total')).toBeVisible();
    await expect(page.getByLabel('Bounding box detection overlay')).toBeVisible();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '02_image_analysis_canvas.png'),
      fullPage: true,
    });

    // Switch to OpenCV server-rendered output
    await page.getByRole('button', { name: 'Server Annotated' }).click();
    await expect(
      page.getByAltText('Server-rendered OpenCV annotation output'),
    ).toBeVisible();

    // 3. Example Gallery Page
    await page.getByRole('button', { name: 'Example Gallery' }).click();
    await expect(page.getByRole('heading', { name: 'Example Gallery' })).toBeVisible();
    await page.getByRole('button', { name: 'Run Inference' }).nth(1).click();
    await expect(page.getByText(/Detected \d+ objects in/)).toBeVisible({
      timeout: 20_000,
    });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '03_example_gallery.png'),
      fullPage: true,
    });

    // 4. Live Monitoring Page (fake Chrome webcam stream -> real /predict/compliance)
    await page.getByRole('button', { name: 'Live Monitoring' }).click();
    await expect(page.getByRole('heading', { name: 'Live Monitoring' })).toBeVisible();
    await page.getByRole('button', { name: 'Start Camera' }).click();
    await expect(page.getByRole('button', { name: 'Stop Camera' })).toBeVisible({
      timeout: 10_000,
    });
    // Wait until at least 2 webcam frames complete against real FastAPI backend.
    await expect
      .poll(
        async () => {
          const txt = await page.locator('text=Processed Frames').locator('..').innerText();
          const num = Number(txt.replace(/\D+/g, ''));
          return num;
        },
        { timeout: 15_000 },
      )
      .toBeGreaterThanOrEqual(2);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '04_live_monitoring.png'),
      fullPage: true,
    });
    await page.getByRole('button', { name: 'Stop Camera' }).click();

    // 5. Analytics Page (Recharts populated with real session detections/latencies)
    await page.getByRole('button', { name: 'Analytics' }).click();
    await expect(page.getByRole('heading', { name: 'Session Analytics' })).toBeVisible();
    await expect(page.locator('.recharts-responsive-container').first()).toBeVisible({
      timeout: 10_000,
    });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '05_analytics.png'),
      fullPage: true,
    });

    // 6. Alert History Page
    await page.getByRole('button', { name: 'Alert History' }).click();
    await expect(
      page.getByRole('heading', { name: 'Alert History & Human Review Queue' }),
    ).toBeVisible();

    // 7. Model Monitoring Page
    await page.getByRole('button', { name: 'Model Monitoring' }).click();
    await expect(
      page.getByRole('heading', { name: 'Model & Operational Monitoring' }),
    ).toBeVisible();
    await expect(page.getByText('Out of')).toBeVisible();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '06_model_monitoring.png'),
      fullPage: true,
    });

    // 8. Settings & Model Information Page
    await page.getByRole('button', { name: 'Settings & Model' }).click();
    await expect(
      page.getByRole('heading', { name: 'Settings & Model Information' }),
    ).toBeVisible();
    await expect(page.getByText('safeguard_yolo11s.onnx')).toBeVisible();
    await expect(page.getByText('4: no_helmet')).toBeVisible();

    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});

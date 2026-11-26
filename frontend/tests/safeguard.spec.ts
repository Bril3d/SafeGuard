import fs from 'node:fs';
import path from 'node:path';
import { expect, test } from '@playwright/test';

const SCREENSHOT_DIR =
  process.env.PLAYWRIGHT_SCREENSHOT_DIR || 'test-results/screenshots';

test.describe('SafeGuard End-to-End Workflows', () => {
  test('verifies every SafeGuard workflow and captures audit screenshots against live ONNX backend', async ({
    page,
  }) => {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

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

    // 1. Overview (Initial State) & Health Refresh
    await expect(page.getByRole('heading', { name: 'Operational Overview' })).toBeVisible();
    await expect(page.getByText('ONNX Model Ready')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('Version: EXP-E-yolo11s-onnx')).toBeVisible();
    await expect(page.getByText('Provider: CPUExecutionProvider')).toBeVisible();
    await expect(page.getByText('opset 17 / 640px')).toBeVisible();
    await page.getByRole('button', { name: 'Refresh Status' }).click();
    await expect(page.getByText('ONNX Model Ready')).toBeVisible({ timeout: 10_000 });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '01_overview.png'),
      fullPage: true,
    });

    // 2. Image Analysis (Client MIME validation + Canvas Overlay + OpenCV Visualization)
    await page.getByRole('button', { name: 'Image Analysis' }).click();
    await expect(page.getByRole('heading', { name: 'Image Analysis' })).toBeVisible();

    const uploadInput = page.getByTestId('image-upload-input');
    await uploadInput.setInputFiles({
      name: 'not_an_image.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('invalid-payload'),
    });
    await expect(page.getByText(/Unsupported file format/)).toBeVisible();

    const samplePath = path.resolve(
      process.cwd(),
      'public/examples/ul_ultralytics_ppe__val__24f563c3cd69.jpg',
    );
    await uploadInput.setInputFiles(samplePath);
    await expect(page.getByText(/Unsupported file format/)).toBeHidden();
    await page.getByRole('button', { name: 'Run Inspection' }).click();

    await expect(page.getByText('Detections Returned')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText('Server Inference / Total')).toBeVisible();
    await expect(page.getByLabel('Bounding box detection overlay')).toBeVisible();
    await page.waitForTimeout(250);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '02_image_analysis_canvas.png'),
      fullPage: true,
    });

    // Switch to OpenCV server-rendered output (/predict/visualize)
    await page.getByRole('button', { name: 'Server Annotated' }).click();
    const annotatedImg = page.getByAltText('Server-rendered OpenCV annotation output');
    await expect(annotatedImg).toBeVisible();
    await expect
      .poll(async () =>
        annotatedImg.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 0),
      )
      .toBe(true);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '03_image_analysis_server_annotated.png'),
      fullPage: true,
    });

    // 3. Example Gallery (One-Click Curated Sample Inference)
    await page.getByRole('button', { name: 'Example Gallery' }).click();
    await expect(page.getByRole('heading', { name: 'Example Gallery' })).toBeVisible();
    await page.getByRole('button', { name: 'Run Inference' }).nth(4).click();
    await expect(page.getByText(/Detected \d+ objects in/)).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByLabel('Bounding box detection overlay')).toBeVisible();
    await page.waitForTimeout(250);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '04_example_gallery.png'),
      fullPage: true,
    });

    // 4. Live Monitoring (Webcam Stream -> /predict/compliance -> ByteTrack + Compliance Alerts)
    await page.getByRole('button', { name: 'Live Monitoring' }).click();
    await expect(page.getByRole('heading', { name: 'Live Monitoring' })).toBeVisible();
    await page.getByRole('button', { name: 'Start Camera' }).click();
    await expect(page.getByRole('button', { name: 'Stop Camera' })).toBeVisible({
      timeout: 10_000,
    });
    await expect
      .poll(
        async () => {
          const txt = await page.locator('text=Processed Frames').locator('..').innerText();
          return Number(txt.replace(/\D+/g, ''));
        },
        { timeout: 25_000 },
      )
      .toBeGreaterThanOrEqual(4);
    await page.waitForTimeout(300);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '05_live_monitoring.png'),
      fullPage: true,
    });
    await page.getByRole('button', { name: 'Reset Track State' }).click();
    await page.getByRole('button', { name: 'Stop Camera' }).click();
    await expect(page.getByRole('button', { name: 'Start Camera' })).toBeVisible();

    // 5. Alert History & Human Review Queue
    await page.getByRole('button', { name: 'Alert History' }).click();
    await expect(
      page.getByRole('heading', { name: 'Alert History & Human Review Queue' }),
    ).toBeVisible();

    const alertRows = page.locator('tbody tr');
    const alertCount = await alertRows.count();
    if (process.env.PLAYWRIGHT_FAKE_VIDEO) {
      expect(alertCount).toBeGreaterThan(0);
    }
    if (alertCount > 0) {
      await alertRows.first().getByRole('button', { name: 'Verify' }).click();
      await expect(alertRows.first().getByText('verified potential event')).toBeVisible();
      if (alertCount > 1) {
        await alertRows.nth(1).getByRole('button', { name: 'Dismiss' }).click();
        await expect(alertRows.nth(1).getByText('dismissed false alarm')).toBeVisible();
      }
      await page.getByRole('button', { name: 'Verified', exact: true }).click();
      await expect(alertRows.first()).toBeVisible();
      await page.getByRole('button', { name: 'All', exact: true }).click();
    } else {
      await expect(page.getByText(/No potential alerts match this filter yet/)).toBeVisible();
    }
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '06_alert_history.png'),
      fullPage: true,
    });

    // 6. Session Analytics (Recharts Bar + Line Charts)
    await page.getByRole('button', { name: 'Analytics' }).click();
    await expect(page.getByRole('heading', { name: 'Session Analytics' })).toBeVisible();
    await expect(page.locator('.recharts-responsive-container svg').first()).toBeVisible({
      timeout: 15_000,
    });
    await page.waitForTimeout(300);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '07_analytics.png'),
      fullPage: true,
    });

    // 7. Model & Operational Monitoring
    await page.getByRole('button', { name: 'Model Monitoring' }).click();
    await expect(
      page.getByRole('heading', { name: 'Model & Operational Monitoring' }),
    ).toBeVisible();
    await expect(page.getByText('Out of')).toBeVisible();
    await expect(page.locator('tbody tr').first()).toBeVisible();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '08_model_monitoring.png'),
      fullPage: true,
    });

    // 8. Settings & Model Information
    await page.getByRole('button', { name: 'Settings & Model' }).click();
    await expect(
      page.getByRole('heading', { name: 'Settings & Model Information' }),
    ).toBeVisible();
    await expect(page.getByText('safeguard_yolo11s.onnx')).toBeVisible();
    await expect(page.getByText('4: no_helmet')).toBeVisible();
    await page.locator('#conf-slider').fill('0.35');
    await expect(page.getByText('0.35')).toBeVisible();
    await page.getByRole('button', { name: 'Restore Model Defaults' }).click();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '09_settings_model_info.png'),
      fullPage: true,
    });

    // 9. Overview (Populated Session Summary)
    await page.getByRole('button', { name: 'Overview' }).click();
    await expect(page.getByRole('heading', { name: 'Operational Overview' })).toBeVisible();
    await expect(page.getByText(/Across \d+ completed inference calls/)).toBeVisible();
    await expect(page.locator('tbody tr').first()).toBeVisible();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '10_overview_populated.png'),
      fullPage: true,
    });

    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});

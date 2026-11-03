import { test, expect } from "@playwright/test";

// Voice clone wizard (upload -> transcript -> name + spoken consent) with the backend mocked.
// Chromium's fake media device stands in for the microphone used to record consent.
test.use({
  permissions: ["microphone"],
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
});

test("clone a voice from an uploaded sample", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("bonclay_demo_mode", "true"));

  let cloned = false;
  let consentSent = false;
  await page.route("**/api/voices/clone", (route) => {
    const body = route.request().postDataBuffer()?.toString("latin1") ?? "";
    consentSent = body.includes('name="consent_audio"') && body.includes('name="consent_text"');
    cloned = true;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "v1", engine: "qwen3-tts" }),
    });
  });
  await page.route("**/api/voices", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        cloned
          ? [{ id: "v1", display_name: "E2E Voice", name: "E2E Voice", language: "en", created_at: new Date().toISOString() }]
          : []
      ),
    })
  );

  await page.goto("/dashboard/voices?new=1");
  const dialog = page.getByRole("dialog", { name: "Clone a voice" });
  await expect(dialog).toBeVisible();

  // Step 1: sample
  await dialog.getByLabel("Choose audio file").setInputFiles({
    name: "sample.wav",
    mimeType: "audio/wav",
    buffer: Buffer.alloc(2048),
  });
  await dialog.getByRole("button", { name: /continue/i }).click();

  // Step 2: transcript (skip it so no transcription call is needed)
  await dialog.getByRole("checkbox", { name: /skip transcript/i }).check();
  await dialog.getByRole("button", { name: /continue/i }).click();

  // Step 3: name + consent
  await dialog.getByLabel(/voice name/i).fill("E2E Voice");
  await dialog.getByRole("button", { name: /record consent/i }).click();
  await page.waitForTimeout(1500);
  await dialog.getByRole("button", { name: /^stop$/i }).click();
  await expect(dialog.getByText("Consent recorded")).toBeVisible();
  await dialog.getByRole("button", { name: /clone voice/i }).click();

  await expect(page.getByText("E2E Voice").first()).toBeVisible();
  expect(consentSent).toBe(true);
});

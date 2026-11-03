This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Performance budget (Lighthouse CI)

`.github/workflows/lighthouse.yml` runs Lighthouse on `/` and `/how-it-works` (production build, mobile emulation, median of 3 runs) on every pull request that touches `frontend/**` and on every push to `main`. Thresholds are in `lighthouserc.json`: performance 0.80 is an error (fails the job) and 0.90, LCP 2.5 s, TBT 300 ms, script size 460 KB and accessibility 0.9 are warnings. Baseline on 23 Oct 2026 (local, median of 3): `/` 85 with LCP 4.4 s, `/how-it-works` 87 with LCP 4.0 s, about 390 KB of script; raise the error bar as the site gets faster. CLS above 0.1 is an error. The job is separate from `ci.yml`, so it does not block the other jobs. To run it locally on port 3204 (stop anything else on that port first): `npm run perf:budget`, which builds and then runs the same config. Chrome must be installed; set `CHROME_PATH` if it is not in the default place.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

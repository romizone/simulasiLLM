# Deployment Guide (simulasiLLM)

## Recommended: Cloudflare Pages

This project is a static site (`index.html`, `style.css`, `app.js`), so Cloudflare Pages is a strong default.

### Why
- Free tier available
- Global CDN
- Easy custom domain setup
- Good fit for static frontend projects

### Steps
1. Create a GitHub repository and push this folder.
2. In Cloudflare dashboard, go to **Workers & Pages** -> **Create application** -> **Pages**.
3. Connect the GitHub repository.
4. Build settings:
   - Framework preset: `None`
   - Build command: *(leave empty)*
   - Build output directory: `/`
5. Deploy.

## Alternative 1: Netlify

### Steps
1. Create site from Git repository.
2. Build command: *(empty)*
3. Publish directory: `/`
4. Deploy.

## Alternative 2: Vercel

### Steps
1. Import Git repository.
2. Framework preset: `Other`.
3. Build command: *(empty)*
4. Output directory: `.`
5. Deploy.

## Alternative 3: GitHub Pages

### Steps
1. Push folder into a repository.
2. In repository settings, enable **Pages**.
3. Select branch/folder to publish (`main` + `/root` or `/docs`).
4. Wait for Pages URL.

## Local test

```bash
cd simulasiLLM
python3 -m http.server 8081
```

Open `http://127.0.0.1:8081/index.html`.

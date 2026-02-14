<p align="center">
  <img src="https://img.shields.io/badge/🤖_Transformer-Explainer-7c63ff?style=for-the-badge&labelColor=1f2437" alt="Transformer Explainer" />
</p>

<h1 align="center">🧠 Transformer Explainer</h1>
<h3 align="center">Interactive LLM Attention Simulation — GPT-2 Style</h3>

<p align="center">
  <em>Visualize how transformers think, one attention head at a time.</em>
</p>

<p align="center">
  <a href="https://simulasillm.vercel.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-simulasillm.vercel.app-5b9dff?style=for-the-badge&labelColor=1f2437" alt="Live Demo" />
  </a>
  &nbsp;
  <a href="https://paper-llm-attention.vercel.app/">
    <img src="https://img.shields.io/badge/📄_Research_Paper-Read_Now-2ec8a8?style=for-the-badge&labelColor=1f2437" alt="Research Paper" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white" alt="Vercel" />
  <img src="https://img.shields.io/badge/Zero_Dependencies-✅-brightgreen?style=flat-square" alt="Zero Dependencies" />
</p>

---

## 🔍 Overview

An interactive, educational web simulation that lets you **see inside a transformer model** — from tokenization to next-token prediction. No black boxes. No abstract equations. Just a live, explorable pipeline modeled after **GPT-2 Small**.

> 💡 Built for students, educators, and anyone curious about how Large Language Models actually work under the hood.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔤 **Token Embedding** | Watch input text split into tokens and map to embedding vectors with positional encoding |
| 🔑 **Q/K/V Inspector** | Examine Query, Key, and Value projections for each attention head |
| 🗺️ **Attention Heatmap** | Interactive matrix with causal masking — see which tokens attend to which |
| 📊 **Probability Distribution** | Real-time softmax output showing candidate tokens and their probabilities |
| ⚡ **Autoregressive Generation** | Generate tokens step-by-step and observe how each new token reshapes attention |
| 🎛️ **Sampling Controls** | Tune Temperature (0.3–1.5), Top-k (1–12), and generation length live |
| 👁️ **Multi-Head View** | Switch between attention heads to compare learned patterns |

---

## 🏗️ Architecture (Simulated)

```
┌─────────────────────────────────────────────────┐
│              GPT-2 Small (Simulation)            │
├──────────────────┬──────────────────────────────┤
│  🧱 Layers       │  12                          │
│  🧠 Attn Heads   │  4 (visual)                  │
│  📐 Hidden Size  │  16                          │
│  🔢 Head Dim     │  4                           │
│  🎭 Causal Mask  │  Active                      │
└──────────────────┴──────────────────────────────┘
```

---

## 🛠️ Tech Stack

| | Technology | Purpose |
|---|-----------|---------|
| 📄 | **HTML5** | Semantic layout & SVG attention diagrams |
| 🎨 | **CSS3** | Custom properties, Grid, Flexbox |
| ⚙️ | **Vanilla JS** | Simulation engine — zero dependencies |
| 🔤 | **Google Fonts** | Orbitron, Space Grotesk, IBM Plex Mono |
| 🌐 | **Vercel** | Static hosting & CDN |

> **No frameworks. No build step. No bundler.** Just clean, dependency-free code.

---

## 📁 Project Structure

```
simulasiLLM/
├── 📄 index.html        # UI layout — toolbar, attention SVG, token stream, panels
├── 🎨 style.css         # Styling with CSS custom properties
├── ⚙️ app.js            # Engine — tokenizer, attention math, rendering, generation
├── 🚀 DEPLOY.md         # Deployment guide (Vercel, Cloudflare, Netlify, GH Pages)
└── 📦 .github/
    └── workflows/       # CI/CD configuration
```

---

## 🚀 Getting Started

### Run Locally

No install required — just serve the static files:

```bash
git clone https://github.com/romizone/simulasiLLM.git
cd simulasiLLM
python3 -m http.server 8081
```

Open [http://127.0.0.1:8081](http://127.0.0.1:8081) in your browser.

Alternatively:

```bash
# Node.js
npx serve .

# PHP
php -S localhost:8081
```

### Deploy

See [DEPLOY.md](./DEPLOY.md) for guides on Vercel, Cloudflare Pages, Netlify, and GitHub Pages.

---

## ⚙️ How It Works

```
  📝 Input Text
       │
       ▼
  ┌──────────┐
  │ 🔤 Token │  Split text into tokens (Unicode-aware regex)
  │   izer   │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ 📐 Embed │  Map tokens → dense vectors + positional encoding
  │   ding   │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ 🔑 Q/K/V │  Project embeddings into Query, Key, Value spaces
  │  Project │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ 🗺️ Attn  │  score = (Q · Kᵀ) / √d_k  →  causal mask  →  softmax
  │  Matrix  │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ 🎲 Sample│  Apply temperature & top-k filtering
  │          │
  └────┬─────┘
       │
       ▼
  ⚡ Next Token  ──→  (autoregressive loop)
```

---

## 📄 Research Paper

<table>
  <tr>
    <td>📄</td>
    <td>
      <strong>"Simulating the Attention Mechanism in Large Language Models Based on the GPT-2 Architecture"</strong><br/><br/>
      <a href="https://paper-llm-attention.vercel.app/">📖 Read the full paper →</a>
    </td>
  </tr>
</table>

**Topics covered:**

- 🔤 Token processing pipeline (BPE tokenization, embedding, positional encoding)
- 📐 Mathematical formulation of scaled dot-product attention
- 🎭 Causal masking in autoregressive generation
- 🌡️ Temperature scaling and top-k sampling strategies
- 🏗️ GPT-2 Small architecture specifications

---

## 🔗 Links

| | Link |
|---|------|
| 🚀 | **Live Simulation** — [simulasillm.vercel.app](https://simulasillm.vercel.app/) |
| 📄 | **Research Paper** — [paper-llm-attention.vercel.app](https://paper-llm-attention.vercel.app/) |
| 💻 | **Source Code** — [github.com/romizone/simulasiLLM](https://github.com/romizone/simulasiLLM) |

---

## 📜 License

This project is open source and available for educational purposes.

---

<p align="center">
  <strong>Made with ❤️ by <a href="https://github.com/romizone">Romi Nur Ismanto</a></strong>
</p>

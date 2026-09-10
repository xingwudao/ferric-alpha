import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Ferric Alpha",
  description:
    "Ferric Alpha is a Rust-powered, Python-friendly factor analysis library inspired by alphalens, built on Polars for high-performance quantitative research.",
  base: "/ferric-alpha/",
  cleanUrls: true,
  head: [
    [
      "meta",
      {
        name: "keywords",
        content:
          "alphalens, factor analysis, quantitative finance, quant research, Rust, Polars, Python, tear sheet, information coefficient",
      },
    ],
    ["meta", { property: "og:title", content: "Ferric Alpha" }],
    [
      "meta",
      {
        property: "og:description",
        content:
          "High-performance factor analysis for Python and Rust, inspired by alphalens.",
      },
    ],
    ["meta", { property: "og:type", content: "website" }],
    [
      "meta",
      { property: "og:image", content: "/ferric-alpha/ferric-alpha-og.svg" },
    ],
  ],
  themeConfig: {
    logo: "/ferric-alpha-og.svg",
    search: {
      provider: "local",
    },
    nav: [
      { text: "Guide", link: "/guide/quickstart" },
      { text: "API", link: "/api/python" },
      { text: "Development", link: "/development/contributing" },
      { text: "GitHub", link: "https://github.com/xingwudao/ferric-alpha" },
    ],
    sidebar: [
      {
        text: "Guide",
        items: [
          { text: "Quickstart", link: "/guide/quickstart" },
          { text: "Data Model", link: "/guide/data-model" },
          { text: "Factor Preparation", link: "/guide/factor-preparation" },
          { text: "Performance Metrics", link: "/guide/performance-metrics" },
          { text: "Tear Sheets", link: "/guide/tear-sheets" },
          { text: "Rendering", link: "/guide/rendering" },
          { text: "Benchmarks", link: "/guide/benchmarks" },
        ],
      },
      {
        text: "API",
        items: [
          { text: "Python API", link: "/api/python" },
          { text: "Rust API", link: "/api/rust" },
        ],
      },
      {
        text: "Development",
        items: [
          { text: "Contributing", link: "/development/contributing" },
          { text: "Release Checks", link: "/development/release-checks" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/xingwudao/ferric-alpha" },
    ],
    footer: {
      message: "Released under the 0BSD license.",
      copyright: "Copyright Ferric Alpha contributors",
    },
  },
});

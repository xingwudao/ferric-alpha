# Third Party Notices

## Roboto static font

- Source: Android Open Source Project, `platform/external/roboto-fonts`
- Upstream path: `RobotoStatic-Regular.ttf`
- Vendored font path:
  `crates/ferric-alpha-render/assets/fonts/RobotoStatic-Regular.ttf`
- Vendored license path:
  `crates/ferric-alpha-render/assets/fonts/Roboto-NOTICE.txt`
- Font SHA-256:
  `06cba01eb71ea5cbd3a7df498910624db68953beead4be18fd91f8ec7dc72351`
- Font byte size: `305,656`
- License: Apache License 2.0

This compact Latin UI font is bundled in the `ferric-alpha-render` crates.io
package so Plotters bitmap rendering has a deterministic registered font
without shipping a multi-megabyte font asset.

## Noto Sans SC

- Source: Google Fonts
- Upstream path: `ofl/notosanssc/NotoSansSC[wght].ttf`
- Source commit: `0cf764bb712367b6079cbb4fd2353e6f54ec6850`
- Vendored font path:
  `crates/ferric-alpha-render/assets/fonts/NotoSansSC-wght.ttf`
- Vendored license path:
  `crates/ferric-alpha-render/assets/fonts/OFL.txt`
- Font SHA-256:
  `a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da`
- License SHA-256:
  `1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9`
- Font byte size: `17,772,300`
- License: SIL Open Font License 1.1

The font is preserved unmodified in the source repository as a development and
visual-regression asset. It is excluded from the `ferric-alpha-render` crates.io
package to keep the crate below registry upload limits. Native rendering uses
the platform sans-serif font resolved by Plotters.

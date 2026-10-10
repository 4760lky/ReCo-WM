# Media

| File | Content | Notes |
|---|---|---|
| `teaser.png` | Fig. 1 of the paper exported as PNG, ≤ 1600 px wide | Referenced by the root README |
| `results_curves.png` | Learning-curve figure (Fig. 5 a–d) | Optional |
| `demo.mp4` | Main demo video (simulation + real robot) | Keep ≤ 100 MB; tracked with Git LFS |
| `demo_thumbnail.png` | Poster frame used as the clickable link to the hosted video | |

## Hosting the demo video

1. **GitHub Release asset (recommended).** Create a release (e.g. `v1.0`) and attach `demo.mp4`; link the asset URL
   from the root README. Release assets have a 2 GB per-file limit and do not count against LFS quota.
2. **Inline playback.** GitHub plays an MP4 inline in Markdown only when the file was uploaded through the web
   editor / an issue comment (it is then served from `user-images.githubusercontent.com`). Paste that generated URL
   into the README under "Demo video".
3. **External hosting.** YouTube or Bilibili link with `demo_thumbnail.png` as the clickable image.

Recommended encoding: H.264, 1080p, 30 fps, AAC audio, `-movflags +faststart`.

# xkcd-artist

Hybrid fine-tuned Diffusion LoRA stick-figure art + 5-head ExplainXKCD scene planner + crisp hand-lettered speech-bubble, leader-line, and hover `alt`-text comic generator in JavaScript, Python, and Browser WebAssembly ONNX.

- **Interactive Studio**: [`docs/index.html`](./docs/index.html)
- **Empirical Science Report**: [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md)
- **Architecture Blueprint**: [`docs/flow-diagram.svg`](./docs/flow-diagram.svg)

```bash
npm install xkcd-artist
pip install xkcd-artist
```

## Quick start (JavaScript / ESM)

```js
import xkcdArtist from 'xkcd-artist';

const comic = xkcdArtist('Sudo make me a sandwich');

console.log(comic.title);  // 'SANDWICH'
console.log(comic.layout); // 'three_panel'
console.log(comic.cast);   // ['cueball', 'megan']
console.log(comic.alt);    // 'Proper User Policy apparently means Simon Says.'

const svg = comic.toSVG(); // Multi-panel hand-wobbled SVG with crisp speech bubbles
```

`xkcdArtist()` parses any prompt into multi-panel XKCD layouts (`single_panel`, `two_panel`, `three_panel`, `four_panel`, `annotated_chart`, `flowchart_diagram`), selects the recurring XKCD cast (`cueball`, `megan`, `black_hat`, `beret_guy`, `ponytail`, `hairbun`) and stage props (`laptop_desk`, `whiteboard`, `server_rack`, `pointer_stick`, `couch`, `chart_board`, `pedestal`), and composites hand-wobbled monochrome ink panels with 100% legible speech bubbles and hover `alt` text.

## Quick start (Python)

```python
from xkcd_artist import xkcd_artist

comic = xkcd_artist(
    "Machine learning pile of linear algebra stir the pile",
    backend="onnx",  # 'zero-dep' | 'onnx' | 'pytorch'
)

svg_markup = comic.to_svg()
png_bytes = comic.to_png("docs/samples/machine_learning_xkcd.png")
```

## Why Hybrid Diffusion LoRA + Crisp Speech-Bubble Compositing?

Pure image diffusion models hallucinate garbled text inside comic speech bubbles. `xkcd-artist` separates **visual ink geometry** from **typographic layout**:

1. **Speech-Bubble-Masked Real XKCD Panel Corpus (`226` PNGs + `891` ExplainXKCD transcripts)**: Existing speech-bubble text in the top `36%` of each training panel is masked out prior to fine-tuning on **NVIDIA L4 GPU (`24 GB VRAM`)**, so the Stable Diffusion UNet LoRA (`sd_xkcd_lora/adapter_model.safetensors`, `r=16`) and `XkcdArtistForGeneration` (`1.59M` params) learn pure Randall Munroe monochrome stick-figure ink geometry without text artifacts.
2. **5 Joint Structural Heads (`layout_head`, `prop_head`, `pose_head`, `bubble_box_head`, `abstention_head`)**: Predicts panel count, stage props, 12-DOF stick-figure joint angles, and non-overlapping upper-panel speech-bubble boxes (`cy + h/2 <= 0.48`).
3. **Crisp Hand-Lettered Speech Bubble & Alt-Text Compositor**: Renders razor-sharp hand-lettered dialogue, speaker leader lines, `2px` wobbled panel borders, and the bottom `alt` text bar in SVG and high-DPI PNG.

## License

MIT © [Hemanth.HM](https://h3manth.com)

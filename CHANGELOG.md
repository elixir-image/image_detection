# Changelog

## [0.4.0] 2026-05-23

### Updated

* Loosen the default crop in face detection.

## [0.3.0] 2026-05-21

### Updated

* Update `nx` and `exla` to `~> 0.12`.

## [0.2.0] 2026-05-02

### Added

* **`Image.FaceDetection`** — fast face detection with bounding boxes, confidence scores, and the five canonical facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner). Default model is [YuNet 2023-March](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) hosted at [`opencv/face_detection_yunet`](https://huggingface.co/opencv/face_detection_yunet) — MIT licensed, ~340 KB on disk, real-time on CPU. Functions: `detect/2`, `boxes/2`, `crop_largest/2`, `draw_boxes/3`. The `crop_largest/2` helper is the wire-in point for face-aware crop bias used by sibling `image_plug` (`gravity: :face`, ImageKit `z-`, Cloudflare `face-zoom`).

* **`Image.Background`** — class-agnostic foreground/background separation. `remove/2` returns the input image with the background made transparent (alpha mask applied); `mask/2` returns the foreground mask alone for custom compositing. Default model is [BiRefNet lite](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX) (MIT, ~210 MB), powered by Ortex.

* **`Image.Captioning`** — natural-language description of an image. `caption/2` returns a string like `"a man riding a horse with a bird of prey"`. Default model is [BLIP base](https://huggingface.co/Salesforce/blip-image-captioning-base) (BSD-3-Clause, ~990 MB), powered by Bumblebee. Heavy enough that it is not autostarted by default; configure `autostart: true` or add the child spec to your supervisor.

* **`Image.ZeroShot`** — classify an image against arbitrary labels you supply at call time, no retraining. `classify/3` returns `[%{label, score}]` sorted descending; `label/3` returns just the best label; `similarity/3` computes CLIP-space cosine similarity between two images. Default model is [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) (MIT, ~600 MB), powered by Bumblebee. Default prompt template `"a photo of {label}"` boosts accuracy on bare-noun labels; override or disable as needed.

* New flags `--background`, `--caption`, and `--zero-shot` for `mix image_vision.download_models` to pre-fetch the new defaults.

### Changed

* The `:files` list in `mix.exs` now ships `logo.jpg` so the docs render the project logo on hexdocs.pm.

See the [README](https://github.com/elixir-image/image_vision/blob/v0.2.0/README.md) for the full feature list and the [background](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/background.md), [captioning](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/captioning.md), and [zero-shot](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/zero_shot.md) guides for detail on the new tasks.

## ImageVision v0.1.0

`image_vision` is a thin, opinionated wrapper around the Elixir ML ecosystem (Bumblebee, Ortex, Nx) that sits next to the [`image`](https://hex.pm/packages/image) library. It exposes three vision tasks through a small API designed for developers who are not ML experts: pass a `t:Vix.Vips.Image.t/0` in, get useful results out. Strong, permissively-licensed defaults handle model selection, backend configuration, and weight downloads automatically.

### Highlights

* **Image classification** via `Image.Classification.classify/2` and `Image.Classification.labels/2` — returns ImageNet-1k labels with confidence scores. Default model is `facebook/convnext-tiny-224` (Apache 2.0, ~110 MB), powered by Bumblebee.

* **Image embeddings** via `Image.Classification.embed/2` — returns a 768-dim feature vector suitable for similarity search, clustering, or as input to a downstream classifier. Default model is `facebook/dinov2-base` (Apache 2.0, ~340 MB).

* **Object detection** via `Image.Detection.detect/2` — returns bounding boxes with class labels and scores across the 80 COCO classes. Default model is `onnx-community/rtdetr_r50vd` (Apache 2.0, ~175 MB), an NMS-free real-time transformer detector that beats YOLOv8 on COCO without YOLO's AGPL licensing constraints.

* **Promptable segmentation** via `Image.Segmentation.segment/2` — point, box, or multi-point prompts produce precise pixel masks via SAM 2. Default model is `SharpAI/sam2-hiera-tiny-onnx` (Apache 2.0, ~150 MB encoder + decoder).

* **Panoptic segmentation** via `Image.Segmentation.segment_panoptic/2` — every region in the image gets a class label across 133 COCO panoptic categories (things and stuff). Default model is `Xenova/detr-resnet-50-panoptic` (Apache 2.0, ~175 MB). Includes a baked-in canonical COCO panoptic id→label map so common stuff classes resolve correctly even on repos with incomplete `config.json` entries.

* **Result composition helpers** that return `t:Vix.Vips.Image.t/0` directly: `Image.Detection.draw_bbox_with_labels/3` (configurable opacity, stroke width, font size, palette), `Image.Segmentation.compose_overlay/3` (colour-coded overlay of all panoptic segments), and `Image.Segmentation.apply_mask/2` (mask as alpha channel for cutouts).

* **Automatic model weight management** via `ImageVision.ModelCache` — ONNX weights download from HuggingFace on first call and cache on disk. Cache directory is configurable via `config :image_vision, :cache_dir, ...`; defaults to an XDG-compliant per-user cache. Bumblebee weights use Bumblebee's own HF cache.

* **`mix image_vision.download_models`** task pre-fetches every default model so first-call latency is eliminated and the library can run offline. Pass `--classify`, `--detect`, or `--segment` to limit scope. Honours user overrides for the Bumblebee classifier and embedder.

* **Optional ML dependencies** — `:bumblebee`, `:nx`, and `:ortex` are all `optional: true` in `mix.exs`. The library compiles cleanly without them; each task module is compile-time gated on its underlying runtime so you only pay for what you use.

* **Strong, opinionated defaults** chosen for permissive licensing (Apache 2.0 / MIT only — no AGPL/GPL, no non-commercial), reasonable size (<500 MB), broad applicability, and proven quality. Power users can override every default through options or app config.

See the [README](https://github.com/elixir-image/image_vision/blob/v0.1.0/README.md) for installation, prerequisites (toolchain, disk space, Livebook Desktop), and quick-start examples. The [classification](https://github.com/elixir-image/image_vision/blob/v0.1.0/guides/classification.md), [detection](https://github.com/elixir-image/image_vision/blob/v0.1.0/guides/detection.md), and [segmentation](https://github.com/elixir-image/image_vision/blob/v0.1.0/guides/segmentation.md) guides cover each task in depth, including how to swap in alternative models.

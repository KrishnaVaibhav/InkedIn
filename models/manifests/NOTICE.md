# Model & Third-Party Notices

InkedIn's **code** is Apache-2.0 (see /LICENSE). InkedIn **does not bundle or
redistribute any model weights** — every model is downloaded on demand, by the
user, from its original source, into the local workspace cache. Each model
keeps its own license, listed here so users know what they are pulling.

| Model | Used for | Source | License | Redistributable? |
| --- | --- | --- | --- | --- |
| manga-colorization-v2 generator | `fast` mode | github.com/qweasdd/manga-colorization-v2 (mirror pinned by SHA-256 in `manga-colorization-v2.json`) | **none published** (research repo) | **No — local use only.** Long-term plan: replace with a properly licensed model (see Research.MD) |
| Counterfeit-V2.5 (SD1.5) | `ai` mode base | HF `gsdf/Counterfeit-V2.5` | CreativeML OpenRAIL-M | per OpenRAIL terms |
| stable-diffusion-v1-5 | `ai` fallback base | HF `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M | per OpenRAIL terms |
| ControlNet lineart-anime | `ai` structure | HF `lllyasviel/control_v11p_sd15s2_lineart_anime` | OpenRAIL | per OpenRAIL terms |
| IP-Adapter (sd15) | `ai` reference | HF `h94/IP-Adapter` | Apache-2.0 | yes |
| WD14 swinv2 tagger v3 | `ai` auto-prompts | HF `SmilingWolf/wd-swinv2-tagger-v3` | Apache-2.0 | yes |
| comic-text-and-bubble-detector (RT-DETR-v2) | text/bubble detection | HF `ogkalu/comic-text-and-bubble-detector` | Apache-2.0 | yes |
| manga-ocr-base | Japanese OCR | HF `kha-white/manga-ocr-base` | Apache-2.0 | yes |
| M2M100-418M | translation | HF `facebook/m2m100_418M` | MIT | yes |
| EasyOCR detection/recognition models | non-JA OCR | JaidedAI/EasyOCR releases | Apache-2.0 | yes |

## Vendored code

`python/inkedin_core/models/v2_nets.py` contains the inference-time network
architecture of manga-colorization-v2 (needed for weight compatibility),
which itself derives from AlacGAN and Tag2Pix. The upstream repository
publishes **no license**. It is included here as an interoperability shim;
if you are the author and object — or can grant a license — please open an
issue. The weights themselves are never distributed with this project.

## Key runtime dependencies

PyTorch (BSD-3), diffusers/transformers/accelerate/huggingface-hub
(Apache-2.0), OpenCV (Apache-2.0), Pillow (MIT-CMU), PyMuPDF (AGPL-3.0 **or**
commercial — note: PyMuPDF's AGPL applies to distribution of combined works;
evaluate before shipping closed binaries), FastAPI/uvicorn/pydantic (MIT),
py7zr (LGPL-2.1), rarfile (ISC; needs an external unar/bsdtar/unrar tool),
EasyOCR (Apache-2.0), onnxruntime (MIT).

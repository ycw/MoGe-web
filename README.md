# About

Infer normal maps locally in the browser. Powered by [MoGe](https://github.com/microsoft/MoGe) and previewed with [three.js](https://github.com/mrdoob/three.js).

Try the live demo at [MoGe-web](https://ycw.github.io/MoGe-web), created by [@ycwhk](https://x.com/ycwhk).

The model `moge-2-vits-normal-fp16.onnx` is an FP16 quantized export of [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal).

**Note:** Exported ONNX operators appear to lack smooth interpolation, causing ViT patch artifacts that are currently mitigated via post-process bilateral filtering.
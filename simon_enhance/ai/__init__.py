"""Neural-network features (background removal, denoising, upscaling).

``engine``   devices (NPU / GPU / CPU), runtimes (ONNX Runtime with OpenVINO, DirectML,
             CUDA; ncnn with Vulkan) and tiled inference
``models``   model registry, download with checksum verification
``features`` the actual image operations used by the app
"""

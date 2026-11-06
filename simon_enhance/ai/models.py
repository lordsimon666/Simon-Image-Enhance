"""Model registry. Small models ship with the app, large ones are downloaded on first
use from Hugging Face, pinned to a fixed commit and verified by SHA-256."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field

from ..store import data_dir

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")


@dataclass(frozen=True)
class ModelFile:
    name: str            # local file name
    url: str             # download URL ("" = bundled with the app)
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelInfo:
    id: str
    title: str
    license: str
    source: str
    files: tuple                       # ModelFile, first one is the ONNX graph
    ncnn: tuple = ()                   # optional (param, bin) bundled files for Vulkan
    tile: int = 0                      # fixed tile size (0 = whole image at `fixed`)
    fixed: tuple = ()                  # fixed (h, w) input size
    scale: int = 1
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def download_size(self) -> int:
        return sum(f.size for f in self.files if f.url and not os.path.exists(path_of(f)))


_HF = "https://huggingface.co"

MODELS = {
    "birefnet": ModelInfo(
        id="birefnet",
        title="BiRefNet-lite (Hintergrund entfernen)",
        license="MIT",
        source="https://huggingface.co/onnx-community/BiRefNet_lite-ONNX",
        files=(ModelFile("birefnet_lite.onnx",
                         f"{_HF}/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22ba131738a16dff04aab8bdf8dc32e3ac1/onnx/model.onnx",
                         "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333", 224005088),),
        fixed=(1024, 1024),
    ),
    "scunet": ModelInfo(
        id="scunet",
        title="SCUNet real (Entrauschen)",
        license="Apache-2.0 / MIT (Kai Zhang et al.)",
        source="https://huggingface.co/Heliosoph/scunet-onnx (Original: github.com/cszn/SCUNet)",
        files=(ModelFile("scunet_color_real_psnr.onnx",
                         f"{_HF}/Heliosoph/scunet-onnx/resolve/6d11417ee2fbcc73783c502a238ac115097754fe/scunet_color_real_psnr.onnx",
                         "231be201ab413dbc999d7951caa9844846b93a12a40a41e037d6b5888ed4e88c", 3798678),
               ModelFile("scunet_color_real_psnr.onnx.data",
                         f"{_HF}/Heliosoph/scunet-onnx/resolve/6d11417ee2fbcc73783c502a238ac115097754fe/scunet_color_real_psnr.onnx.data",
                         "98825ea1210b641c71e5f052f582c70c49fd44b35387ebe2c034268c17df3feb", 73138176)),
        tile=256,
    ),
    "realesr": ModelInfo(
        id="realesr",
        title="Real-ESRGAN general x4v3 (Hochskalieren)",
        license="BSD-3-Clause (Xintao Wang)",
        source="https://github.com/xinntao/Real-ESRGAN",
        files=(ModelFile("realesr-general-x4v3.onnx", "",
                         "1940a93ee08283a0a7286183186357b1688fe9fa8ede74604b424586aaddf112", 4866417),),
        ncnn=("realesr-general-x4v3.ncnn.param", "realesr-general-x4v3.ncnn.bin"),
        tile=192,
        scale=4,
    ),
}


def models_dir() -> str:
    d = os.path.join(data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def path_of(f: ModelFile) -> str:
    if not f.url:
        return os.path.join(ASSETS, f.name)
    return os.path.join(models_dir(), f.name)


def is_available(model_id: str) -> bool:
    return all(os.path.exists(path_of(f)) for f in MODELS[model_id].files)


def onnx_path(model_id: str) -> str:
    return path_of(MODELS[model_id].files[0])


def ncnn_paths(model_id: str):
    m = MODELS[model_id]
    if not m.ncnn:
        return None
    p = tuple(os.path.join(ASSETS, n) for n in m.ncnn)
    return p if all(os.path.exists(x) for x in p) else None


class DownloadCancelled(Exception):
    pass


def download(model_id: str, progress=None, opener=urllib.request.urlopen) -> None:
    """Download missing files of a model. ``progress(done_bytes, total_bytes)`` may
    return False to cancel. Files are verified before they are used."""
    m = MODELS[model_id]
    todo = [f for f in m.files if f.url and not os.path.exists(path_of(f))]
    total = sum(f.size for f in todo)
    done = 0
    for f in todo:
        dst = path_of(f)
        tmp = dst + ".part"
        h = hashlib.sha256()
        with opener(f.url, timeout=60) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress is not None and progress(done, total) is False:
                    out.close()
                    os.remove(tmp)
                    raise DownloadCancelled()
        if h.hexdigest() != f.sha256:
            os.remove(tmp)
            raise IOError(f"Prüfsumme von {f.name} stimmt nicht – Download beschädigt oder verändert.")
        os.replace(tmp, dst)

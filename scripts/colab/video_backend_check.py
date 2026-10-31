"""영상 디코딩 백엔드(`torchcodec`/`pyav`) 실제 가용성 확인.

RoCo 영상은 h264다(이번 요청에서 확인). `LeRobotDataset`의
`get_safe_default_codec()`(`third_party/lerobot/.../datasets/video_utils.py:117`)은
`importlib.util.find_spec("torchcodec")`로 **패키지 import 가능 여부만** 보고
`torchcodec`을 기본 디코더로 고른다 — **그런데 import가 되는 것과 실제로 디코딩이
되는 것은 다르다**: 이 샌드박스에서 직접 재현 확인한 바로, `torchcodec` 파이썬
패키지는 import는 되지만 `VideoDecoder`를 실제로 쓰는 순간
(`torch.ops.load_library`가 `libavutil.so.*` 등 시스템 FFmpeg 공유 라이브러리를
찾는 지점에서) `RuntimeError: Could not load libtorchcodec`로 죽는다(시스템에
FFmpeg 공유 라이브러리가 없어서) — 반면 `pyav`는 같은 환경에서 실제로 h264 영상을
끝까지 디코딩했다(자체 바이너리 휠에 FFmpeg를 번들링하는 경우가 많아서로 보임).

그래서 이 모듈은 "import가 되는지"가 아니라 **작은 합성 h264 클립을 실제로 인코딩
(PyAV로)하고 디코딩까지 해보고** 판정한다. `libsvtav1`은 이 경로에 전혀 등장하지
않는다 — 레포 코드를 직접 확인한 결과, `vcodec="libsvtav1"`은
`LeRobotDataset.create()`/`dataset_tools.py`의 **인코딩**(새 데이터셋을 만들 때)
경로에만 있고, 디코딩(`decode_video_frames`/`get_safe_default_codec`/
`VideoDecoderCache`)에는 전혀 나오지 않는다(grep으로 확인됨) — 즉 "libsvtav1
인코딩 의존성이 디코딩 경로에 섞여 있다"는 우려는 이 레포에서는 **해당되지
않는다**(검증됨, 제거할 것이 없음). RoCo 영상을 받아서 읽기만 하는 이 스크립트는
어차피 `LeRobotDataset.create()`를 호출하지 않으므로 `libsvtav1`과 아예 무관하다."""

from __future__ import annotations

import dataclasses
import importlib.util
import io


def _make_test_h264_clip(width: int = 32, height: int = 32, num_frames: int = 4, fps: int = 10) -> bytes:
    """PyAV로 작은 h264 클립을 메모리에서 인코딩한다(실제 RoCo 영상과 같은 코덱)."""
    import av
    import numpy as np

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        value = i * 255 // max(num_frames - 1, 1)
        frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value, dtype=np.uint8), format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


def _try_decode_with_pyav(clip_bytes: bytes) -> bool:
    import av

    container = av.open(io.BytesIO(clip_bytes))
    try:
        count = sum(1 for _ in container.decode(video=0))
    finally:
        container.close()
    return count > 0


def _try_decode_with_torchcodec(clip_bytes: bytes) -> bool:
    import tempfile

    from torchcodec.decoders import VideoDecoder

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(clip_bytes)
        f.flush()
        decoder = VideoDecoder(f.name)
        return len(decoder) > 0 and decoder[0] is not None


@dataclasses.dataclass
class VideoBackendStatus:
    pyav_importable: bool
    pyav_can_decode_h264: bool
    torchcodec_importable: bool
    torchcodec_can_decode_h264: bool
    selected_backend: str | None  # lerobot의 get_safe_default_codec()과 같은 우선순위(torchcodec > pyav)
    detail: str

    @property
    def any_backend_works(self) -> bool:
        return self.selected_backend is not None


def check_video_backends() -> VideoBackendStatus:
    """실제로 h264 클립 하나를 인코딩+디코딩해보고 어느 백엔드가 쓸 수 있는 상태인지
    판정한다. 둘 다 안 되면 `selected_backend=None` — 실제 영상이 있는 데이터셋은
    로드 자체가 불가능하다는 뜻이므로, 호출자는 이 경우를 실패로 다뤄야 한다."""
    pyav_importable = importlib.util.find_spec("av") is not None
    torchcodec_importable = importlib.util.find_spec("torchcodec") is not None

    clip_bytes: bytes | None = None
    pyav_ok = False
    if pyav_importable:
        try:
            clip_bytes = _make_test_h264_clip()
            pyav_ok = _try_decode_with_pyav(clip_bytes)
        except Exception:  # noqa: BLE001 - 가용성 "확인"이 목적이라 예외 자체가 결과다
            pyav_ok = False

    torchcodec_ok = False
    if torchcodec_importable and clip_bytes is not None:
        try:
            torchcodec_ok = _try_decode_with_torchcodec(clip_bytes)
        except Exception:  # noqa: BLE001
            torchcodec_ok = False

    if torchcodec_ok:
        selected = "torchcodec"
    elif pyav_ok:
        selected = "pyav"
    else:
        selected = None

    detail = (
        f"pyav(import={pyav_importable}, h264 디코딩={pyav_ok}), "
        f"torchcodec(import={torchcodec_importable}, h264 디코딩={torchcodec_ok}) -> 선택={selected!r}"
    )
    return VideoBackendStatus(
        pyav_importable=pyav_importable,
        pyav_can_decode_h264=pyav_ok,
        torchcodec_importable=torchcodec_importable,
        torchcodec_can_decode_h264=torchcodec_ok,
        selected_backend=selected,
        detail=detail,
    )

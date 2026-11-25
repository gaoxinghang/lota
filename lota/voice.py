"""用本机 Breeze TTS 2 播报最终回复。"""

import os
from functools import cache
from pathlib import Path

BREEZE_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"


@cache
def _load_model():
  """加载 Breeze 4-bit 模型，并在本次会话中复用。

  Returns:
      MLX Audio 的 Breeze 语音模型。

  Raises:
      ImportError: 未安装语音依赖。
  """
  from mlx_audio.tts.utils import load_model

  return load_model(BREEZE_MODEL)


def speak(text: str) -> None:
  """用本地参考音频和台词边合成边播放。

  Args:
      text: 要播报的最终回复。

  Raises:
      ValueError: 未配置参考音频或对应台词。
      FileNotFoundError: 找不到参考音频。
      ImportError: 未安装语音依赖。
      RuntimeError: 模型没有生成音频。
  """
  ref_path = os.getenv("LOTA_VOICE_REFERENCE", "").strip()
  ref_text = os.getenv("LOTA_VOICE_TRANSCRIPT", "").strip()
  if not ref_path or not ref_text:
    raise ValueError(
      "请在 .env 设置 LOTA_VOICE_REFERENCE 和 LOTA_VOICE_TRANSCRIPT"
    )

  reference = Path(ref_path).expanduser()
  if not reference.is_file():
    raise FileNotFoundError(f"找不到参考音频：{reference}")

  from mlx.core import zeros
  from mlx_audio.tts.audio_player import AudioPlayer

  model = _load_model()
  player = AudioPlayer(sample_rate=model.sample_rate)
  player.min_buffer_seconds = 0.35
  has_audio = False
  try:
    for result in model.generate(
      text=text,
      ref_audio=str(reference),
      ref_text=ref_text,
      stream=True,
      streaming_interval=0.5,
      verbose=False,
    ):
      if result.audio.size:
        player.queue_audio(result.audio)
        has_audio = True
    if not has_audio:
      raise RuntimeError("Breeze 没有生成音频")
    player.queue_audio(zeros((model.sample_rate // 2,)))
    player.wait_for_drain()
  finally:
    player.flush()

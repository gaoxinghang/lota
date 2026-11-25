"""离线验证语音片段的播放顺序和资源清理。"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from lota.voice import speak


class VoiceTest(unittest.TestCase):
  """用假的 Breeze 模型和播放器验证流式播报。"""

  def setUp(self):
    """配置测试参考音频并替换模型和播放器。"""
    directory = self.enterContext(TemporaryDirectory())
    self.reference = Path(directory) / "reference.wav"
    self.reference.touch()
    self.enterContext(patch.dict("os.environ", {
      "LOTA_VOICE_REFERENCE": str(self.reference),
      "LOTA_VOICE_TRANSCRIPT": "参考台词",
    }))
    self.model = Mock(sample_rate=24000)
    self.player = Mock()
    self.player_factory = Mock(return_value=self.player)
    self.zeros = Mock()
    self.enterContext(patch("lota.voice._load_model", return_value=self.model))
    self.enterContext(patch.dict(sys.modules, {
      "mlx.core": SimpleNamespace(zeros=self.zeros),
      "mlx_audio.tts.audio_player": SimpleNamespace(
        AudioPlayer=self.player_factory,
      ),
    }))


  def test_chunks_are_queued_before_generation_finishes(self):
    """第一块立即入队；末尾加半秒静音后等待播放结束。"""
    first = SimpleNamespace(size=100)
    second = SimpleNamespace(size=50)

    def generate(**kwargs):
      """产出两块音频，并检查第一块在生成结束前已入队。"""
      self.assertTrue(kwargs["stream"])
      self.assertEqual(kwargs["ref_audio"], str(self.reference))
      self.assertEqual(kwargs["ref_text"], "参考台词")
      yield SimpleNamespace(audio=first)
      self.player.queue_audio.assert_called_once_with(first)
      self.player.wait_for_drain.assert_not_called()
      yield SimpleNamespace(audio=second)

    self.model.generate.side_effect = generate
    speak("你好")

    self.player_factory.assert_called_once_with(sample_rate=24000)
    self.zeros.assert_called_once_with((12000,))
    self.assertEqual(self.player.mock_calls, [
      call.queue_audio(first),
      call.queue_audio(second),
      call.queue_audio(self.zeros.return_value),
      call.wait_for_drain(),
      call.flush(),
    ])


  def test_interrupt_flushes_partial_playback(self):
    """生成被打断时丢弃未播完的音频，不尝试继续播放。"""
    def generate(**kwargs):
      """模拟在首块音频产出之后中断。"""
      yield SimpleNamespace(audio=SimpleNamespace(size=100))
      raise KeyboardInterrupt

    self.model.generate.side_effect = generate
    with self.assertRaises(KeyboardInterrupt):
      speak("你好")
    self.zeros.assert_not_called()
    self.player.wait_for_drain.assert_not_called()
    self.player.flush.assert_called_once_with()

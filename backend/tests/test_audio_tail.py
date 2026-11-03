import numpy as np

from services.audio_tail import trim_tail

SR = 24000
TEXT = "नमस्ते आयुष मीटिंग जल्द ही समाप्त होने वाली है हम बाद में बात करने के लिए तैयार हैं"  # ~63 letters


def tone(seconds, level=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_babble_after_a_pause_is_cut():
    speech = np.concatenate([tone(4.0), silence(0.4), tone(3.6)])          # ~8 s of real speech
    clip = np.concatenate([speech, silence(0.5), tone(3.0, 0.1)])          # 3 s of babble after a pause
    out = trim_tail(clip, SR, TEXT)
    assert len(out) / SR < 8.6
    assert len(out) / SR > 7.9


def test_a_normal_clip_is_left_alone():
    clip = np.concatenate([tone(4.0), silence(0.4), tone(3.6)])
    assert trim_tail(clip, SR, TEXT) is clip


def test_too_long_clip_without_a_pause_is_left_alone():
    clip = tone(14.0)
    assert trim_tail(clip, SR, TEXT) is clip


def test_never_cuts_before_the_text_could_be_spoken():
    clip = np.concatenate([tone(1.0), silence(0.6), tone(12.0)])           # pause far too early to be the end
    assert trim_tail(clip, SR, TEXT) is clip


def test_short_text_is_not_touched():
    clip = np.concatenate([tone(1.0), silence(0.6), tone(5.0)])
    assert trim_tail(clip, SR, "नमस्ते") is clip


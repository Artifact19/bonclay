"""services/asr_scoring.py: the same Hinglish words in Roman and Devanagari must score as the same words."""
import pytest

from services import asr_scoring as sc


@pytest.mark.parametrize("roman, devanagari", [
    ("kitna hai", "कितना है"),
    ("mera naam rahul hai", "मेरा नाम राहुल है"),
    ("kal subah nau baje", "कल सुबह नौ बजे"),
    ("bahut shukriya", "बहुत शुक्रिया"),
    ("mujhe ek glass paani aur do chai chahiye", "मुझे एक ग्लास पानी और दो चाय चाहिए"),
])
def test_roman_and_devanagari_fold_to_the_same_spelling(roman, devanagari):
    assert sc.fold(roman) == sc.fold(devanagari), (sc.fold(roman), sc.fold(devanagari))


def test_a_perfect_devanagari_transcript_of_a_roman_reference_scores_near_zero_after_folding():
    ref = "Kal subah nau baje doctor se appointment hai."
    hyp = "कल सुबह नौ बजे डॉक्टर से अपॉइंटमेंट है।"
    s = sc.hinglish_scores(ref, hyp)
    assert s["wer_raw"] >= 0.9                 # the old metric: every word "wrong"
    assert s["wer_folded"] <= 0.3              # only the two English loanwords keep a residue
    assert s["cer_folded"] <= 0.1
    assert s["hyp_devanagari"] is True
    # pure Hindi words come out identical
    assert sc.hinglish_scores("Kal subah nau baje hai.", "कल सुबह नौ बजे है।")["wer_folded"] == 0.0


def test_spelling_variants_fold_together():
    for a, b in [("shukriya", "shukriyaa"), ("bhej dijiye", "bhej deejiye"), ("paani", "pani"), ("chai", "chaai"),
                 ("kharida", "khareeda"), ("pahunchegi", "pahunchegee"), ("hoon", "hun"), ("Mumbai", "mumbai"),
                 ("zaroor", "jaroor"), ("wahan", "vahan"), ("naam", "nam"), ("kitna", "kitanA")]:
        assert sc.fold(a) == sc.fold(b), (a, b, sc.fold(a), sc.fold(b))


def test_real_errors_still_count():
    s = sc.hinglish_scores("Mujhe ek glass paani aur do chai chahiye.", "Mujhe ek glass paani chahiye.")
    assert 0 < s["wer_folded"] < 1 and 0 < s["cer_folded"] < 1
    assert sc.hinglish_scores("kal subah", "")["wer_folded"] == 1.0
    assert sc.fold("") == "" and sc.wer("", "") == 0.0


def test_devanagari_digits_and_punctuation_are_normalised():
    assert sc.fold("९ बजे") == sc.fold("9 baje")
    assert sc.fold("Yeh train Mumbai kab pahunchegi, kuch pata hai?") == sc.fold("yeh train mumbai kab pahunchegi kuch pata hai")

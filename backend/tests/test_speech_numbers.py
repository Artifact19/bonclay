"""Digits in a Hindi reply are rewritten for the Hindi voice (services/speech_numbers.py): espeak-ng, which Kokoro uses
for Hindi, splits "1,60,000" at its commas ("ek, saath, shunya shunya shunya"), skips "₹" and reads a phone number as
one big amount. The grounding prompt now asks for digits in every language, so this matters on every Hindi call."""
import asyncio

import pytest

from services.speech_numbers import hindi_numbers_for_speech as speak
from tests.test_offline_agents import env  # noqa: F401  (a fixture)


@pytest.mark.parametrize("text, said", [
    # Indian and Western digit grouping, with and without a currency mark
    ("MBA की सालाना फीस ₹1,60,000 है।", "MBA की सालाना फीस 160000 रुपये है।"),
    ("फीस 2,50,00,000 रुपये", "फीस 25000000 रुपये"),
    ("कुल 1,500 सीटें", "कुल 1500 सीटें"),
    ("फीस Rs. 500 है, फॉलो-अप Rs 1,500/- रुपये।", "फीस 500 रुपये है, फॉलो-अप 1500 रुपये।"),
    ("ICU का किराया ₹9,000 प्रति दिन", "ICU का किराया 9000 रुपये प्रति दिन"),
    ("₹2500000 का पैकेज", "2500000 रुपये का पैकेज"),        # an amount without commas is not a phone number
    ("फीस ₹१,६०,००० है", "फीस 160000 रुपये है"),             # Devanagari digits
    # phone numbers: digit by digit
    ("हेल्पलाइन 020 5550 1210 है।", "हेल्पलाइन 0 2 0 5 5 5 0 1 2 1 0 है।"),
    ("कॉल करें +91 98765 43210 पर", "कॉल करें 9 8 7 6 5 4 3 2 1 0 पर"),
    ("नंबर 9876543210 है", "नंबर 9 8 7 6 5 4 3 2 1 0 है"),
    ("500 - 1000 रुपये", "500 - 1000 रुपये"),                # a range is not a phone number
    # clock times as they are said
    ("ओपीडी 9:30 AM से 1:00 PM तक खुली है।", "ओपीडी सुबह साढ़े 9 बजे से दोपहर 1 बजे तक खुली है।"),
    ("शाम 16:00 बजे से", "शाम 4 बजे से"),
    ("कार्डियोलॉजी 9 AM से 1 PM", "कार्डियोलॉजी सुबह 9 बजे से दोपहर 1 बजे"),
    ("12:45 PM", "दोपहर पौने 1 बजे"),
    ("1:30 PM", "दोपहर डेढ़ बजे"),
    ("4:10 PM", "शाम 4 बजकर 10 मिनट"),
    # decimals and percentages
    ("फीस 1.5 लाख है", "फीस डेढ़ लाख है"),
    ("2.5 लाख", "ढाई लाख"),
    ("20% छूट", "20 प्रतिशत छूट"),
    ("ग्रेड 7.25", "ग्रेड 7 दशमलव 2 5"),
    # left alone
    ("B.Tech Mechanical Engineering में कुल 60 सीटें उपलब्ध हैं।", "B.Tech Mechanical Engineering में कुल 60 सीटें उपलब्ध हैं।"),
    ("नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?", "नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?"),
    ("", ""),
])
def test_hindi_numbers_for_speech(text, said):
    assert speak(text) == said


def test_the_hindi_voice_reads_the_rewritten_amount_in_lakh():
    """With the real G2P of Kokoro's Hindi voice (misaki + espeak-ng), when it is installed here."""
    try:
        from services.espeak import configure_espeak
        configure_espeak()
        from misaki import espeak
        g2p = espeak.EspeakG2P(language="hi")
        before = g2p("फीस ₹1,60,000 है।")[0]
    except Exception as e:  # noqa: BLE001 - no espeak-ng / misaki on this machine
        pytest.skip(f"Hindi G2P not available: {e}")
    after = g2p(speak("फीस ₹1,60,000 है।"))[0]
    assert "lˈaːkʰ" not in before                 # the raw digits: "ek, saath, shunya shunya shunya"
    assert "lˈaːkʰ" in after and "hˈʌɟaːɾ" in after and "ɾ" in after.split("hˈʌɟaːɾ")[1]  # lakh, hazaar, rupaye


def test_turn_spoken_rewrites_numbers_for_hindi_only(env):
    hi, lang = asyncio.run(env._turn_spoken("hi", "फीस ₹1,60,000 है।", False))
    assert lang == "hi" and hi == "फीस 160000 रुपये है।"
    en, lang = asyncio.run(env._turn_spoken("en", "The fee is ₹1,60,000.", False))
    assert lang == "en" and en == "The fee is ₹1,60,000."
    # Hinglish is spoken by the Hindi voice from Devanagari: its numbers are rewritten too
    hg, lang = asyncio.run(env._turn_spoken("hinglish", "Fees ₹1,60,000 hai.", False, use_llm=False))
    assert lang == "hi" and "160000 रुपये" in hg

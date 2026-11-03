"""First clause, not first sentence (services/sentences.py, TURN_FIRST_CLAUSE=on, the default): the first piece of a
reply goes to the voice at the first comma, semicolon, colon or dash once 4 words are in, or at the conjunction / 8th
word when there is none. English, Hindi (Devanagari) and Hinglish; numbers, times, abbreviations and URLs are never
cut; the result does not depend on how the text was chopped into streamed deltas; later pieces are plain sentences."""
import pytest

from services import sentences as sentences_mod
from services.sentences import TURN_MAX_CHARS, turn_splitter


def run(text, step=3, **kw):
    sp = turn_splitter(**kw)
    out = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    return out + sp.flush()


def first(text, **kw):
    return run(text, **kw)[0]


# ------------------------------------------------------------------------------------------------ clause marks

@pytest.mark.parametrize("text, expected", [
    ("Thank you for calling, how can I help you today?", "Thank you for calling,"),
    ("Yes we can see you tomorrow morning; the doctor is free after ten o'clock.", "Yes we can see you tomorrow morning;"),
    ("Here are the opening hours: nine to five on weekdays.", "Here are the opening hours:"),
    ("We open at nine in the morning — and close at five.", "We open at nine in the morning —"),
    ("The clinic is open from nine to five on weekdays, and the pharmacy stays open until seven.",
     "The clinic is open from nine to five"),   # eight words arrive before the comma: the eighth word ends the piece
])
def test_a_clause_mark_ends_the_first_piece_once_four_words_are_in(text, expected):
    assert first(text) == expected


def test_the_comma_after_the_fourth_word_is_enough_and_the_third_is_not():
    assert first("Please call us back, we are open all day long.") == "Please call us back,"
    assert run("Sure, the clinic is open on Sunday too.") == ["Sure, the clinic is open on Sunday too."]
    assert run("Thanks for calling, how can I help?")[0] == "Thanks for calling, how can I help?"  # three words only


def test_the_first_piece_is_emitted_the_moment_the_mark_arrives():
    sp = turn_splitter()
    got = []
    for word in "Thank you for calling, how can I help you today?".split():
        got += sp.feed(word + " ")
        if got:
            break
    assert got == ["Thank you for calling,"]            # before "how" has even been written


def test_a_comma_that_ends_the_text_so_far_is_a_boundary_unless_a_digit_precedes_it():
    sp = turn_splitter()
    assert sp.feed("Thank you for calling,") == ["Thank you for calling,"]
    sp = turn_splitter()
    assert sp.feed("Please pay 1,") == []  # "1," may be the start of 1,500
    assert sp.feed("500 rupees at the desk") == []
    assert sp.flush() == ["Please pay 1,500 rupees at the desk"]


# ------------------------------------------------------------------------------------------------ never cut

def test_numbers_times_decimals_and_urls_are_never_cut():
    text = "The surgery costs 1,500 rupees at 10:30 including 4.5 percent tax, see https://bonclay.in/pay for details."
    pieces = run(text)
    assert " ".join(pieces) == text
    for piece in pieces:
        assert not piece.endswith(("1,", "10:", "4.", "https:", "https://bonclay.in/pay,")) or piece.endswith("tax,")
    assert "1,500" in pieces[0] and "10:30" in pieces[0]
    assert all("https://bonclay.in/pay" in p for p in pieces if "bonclay" in p)


@pytest.mark.parametrize("text", [
    "Dr. Kulkarni sees patients on Mondays and Tuesdays.",
    "Please bring items such as id cards, e.g. a passport or licence.",
    "Please bring documents, i.e. your id card and report.",
    "We close at 5 p.m. on Saturdays unless noted.",
])
def test_abbreviations_do_not_end_a_sentence_or_the_first_piece(text):
    pieces = run(text)
    assert " ".join(pieces) == text
    assert not any(p.endswith(("Dr.", "e.g.", "i.e.", "p.m.")) for p in pieces[:-1])


def test_the_comma_after_e_g_or_i_e_is_not_a_boundary():
    pieces = run("We take payments in many ways, e.g., cash and card and UPI today.")
    assert pieces[0] == "We take payments in many ways,"
    pieces = run("You can pay in several forms e.g., cash or card or UPI today please.")
    assert not pieces[0].endswith("e.g.,")


def test_a_time_with_a_colon_is_not_a_clause():
    assert first("The clinic opens at 10:30 sharp on weekdays and stays open late.").startswith("The clinic opens at 10:30")


def test_a_url_is_not_cut_at_its_colon():
    pieces = run("You can book online at https://bonclay.example/book today, thank you very much for asking.")
    assert "https://bonclay.example/book" in pieces[0]


# ------------------------------------------------------------------------------------------------ the 8-word fallback

def test_without_a_mark_the_piece_ends_before_the_conjunction_or_at_the_eighth_word():
    text = "The dentist consultation costs four hundred rupees and the paediatrician charges three hundred."
    assert first(text) == "The dentist consultation costs four hundred rupees"
    text = "The doctor will see every patient in the clinic whenever the nurse calls the next name loudly."
    assert first(text) == "The doctor will see every patient in the clinic"   # "the" cannot end a piece: one word more
    assert " ".join(run(text)) == text


def test_the_fallback_never_stops_on_a_word_that_cannot_end_a_phrase():
    pieces = run("We keep the clinic open from nine to five on weekdays for everyone who needs us.")
    assert not pieces[0].split()[-1].lower() in ("the", "of", "to", "in", "on", "at", "for", "from", "and")


def test_the_fallback_does_not_part_a_number_from_its_unit():
    pieces = run("The surgery costs 1,500 rupees including the 4.5 percent tax for the whole family today.")
    assert pieces[0].endswith("4.5 percent") and pieces[1].startswith("tax")


def test_a_sentence_that_ends_before_the_eighth_word_stays_whole():
    assert run("Yes, we are open on Sunday.") == ["Yes, we are open on Sunday."]
    assert run("We open at nine. Please come then.") == ["We open at nine.", "Please come then."]


def test_a_mark_inside_the_first_eight_words_wins_over_the_fallback():
    pieces = run("Our clinic is open from nine, to five Monday to Saturday.")
    assert pieces[0] == "Our clinic is open from nine,"


# ------------------------------------------------------------------------------------------------ Hindi and Hinglish

def test_hindi_clause_mark_and_danda():
    pieces = run("जी हाँ, आपकी बुकिंग हो गई है, कृपया सुबह दस बजे आइए।")
    assert pieces[0] == "जी हाँ, आपकी बुकिंग हो गई है,"           # the first comma has two words before it: skipped
    assert pieces[1] == "कृपया सुबह दस बजे आइए।"


def test_a_short_hindi_sentence_ends_at_the_danda_whatever_its_length():
    assert run("जी हाँ। आपकी बुकिंग हो गई है।") == ["जी हाँ।", "आपकी बुकिंग हो गई है।"]


def test_hindi_without_a_mark_is_cut_at_the_eighth_word_not_before_a_postposition():
    text = "क्लिनिक सोमवार से शनिवार सुबह नौ बजे से रात आठ बजे तक खुलता है और रविवार को सिर्फ आपातकालीन मरीजों के लिए खुला रहता है।"
    pieces = run(text)
    assert " ".join(pieces) == text
    assert 7 <= len(pieces[0].split()) <= 11
    assert pieces[1].split()[0] not in ("से", "को", "का", "की", "के", "में", "पर", "तक", "ने", "है")


def test_hindi_conjunction_among_the_first_eight_words_ends_the_piece():
    text = "आपकी बुकिंग हो गई और डॉक्टर कल सुबह दस बजे मिलेंगे।"
    assert first(text) == "आपकी बुकिंग हो गई"


def test_hinglish_first_clause():
    pieces = run("Ji haan, aapki booking ho gayi hai, kal subah das baje aaiye.")
    assert pieces[0] == "Ji haan, aapki booking ho gayi hai,"
    pieces = run("Clinic subah nau baje se raat aath baje tak khulta hai aur Sunday ko sirf emergency ke liye khula rehta hai.")
    assert pieces[0] == "Clinic subah nau baje se raat aath baje tak khulta hai"
    assert pieces[1].startswith("aur Sunday")


# ------------------------------------------------------------------------------------------------ invariants

TEXTS = [
    "Thank you for calling, how can I help you today?",
    "The clinic is open from nine to five on weekdays, and the pharmacy inside is open until seven thirty.",
    "क्लिनिक सोमवार से शनिवार सुबह नौ बजे से रात आठ बजे तक खुलता है और रविवार को सिर्फ आपातकालीन मरीजों के लिए खुला रहता है।",
    "Clinic subah nau baje se raat aath baje tak khulta hai aur Sunday ko sirf emergency ke liye khula rehta hai.",
    "The surgery costs 1,500 rupees at 10:30 including 4.5 percent tax, see https://bonclay.in/pay for details. Dr. Rao will call you.",
    "Yes, we are open on Sunday. Please come at ten, and bring your id card with you.",
]


@pytest.mark.parametrize("text", TEXTS)
def test_the_pieces_do_not_depend_on_how_the_text_was_chopped_up(text):
    whole = run(text, step=len(text))
    for step in (1, 2, 3, 5, 7, 11):
        assert run(text, step=step) == whole, step


@pytest.mark.parametrize("text", TEXTS)
def test_no_text_is_lost_or_invented_and_pieces_stay_under_the_cap(text):
    pieces = run(text)
    assert " ".join(pieces).split() == text.split()
    assert all(len(p) <= TURN_MAX_CHARS * 1.3 for p in pieces)


def test_only_the_first_piece_is_a_clause_later_ones_are_sentences():
    text = "Thank you for calling, the clinic is open on weekdays, and on Saturday mornings too. Is there anything else I can do for you, sir?"
    pieces = run(text)
    assert pieces[0] == "Thank you for calling,"
    assert pieces[1:] == ["the clinic is open on weekdays, and on Saturday mornings too.",
                          "Is there anything else I can do for you, sir?"]


def test_a_regenerated_reply_has_no_first_clause():
    text = "Thank you for calling, how can I help you today?"
    assert run(text, first_phrase=False) == [text]


def test_the_switch_off_restores_the_older_rule(monkeypatch):
    text = "Thanks a lot for calling, sir."
    monkeypatch.setenv("TURN_FIRST_CLAUSE", "off")
    assert sentences_mod.first_clause_enabled() is False
    assert run(text) == [text]                                       # the older rule needs 8 words before it cuts
    monkeypatch.setenv("TURN_FIRST_CLAUSE", "on")
    assert sentences_mod.first_clause_enabled() is True
    assert run(text) == ["Thanks a lot for calling,", "sir."]
    monkeypatch.delenv("TURN_FIRST_CLAUSE")
    assert sentences_mod.first_clause_enabled() is True               # the default


def test_the_plain_splitter_is_untouched():
    text = "Thank you for calling, how can I help you today?"
    sp = sentences_mod.SentenceSplitter()
    assert sp.feed(text) + sp.flush() == [text]

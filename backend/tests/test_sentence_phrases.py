"""Spoken-turn sentence splitting (services/sentences.py turn_splitter) under the older first-phrase rule
(TURN_FIRST_CLAUSE=off: a cut at a comma or conjunction only once 8 words have arrived), a 110-character cap on
every piece, and the plain splitter unchanged. The first-clause rule that is on by default is in test_first_clause.py."""
import functools

import pytest

from services.sentences import SentenceSplitter, TURN_MAX_CHARS
from services.sentences import turn_splitter as _turn_splitter

turn_splitter = functools.partial(_turn_splitter, first_clause=False)


def run(text, splitter, step=5):
    out = []
    for i in range(0, len(text), step):
        out += splitter.feed(text[i:i + step])
    return out + splitter.flush()


LONG = ("The clinic is open from nine to five on weekdays, and the pharmacy inside the clinic is open until seven "
        "thirty in the evening, so you can collect medicines after your visit.")


def test_the_first_phrase_is_cut_at_the_first_comma_once_eight_words_have_arrived():
    pieces = run(LONG, turn_splitter())
    assert pieces[0] == "The clinic is open from nine to five on weekdays,"
    assert " ".join(pieces) == LONG


def test_the_first_piece_is_emitted_as_soon_as_the_text_allows_it():
    sp = turn_splitter()
    got = []
    for word in LONG.split():
        got += sp.feed(word + " ")
        if got:
            break
    assert got == ["The clinic is open from nine to five on weekdays,"]      # before the sentence has ended


def test_a_conjunction_ends_the_first_phrase_when_there_is_no_comma():
    text = "The dentist consultation costs four hundred rupees and the paediatrician charges three hundred."
    assert run(text, turn_splitter())[0] == "The dentist consultation costs four hundred rupees"


def test_hindi_first_phrase_splits_before_a_conjunction():
    text = "क्लिनिक सोमवार से शनिवार सुबह नौ बजे से रात आठ बजे तक खुलता है और रविवार को सिर्फ आपातकालीन मरीजों के लिए खुला रहता है।"
    pieces = run(text, turn_splitter())
    assert pieces[0].endswith("खुलता है") and pieces[1].startswith("और रविवार")


def test_hinglish_conjunction_splits_too():
    text = "Clinic subah nau baje se raat aath baje tak khulta hai aur Sunday ko sirf emergency ke liye khula rehta hai."
    pieces = run(text, turn_splitter())
    assert pieces[0] == "Clinic subah nau baje se raat aath baje tak khulta hai" and pieces[1].startswith("aur Sunday")


def test_short_sentences_are_not_split():
    assert run("Sure, I can help with that.", turn_splitter()) == ["Sure, I can help with that."]
    assert run("Yes, we are open on Sunday. Please come at ten.", turn_splitter()) == [
        "Yes, we are open on Sunday.", "Please come at ten."]


def test_a_comma_too_early_is_not_a_first_phrase():
    # "Sure," has one word before the comma: the verb and its object must not be parted
    assert run("Sure, the clinic opens at nine in the morning every weekday.", turn_splitter()) == [
        "Sure, the clinic opens at nine in the morning every weekday."]


def test_only_the_first_piece_is_cut_short():
    pieces = run("Yes, we can see you tomorrow morning, and the doctor is free. " * 2, turn_splitter())
    assert pieces[0] == "Yes, we can see you tomorrow morning,"
    assert pieces[1:] == ["and the doctor is free.", "Yes, we can see you tomorrow morning, and the doctor is free."]


def test_numbers_with_commas_and_decimals_are_not_boundaries():
    pieces = run("The surgery costs 1,500 rupees including the 4.5 percent tax for the whole family today.",
                 turn_splitter())
    assert pieces == ["The surgery costs 1,500 rupees including the 4.5 percent tax for the whole family today."]


@pytest.mark.parametrize("text", [
    "A " * 30 + "end.",
    "This sentence has no comma but it keeps going and going and going well past the cap of one hundred and ten characters so it needs a cut somewhere sensible.",
    LONG + " " + LONG,
])
def test_no_piece_is_longer_than_the_cap(text):
    pieces = run(text, turn_splitter())
    assert all(len(p) <= TURN_MAX_CHARS * 1.3 for p in pieces), [len(p) for p in pieces]
    assert " ".join(pieces).split() == text.split()


def test_a_finished_long_sentence_is_cut_at_clauses():
    text = ("When you arrive, please go to the reception desk on the first floor, show your booking message to the "
            "receptionist, and wait for your name to be called by the nurse.")
    pieces = run(text, turn_splitter(first_phrase=False))
    assert len(pieces) >= 2 and all(len(p) <= TURN_MAX_CHARS for p in pieces)
    assert all(p.endswith(",") or p.endswith(".") for p in pieces)


def test_no_first_phrase_for_text_that_is_already_complete():
    text = "The clinic is open from nine to five on weekdays, and the pharmacy is open until seven."   # under the cap
    assert run(text, turn_splitter(first_phrase=False)) == [text]
    assert run(text, turn_splitter())[0] == "The clinic is open from nine to five on weekdays,"


def test_the_plain_splitter_is_unchanged():
    assert run(LONG, SentenceSplitter()) == [LONG]
    assert SentenceSplitter().max_chars == 220

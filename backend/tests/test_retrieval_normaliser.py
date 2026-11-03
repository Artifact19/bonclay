"""The one Hindi / Hinglish / English normaliser (services/retrieval.py tokenizer, contract 5): Roman Hinglish spelling
variants, Devanagari <-> Roman, nukta / chandrabindu / diacritics, digits of Indian scripts, thousands separators,
₹ / Rs / rupees, phone numbers, and the cross-lingual keys for a few frequent words. No model."""
import pytest

from services import retrieval

tok = retrieval.tokenize


def same(a: str, b: str) -> bool:
    return tok(a, drop_stopwords=False) == tok(b, drop_stopwords=False)


# ------------------------------------------------------------------------------------------------ Roman Hinglish

@pytest.mark.parametrize("variants", [
    ("kab", "kb", "kabb"),
    ("kitna", "kitni", "kitne", "kitnaa", "kitana"),
    ("hai", "he", "h", "hain", "hein", "hy"),
    ("nahi", "nahin", "nai", "nhi", "nahee"),
    ("doctor", "dr", "dactor", "daktar", "docter", "doktor", "doctors"),
    ("fees", "fee", "phees", "phee"),
    ("kaise", "kese", "kaisa", "kaisi"),
    ("kaun", "kon", "koun", "konsa", "kaunsa"),
    ("chahiye", "chaiye", "chahie"),
    ("zaroor", "jaroor", "jarur"),
    ("bhaiya", "bhaiyaa"),
    ("appointment", "apointment", "appoinment"),
    ("rupees", "rupaye", "rupye", "rs", "inr", "rupee"),
])
def test_roman_hinglish_spelling_variants_share_one_key(variants):
    keys = {retrieval.normalise_token(v) for v in variants}
    assert len(keys) == 1, dict(zip(variants, (retrieval.normalise_token(v) for v in variants)))


@pytest.mark.parametrize("a,b", [
    ("fee", "time"), ("sunday", "monday"), ("doctor", "dentist"), ("kab", "kya"), ("clinic", "hospital"),
    ("fee", "free"), ("open", "closed"), ("morning", "evening"), ("main", "mein"),
])
def test_unrelated_words_stay_apart(a, b):
    assert retrieval.normalise_token(a) != retrieval.normalise_token(b)


def test_plain_english_words_are_stable():
    for w in ("pharmacy", "emergency", "parking", "insurance", "cardiology", "paediatrician", "orthopaedics"):
        key = retrieval.normalise_token(w)
        assert key and key == retrieval.normalise_token(w.upper().lower())
    assert retrieval.normalise_token("main") != ""          # "main gate" keeps its word (not a stopword)
    assert "main" in tok("main gate") or "man" in tok("main gate")


# ------------------------------------------------------------------------------------------------ Devanagari

@pytest.mark.parametrize("deva,roman", [
    ("कितना", "kitna"), ("कितनी", "kitni"), ("कितने", "kitne"), ("कब", "kab"), ("नहीं", "nahi"), ("नही", "nahin"),
    ("है", "hai"), ("हैं", "hain"), ("फीस", "fees"), ("फ़ीस", "phees"), ("डॉक्टर", "doctor"), ("डाक्टर", "dr"),
    ("क्लिनिक", "clinic"), ("हिंदी", "hindi"), ("हिन्दी", "hindi"), ("प्रकाश", "prakash"), ("भोजन", "bhojan"),
    ("समय", "samay"), ("बजे", "baje"), ("ज़रूर", "zaroor"), ("मुझे", "mujhe"), ("चाहिए", "chahiye"),
    ("अपॉइंटमेंट", "appointment"), ("इमरजेंसी", "emergency"), ("हॉस्टल", "hostel"), ("कॉलेज", "college"),
    ("रुपये", "rupees"), ("कहाँ", "kahan"), ("कौन", "kaun"), ("ज्ञान", "gyan"), ("ऋषि", "rishi"),
])
def test_devanagari_and_roman_forms_match(deva, roman):
    assert same(deva, roman), (tok(deva, drop_stopwords=False), tok(roman, drop_stopwords=False))


def test_nukta_chandrabindu_and_anusvara_variants():
    assert same("ज़रूर", "जरूर") and same("फ़ोन", "फोन") and same("हूँ", "हूं") and same("कहाँ", "कहां")
    assert same("सम्बन्ध", "संबंध")                               # half nasal vs anusvara
    assert same("ड़", "ड")


def test_zero_width_characters_do_not_split_words():
    assert tok("क्‍लिनिक") == tok("क्लिनिक")
    assert tok("doc­tor") == tok("doctor")


def test_latin_diacritics_are_stripped_but_devanagari_signs_kept():
    assert same("Pāṇḍey", "pandey") and same("café", "cafe")
    assert not same("कितना", "कत")                                # vowel signs are not "diacritics"


# ------------------------------------------------------------------------------------------------ cross-lingual keys

@pytest.mark.parametrize("hindi,english", [
    ("रविवार", "Sunday"), ("ravivar", "sunday"), ("itwar", "Sunday"), ("सोमवार", "Monday"), ("मंगलवार", "Tuesday"),
    ("शनिवार", "Saturday"), ("शुल्क", "fee"), ("अस्पताल", "hospital"), ("aspatal", "hospital"), ("दवाई", "medicine"),
    ("बच्चों", "children"), ("bachche", "child"), ("खुलता", "open"), ("khulega", "open"), ("बंद", "closed"),
    ("घंटे", "hours"), ("समय", "time"), ("timings", "time"), ("नंबर", "number"), ("फोन", "phone"), ("आज", "today"),
    ("सुबह", "morning"), ("शाम", "evening"), ("परीक्षा", "exam"), ("छुट्टी", "holiday"), ("मिलेगा", "available"),
])
def test_frequent_hindi_words_find_the_english_document_word(hindi, english):
    assert same(hindi, english), (tok(hindi, drop_stopwords=False), tok(english, drop_stopwords=False))


def test_the_variant_table_has_no_clashes():
    seen = {}
    for canonical, variants in retrieval._VARIANTS.items():
        canon = retrieval._raw_key(canonical.lower())
        for v in (canonical, *variants):
            key = retrieval._raw_key(v.lower())
            assert seen.setdefault(key, canon) == canon, f"{v!r} maps to two words"
    table = retrieval._variant_table()
    assert all(table.get(c) is None for c in {retrieval._raw_key(c.lower()) for c in retrieval._VARIANTS})


# ------------------------------------------------------------------------------------------------ numbers and money

def test_indian_script_digits_become_ascii():
    assert tok("४००") == ["400"] and tok("৪০০") == ["400"] and tok("૪૦૦") == ["400"] and tok("௪௦௦") == ["400"]
    assert same("फीस ४५०", "fees 450")


def test_thousands_separators_indian_and_western():
    assert tok("45,000") == tok("45000") == ["45000"]
    assert tok("₹1,20,000") == tok("Rs 120000")
    assert tok("1,2,3") == ["1", "2", "3"]                         # a list, not a number


@pytest.mark.parametrize("text", ["₹300", "Rs 300", "Rs.300", "rs300", "300 rupees", "300 रुपये", "३०० रुपए", "INR 300"])
def test_rupee_forms_are_one_word(text):
    assert sorted(tok(text)) == sorted(tok("₹ 300")), tok(text)


def test_phone_numbers_in_groups_also_match_joined():
    assert "02055501299" in tok("Ambulance: 020-5550-1299") and "02055501299" in tok("020 5550 1299")
    assert "02055501299" in tok("02055501299")
    assert "98200" in tok("98200 12345") and "9820012345" in tok("98200 12345")
    assert tok("10 to 1") == ["10", "1"]                           # short groups are not glued


def test_letters_and_digits_split():
    assert tok("10am") == tok("10 am") and tok("Room12") == tok("room 12")


# ------------------------------------------------------------------------------------------------ whole questions

@pytest.mark.parametrize("a,b", [
    ("Sunday ko clinic kab khulta hai?", "रविवार को क्लिनिक कब खुलता है?"),
    ("dr sahab ki phees kitni hai", "डॉक्टर साहब की फीस कितनी है"),
    ("bachchon ke doctor kab baithte hain", "बच्चों के डॉक्टर कब बैठते हैं"),
    ("dentist ki fees kitni hai", "Dentist ki fee kitne hai"),
])
def test_same_question_in_two_scripts_gives_the_same_tokens(a, b):
    assert tok(a) == tok(b)


def test_hinglish_question_matches_english_passage_with_bm25():
    texts = ["Clinic hours: Monday to Saturday 9 AM to 8 PM. Sunday: 10:00 AM to 1:00 PM.",
             "Consultation fees: General physician ₹400. Dentist: ₹300 consultation.",
             "Emergency: call 108 or go to Lakeside General Hospital (24 hours)."]
    assert retrieval.bm25_ranking("ravivar ko clinic kitne baje khulta hai", texts)[0] == 0
    assert retrieval.bm25_ranking("दाँत के डॉक्टर की फीस कितनी है", texts)[0] == 1
    assert retrieval.bm25_ranking("aspatal emergency number", texts)[0] == 2
    assert retrieval.bm25_ranking("300 rupaye", texts)[0] == 1


def test_stopwords_never_leave_an_empty_query_and_content_tokens_do():
    assert tok("kya hai") != [] and retrieval.content_tokens("kya hai") == []
    assert retrieval.content_tokens("डॉक्टर कब बैठते हैं") == ["doctor", tok("बैठते")[0]]


def test_normalise_text_for_other_modules():
    assert retrieval.normalise_text("Dr. Sharma ki fees ₹४००") == retrieval.normalise_text("डॉक्टर sharma की फीस Rs 400")
    assert retrieval.normalise_text("") == ""


def test_tokenizer_without_the_transliteration_package(monkeypatch):
    """indic_transliteration missing: Devanagari words stay Devanagari (still normalised), nothing raises."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("indic_transliteration"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    retrieval._transliterate_devanagari.cache_clear()
    retrieval.normalise_token.cache_clear()
    retrieval._tokenize_cached.cache_clear()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert tok("ज़रूर") == tok("जरूर") and tok("कितना") != []
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        retrieval._transliterate_devanagari.cache_clear()
        retrieval.normalise_token.cache_clear()
        retrieval._tokenize_cached.cache_clear()
        retrieval._variant_table.cache_clear()
        retrieval._stopwords.cache_clear()

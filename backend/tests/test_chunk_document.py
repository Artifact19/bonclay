"""Structure-aware chunking: headings, FAQ pairs, table rows, Hindi sentence ends, long prose, headers."""
from pathlib import Path

import pytest

from services import chunking
from services.chunking import chunk_document, embedding_text, guess_title, split_sentences

SEP = chunking.HEADER_SEP
DEMO_PDF = Path(__file__).resolve().parents[2] / "docs" / "demo" / "Sahyadri_Care_Hospital_Pune_FAQ.pdf"


def by_header(passages, part):
    return [p for p in passages if part in p["header"]]


# ----------------------------------------------------------------------------------------------- headings

MD = """# Clinic Guide

Welcome to our clinic. We are glad you came.

## Timings

We open at 9 a.m. to 5 p.m. Dr. Rao sees patients on Monday. Call 020 5550 1200.

### Weekends

Closed on Sunday.

## Fees

Consultation is Rs. 400. Follow-up is free.
"""


def test_markdown_headings_build_the_header_path():
    ps = chunk_document(MD, "Clinic Guide")
    assert all(p["kind"] == "passage" and set(p) == {"content", "header", "kind"} for p in ps)
    got = {p["header"]: p["content"] for p in ps}
    assert got["Clinic Guide"].startswith("Welcome to our clinic.")
    assert got[f"Clinic Guide{SEP}Timings"].startswith("We open at 9 a.m. to 5 p.m. Dr. Rao sees patients")
    assert got[f"Clinic Guide{SEP}Timings{SEP}Weekends"] == "Closed on Sunday."
    assert got[f"Clinic Guide{SEP}Fees"] == "Consultation is Rs. 400. Follow-up is free."
    # the heading text is in the header, not repeated in the content
    assert not any(p["content"].startswith("Timings") for p in ps)


def test_title_line_is_not_a_section_and_header_has_no_duplicate_parts():
    text = "Sahyadri Care Hospital, Pune\n\nAbout the hospital\n\nA 120-bed hospital in Baner.\n"
    ps = chunk_document(text, "Sahyadri Care Hospital, Pune")
    assert [p["header"] for p in ps] == [f"Sahyadri Care Hospital, Pune{SEP}About the hospital"]
    assert chunk_document("# Fees\nRs 400 for a visit today.", "Fees")[0]["header"] == "Fees"


def test_numbered_all_caps_and_bold_headings():
    text = ("1. ELIGIBILITY\nStudents must be eighteen years old.\n\n"
            "2.1 Documents needed\nBring your Aadhaar card and two photos.\n\n"
            "**Payment**\nPay by UPI or cash at the counter.\n\n"
            "FEES AND CHARGES\nThe annual fee is Rs 45,000.\n")
    ps = chunk_document(text, "Prospectus")
    assert [p["header"] for p in ps] == [
        f"Prospectus{SEP}ELIGIBILITY",
        f"Prospectus{SEP}ELIGIBILITY{SEP}Documents needed",          # 2.1 is a sub-section of 1
        f"Prospectus{SEP}ELIGIBILITY{SEP}Documents needed{SEP}Payment",   # bold lines rank below numbered ones
        f"Prospectus{SEP}FEES AND CHARGES"]                          # an ALL-CAPS line starts a new top-level section
    assert ps[0]["content"] == "Students must be eighteen years old."


def test_a_numbered_list_is_not_turned_into_headings():
    text = "How to apply\n\n1. Bring your ID card\n2. Fill the form at the desk\n3. Pay the fee at the counter\n"
    ps = chunk_document(text, "Guide")
    assert len(ps) == 1 and "1. Bring your ID card" in ps[0]["content"] and "3. Pay the fee" in ps[0]["content"]


def test_headers_are_trimmed_and_embedding_text_is_header_then_content():
    ps = chunk_document("# " + " ".join(["Long"] * 25) + "\nSome text is here for the passage to have words in it.", "T")
    assert all(len(part) <= chunking.MAX_HEADER_PART_CHARS + 3 for p in ps for part in p["header"].split(SEP))
    assert embedding_text("Doc › Fees", "Rs 400") == "Doc › Fees\nRs 400"
    assert embedding_text("", "Rs 400") == "Rs 400" and embedding_text(None, "x") == "x"


def test_guess_title():
    assert guess_title("Sahyadri Care Hospital, Pune\nPatient information.", "x.pdf") == "Sahyadri Care Hospital, Pune"
    assert guess_title("# Clinic Guide\ntext", "x.md") == "Clinic Guide"
    long_first = "This document explains how every part of the hospital works and what we offer to patients."
    assert guess_title(long_first, "Sahyadri_Care_FAQ.pdf") == "Sahyadri Care FAQ"
    assert guess_title("", "my-file_name.txt") == "my file name"


# ---------------------------------------------------------------------------------------------------- FAQ

def test_faq_pairs_stay_together_in_every_notation():
    text = ("## FAQ\n\n"
            "### How do I book an appointment?\nCall the desk or use the website. Walk-ins are welcome.\n\n"
            "Q: Is parking free?\nA: Yes, in the basement.\n\n"
            "Q: Where is the pharmacy? A: On the ground floor.\n\n"
            "Do you accept insurance?\nYes, cashless with Star Health and HDFC Ergo.\n\n"
            "**Can I cancel?**\nYes, up to two hours before the slot.\n")
    ps = chunk_document(text, "Hospital")
    got = {p["header"].split(SEP)[-1]: p["content"] for p in ps}
    assert got["How do I book an appointment?"] == ("Q: How do I book an appointment?\n"
                                                    "A: Call the desk or use the website. Walk-ins are welcome.")
    assert got["Is parking free?"] == "Q: Is parking free?\nA: Yes, in the basement."
    assert got["Where is the pharmacy?"] == "Q: Where is the pharmacy?\nA: On the ground floor."
    assert got["Do you accept insurance?"] == "Q: Do you accept insurance?\nA: Yes, cashless with Star Health and HDFC Ergo."
    assert got["Can I cancel?"].endswith("A: Yes, up to two hours before the slot.")
    assert all(p["header"].startswith(f"Hospital{SEP}FAQ{SEP}") for p in ps)
    assert len(ps) == 5


def test_hindi_faq_and_long_answer_keeps_the_question_in_every_part():
    hi = "क्या पार्किंग मुफ़्त है?\nजी हाँ, तहखाने में पार्किंग मुफ़्त है।\n\nप्रश्न: रविवार को क्या अस्पताल खुला है?\nउत्तर: केवल आपातकालीन विभाग खुला है।"
    ps = chunk_document(hi, "अस्पताल")
    assert [p["content"].split("\n")[0] for p in ps] == ["Q: क्या पार्किंग मुफ़्त है?", "Q: रविवार को क्या अस्पताल खुला है?"]
    assert ps[1]["content"].endswith("A: केवल आपातकालीन विभाग खुला है।")
    answer = " ".join(f"Step {i} of the refund process takes {i} days to complete." for i in range(1, 41))
    ps = chunk_document(f"How does a refund work?\n{answer}\n", "Policy")
    assert len(ps) > 1
    for p in ps:
        assert p["content"].startswith("Q: How does a refund work?\nA: ")
        assert p["header"].endswith("How does a refund work?")
        assert p["content"].rstrip().endswith(".")


# ---------------------------------------------------------------------------------------------------- tables

def test_markdown_table_one_row_per_passage_with_the_header_repeated():
    text = ("## OPD\n\n| Department | Days | Time |\n|---|---|---|\n| Cardiology | Mon–Sat | 9–1 |\n"
            "| Eye | Mon–Fri | 2–6 |\n| Skin | Mon–Wed | 10–12 |\n")
    ps = chunk_document(text, "Hospital")
    assert [p["content"] for p in ps] == ["Department | Days | Time\nCardiology | Mon–Sat | 9–1",
                                          "Department | Days | Time\nEye | Mon–Fri | 2–6",
                                          "Department | Days | Time\nSkin | Mon–Wed | 10–12"]
    assert {p["header"] for p in ps} == {f"Hospital{SEP}OPD"}


LAYOUT_TABLE = """Outpatient (OPD) timings

Department                                Days                                  Timings

General Medicine                          Monday to Saturday                    9:00 am to 5:00 pm

Orthopaedics                              Monday, Wednesday, Friday             11:00 am to 3:00 pm

Cardiology                                Tuesday, Thursday, Saturday           9:00 am to 1:00 pm

The OPD is closed on Sundays.

Contact

Item                           Details

Emergency and                  020 5550 1299 (24 hours, every day)
ambulance

Nearest landmarks              Opposite Baner Gaon bus stop. About 6 km from Pune University
                               (Savitribai Phule Pune University).

Parking                        Free for patients in the basement.
"""


def test_layout_table_rows_with_wrapped_cells():
    ps = chunk_document(LAYOUT_TABLE, "Hospital")
    opd = by_header(ps, "Outpatient (OPD) timings")
    assert [p["content"] for p in opd[:3]] == [
        "Department | Days | Timings\nGeneral Medicine | Monday to Saturday | 9:00 am to 5:00 pm",
        "Department | Days | Timings\nOrthopaedics | Monday, Wednesday, Friday | 11:00 am to 3:00 pm",
        "Department | Days | Timings\nCardiology | Tuesday, Thursday, Saturday | 9:00 am to 1:00 pm"]
    assert opd[3]["content"] == "The OPD is closed on Sundays."
    contact = {p["content"] for p in by_header(ps, "Contact")}
    assert "Item | Details\nEmergency and ambulance | 020 5550 1299 (24 hours, every day)" in contact
    assert ("Item | Details\nNearest landmarks | Opposite Baner Gaon bus stop. About 6 km from Pune University "
            "(Savitribai Phule Pune University).") in contact


def test_double_spaces_in_prose_do_not_make_a_table():
    text = "Hello.  How are you today.  We hope you are well and happy.\nPlease sit down.  The doctor will come soon.\n"
    ps = chunk_document(text, "Note")
    assert len(ps) == 1 and "|" not in ps[0]["content"]


# ------------------------------------------------------------------------------------------ sentences, Hindi

def test_sentence_boundaries():
    assert split_sentences("Open 9 a.m. to 5 p.m. Dr. A. K. Rao is in. Cost Rs. 400. Really?! Yes... fine.") == [
        "Open 9 a.m. to 5 p.m.", "Dr. A. K. Rao is in.", "Cost Rs. 400.", "Really?!", "Yes... fine."]
    assert split_sentences("हिंदी वाक्य। दूसरा वाक्य॥तीसरा वाक्य।चौथा") == [
        "हिंदी वाक्य।", "दूसरा वाक्य॥", "तीसरा वाक्य।", "चौथा"]
    assert split_sentences("Is it open? Yes! Come in.") == ["Is it open?", "Yes!", "Come in."]
    assert split_sentences("Fee is 9.50 rupees only.") == ["Fee is 9.50 rupees only."]


def test_hindi_danda_text_is_cut_only_at_sentence_ends():
    sentence = "ओपीडी सोमवार से शनिवार तक सुबह नौ बजे से शाम पाँच बजे तक खुली रहती है।"
    text = "# समय\n" + " ".join([sentence] * 30)
    ps = chunk_document(text, "अस्पताल")
    assert len(ps) > 1
    for p in ps:
        assert len(p["content"].split()) <= 120
        assert p["content"].rstrip().endswith("।")
        assert p["header"] == f"अस्पताल{SEP}समय"


# ---------------------------------------------------------------------------------------------- long prose

def test_long_prose_is_120_words_with_a_whole_sentence_overlap_and_never_cut_mid_sentence():
    sentences = [f"Sentence number {i} says something useful about item {i}." for i in range(80)]
    ps = chunk_document("# Facts\n" + " ".join(sentences), "Doc")
    assert len(ps) > 4
    for p in ps:
        assert len(p["content"].split()) <= 120
        assert p["content"].endswith(".") and p["content"].startswith("Sentence number")
        assert all(part in sentences for part in split_sentences(p["content"]))
    # consecutive passages share trailing/leading sentences worth at most 20 words
    for a, b in zip(ps, ps[1:]):
        sa, sb = split_sentences(a["content"]), split_sentences(b["content"])
        shared = [s for s in sb if s in sa]
        assert shared and sum(len(s.split()) for s in shared) <= 20
        assert sb[: len(shared)] == shared and sa[-len(shared):] == shared
    # nothing is lost
    seen = {s for p in ps for s in split_sentences(p["content"])}
    assert seen == set(sentences)


def test_prose_never_crosses_a_heading_and_wrapped_lines_are_joined():
    text = ("## A\n\nThe first line of a wrapped paragraph that goes on and on and on until it\n"
            "reaches the end of this sentence. A second sentence.\n\n## B\n\nOther text lives here.\n")
    ps = chunk_document(text, "T")
    assert [p["header"] for p in ps] == [f"T{SEP}A", f"T{SEP}B"]
    assert "until it reaches the end of this sentence." in ps[0]["content"]
    assert "Other" not in ps[0]["content"]


def test_a_huge_unpunctuated_run_is_cut_into_word_windows():
    ps = chunk_document(" ".join(f"w{i}" for i in range(500)), "Doc")
    assert [len(p["content"].split()) for p in ps] == [120, 120, 120, 120, 100]


def test_a_very_long_sentence_is_cut_at_commas_not_words():
    clause = "the fee covers item number {} and its handling"
    sentence = ", ".join(clause.format(i) for i in range(40)) + "."
    ps = chunk_document(sentence, "Doc")
    assert len(ps) > 1
    assert all(len(p["content"].split()) <= 120 for p in ps)
    assert all(p["content"].split(",")[0].startswith("the fee covers") or p["content"].startswith("the fee") for p in ps)


def test_empty_and_tiny_input():
    assert chunk_document("", "T") == [] and chunk_document("  \n \n", "T") == []
    assert chunk_document("One line.", "T") == [{"content": "One line.", "header": "T", "kind": "passage"}]


def test_chunk_text_api_is_unchanged():
    assert chunking.chunk_text("# Fees\nRs 400.\nFree follow-up.") == ["Fees\nRs 400.\nFree follow-up."]


# ----------------------------------------------------------------------------------- the demo hospital PDF

@pytest.mark.skipif(not DEMO_PDF.exists(), reason="demo PDF not in the checkout")
def test_demo_pdf_tables_become_one_row_per_passage():
    pypdf = pytest.importorskip("pypdf")
    text = chunking.pdf_text_from_reader(pypdf.PdfReader(str(DEMO_PDF)))
    title = guess_title(text, DEMO_PDF.name)
    assert title == "Sahyadri Care Hospital, Pune"
    ps = chunk_document(text, title)
    cardiology = [p for p in ps if p["content"].startswith("Department | Days | Timings\nCardiology")]
    assert len(cardiology) == 1
    assert cardiology[0]["content"].endswith("Cardiology | Tuesday, Thursday, Saturday | 9:00 am to 1:00 pm")
    assert cardiology[0]["header"] == f"Sahyadri Care Hospital, Pune{SEP}Outpatient (OPD) timings"
    fee = [p for p in ps if "General Medicine consultation" in p["content"]]
    assert fee and fee[0]["content"] == "Service | Fee\nGeneral Medicine consultation | Rs 400"
    assert any("Emergency and ambulance | 020 5550 1299" in p["content"] for p in ps)
    assert any(p["content"].startswith("Package or test | Price | Notes\nMRI and CT scan | Rs 3,500 to Rs 7,000")
               for p in ps)
    # every number of the source survives
    joined = " ".join(p["content"] for p in ps)
    for needle in ("020 5550 1200", "Rs 9,000", "150 cars", "MJPJAY", "Maharashtra Day"):
        assert needle in joined

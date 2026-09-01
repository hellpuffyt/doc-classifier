from __future__ import annotations

from doc_classifier.features import extract, tokenize
from doc_classifier.types import Document


class TestTokenize:
    def test_lowercases_and_splits_on_non_word_characters(self) -> None:
        assert tokenize("Bill To: ACME Corp.") == ["bill", "to", "acme", "corp"]

    def test_keeps_apostrophes_within_words(self) -> None:
        assert tokenize("customer's invoice") == ["customer's", "invoice"]

    def test_digits_alone_are_not_tokens(self) -> None:
        assert tokenize("total 1204.00 due") == ["total", "due"]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("") == []


class TestExtract:
    def test_token_counts(self) -> None:
        features = extract(Document(text="invoice invoice total"))
        assert features.tokens == {"invoice": 2, "total": 1}
        assert features.token_total() == 3

    def test_digit_ratio(self) -> None:
        features = extract(Document(text="ab12"))
        assert features.digit_ratio == 0.5

    def test_table_line_ratio_detects_multi_space_columns(self) -> None:
        text = "Item          Qty    Price\nWidget           2      9.99\nplain text line"
        features = extract(Document(text=text))
        assert features.table_line_ratio == 2 / 3

    def test_header_line_ratio_detects_all_caps_lines(self) -> None:
        text = "INVOICE\nBill to: someone\nTOTAL DUE"
        features = extract(Document(text=text))
        assert features.header_line_ratio == 2 / 3

    def test_header_detection_ignores_lines_with_no_letters(self) -> None:
        text = "1234\nsome text"
        features = extract(Document(text=text))
        assert features.header_line_ratio == 0.0

    def test_avg_line_length_in_tokens(self) -> None:
        features = extract(Document(text="a b c\nd e"))
        assert features.avg_line_length == 2.5

    def test_currency_detection_symbol(self) -> None:
        assert extract(Document(text="Total: $5.00")).has_currency

    def test_currency_detection_code(self) -> None:
        assert extract(Document(text="Amount: 500 USD")).has_currency

    def test_currency_detection_decimal_amount(self) -> None:
        assert extract(Document(text="pay 1204.00 now")).has_currency

    def test_no_currency_when_none_present(self) -> None:
        assert not extract(Document(text="no money words here at all")).has_currency

    def test_single_line_document_does_not_divide_by_zero(self) -> None:
        features = extract(Document(text="one line only"))
        assert features.table_line_ratio == 0.0
        assert features.avg_line_length == 3.0

    def test_empty_document_does_not_crash(self) -> None:
        features = extract(Document(text=""))
        assert features.digit_ratio == 0.0
        assert features.avg_line_length == 0.0

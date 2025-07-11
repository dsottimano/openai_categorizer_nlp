import os, pytest
from openai_categorizer_nlp import parse_one    

pytestmark = pytest.mark.integration  # custom marker

def test_parse_one_real_api():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Set OPENAI_API_KEY to run integration tests")

    cats = ["politics", "international_relations"]
    out = parse_one("biden afghanistan withdrawal", categories=cats)

    # simple sanity checks to keep token usage low
    ent_names = [e.full_name.lower() for e in out.entities]
    assert "joe biden" in ent_names
    assert "politics" in out.categories
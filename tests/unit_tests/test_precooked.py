from NERDA.precooked import DA_ELECTRA_DA

def test_load_precooked():
    """Test that precooked model can be (down)loaded, instantiated and works end-to-end"""
    m = DA_ELECTRA_DA()
    m.download_network()
    m.load_network()
    m.predict_text("Jens Hansen har en bondegård. Det har han!")

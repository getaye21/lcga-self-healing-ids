import numpy as np

def test_lcga_build():
    from src.models.lcga_model import build_lcga
    model = build_lcga(n_features=73, n_classes=15)
    assert model.count_params() < 200_000
    assert model.count_params() > 10_000

def test_lcga_forward():
    from src.models.lcga_model import build_lcga
    model = build_lcga(n_features=73, n_classes=15)
    X = np.random.randn(4, 73, 1).astype(np.float32)
    out = model(X, training=False)
    assert out.shape == (4, 15)

def test_knowledge_base(tmp_path):
    import yaml
    intents = {
        "intents": {"I1": {"name": "HTTP Latency", "metric": "http_latency_ms", "operator": "lt", "threshold": 200}},
        "attack_intent_map": {"DoS Hulk": ["I1"], "BENIGN": []},
        "healing_actions": {"BLOCK_IP": {"description": "Block IP", "severity": "medium", "t_verify_override": None}},
        "default_action_map": {"DoS Hulk": "BLOCK_IP", "BENIGN": None},
    }
    p = tmp_path / "intents.yaml"
    p.write_text(yaml.dump(intents))
    from src.mape_k.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(str(p), str(tmp_path / "state.json"))
    assert kb.get_violated_intents("DoS Hulk") == ["I1"]
    assert kb.select_action("DoS Hulk") == "BLOCK_IP"

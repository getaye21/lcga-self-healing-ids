"""MAPE-K Knowledge Base — loads intents.yaml, tracks action history."""
import yaml, json, os

class KnowledgeBase:
    def __init__(self, intents_path, state_path):
        with open(intents_path) as f:
            data = yaml.safe_load(f)
        self.intents = data["intents"]
        self.attack_map = data["attack_intent_map"]
        self.healing_actions = data["healing_actions"]
        self.default_actions = data["default_action_map"]
        self.state_path = state_path
        self.history = {}
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.history = json.load(f)

    def save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_violated_intents(self, attack_class):
        return self.attack_map.get(attack_class, [])

    def select_action(self, attack_class):
        default = self.default_actions.get(attack_class)
        if not default: return None
        best, best_rate = default, -1.0
        for action in self.healing_actions:
            if attack_class in self.history and action in self.history[attack_class]:
                succ, fail = self.history[attack_class][action]
                total = succ + fail
                if total > 0 and succ / total > best_rate:
                    best_rate = succ / total
                    best = action
        return best

    def record(self, attack_class, action, success):
        if attack_class not in self.history: self.history[attack_class] = {}
        if action not in self.history[attack_class]: self.history[attack_class][action] = [0, 0]
        if success: self.history[attack_class][action][0] += 1
        else: self.history[attack_class][action][1] += 1
        self.save_state()

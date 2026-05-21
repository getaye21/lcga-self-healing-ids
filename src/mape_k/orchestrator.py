"""MAPE-K Closed-Loop Self-Healing Orchestrator."""
import threading, time, logging
from .knowledge_base import KnowledgeBase

class MAPEOrchestrator:
    def __init__(self, kb_path, state_path, analyzer, monitor):
        self.kb = KnowledgeBase(kb_path, state_path)
        self.analyzer = analyzer
        self.monitor = monitor
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            telemetry = self.monitor.capture()
            anomaly, attack_class, shap_vals = self.analyzer.analyze(telemetry)
            if anomaly:
                intents = self.kb.get_violated_intents(attack_class)
                action = self.kb.select_action(attack_class)
                if action:
                    self._execute(action)
                    time.sleep(60)
                    restored = True  # simplified
                    self.kb.record(attack_class, action, restored)
            time.sleep(5)

    def _execute(self, action):
        logging.info(f"Executing {action}")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

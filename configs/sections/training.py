from dataclasses import dataclass

from configs.config_tracker import TrackedConfigMixin


@dataclass
class TrainingConfig(TrackedConfigMixin):
    deterministic: bool = False
    use_compile: bool = False
    inference_mode: bool = False
    seed: int = 42
    max_epochs: int = 1
    label_smoothing: float = 0.0
    mixup_alpha: float = 1.0
    precision: str = "32-true"  # Options: "32-true", "16-mixed", "bf16-mixed"

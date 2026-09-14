# Contract: Common Segmentation-Method Interface (FROZEN for Spec 2)

**Normative for**: FR-021–FR-026, FR-035, SC-008. Spec 2 deep-learning adapters implement this
exact interface; `metrics.py`, the output writer and sidecar schema are untouched by Spec 2.

```python
class BaseSegmentationMethod:
    @property
    def name(self) -> str: ...

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Grayscale uint8 [H, W] in -> binary uint8 [H, W] out, values in {0, 255}."""
```

Registry (`methods/__init__.py`):

```python
def register(name: str, factory: Callable[[dict], BaseSegmentationMethod]) -> None: ...
def create(name: str, config: dict) -> BaseSegmentationMethod: ...
def available() -> list[str]: ...
```

## Rules
1. `segment` MUST accept single-channel uint8 and return single-channel uint8 with values
   exclusively in {0, 255} (background = 0, text = 255).
2. Methods MUST NOT read the GT mask, the manifest, or the filesystem.
3. All parameters arrive via `config` at `create` time; nothing else is configurable per call.
4. Preprocessing (grayscale/denoise) happens before `segment`; alignment/metrics after.
   A method never resizes its input.
5. Adding a method = one new file under `methods/` + one `register` call. No other file changes.

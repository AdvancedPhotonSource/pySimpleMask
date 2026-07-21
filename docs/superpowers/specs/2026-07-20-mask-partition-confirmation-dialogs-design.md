# Mask/partition confirmation dialogs + bold emphasis — design

**Date:** 2026-07-20
**Branch:** mc_dev
**Scope:** `src/pysimplemask/gui/control/main_window.py` only (no core/, no `.ui` changes).
Touches `mask_evaluate_current_tab`, `mask_apply_current_tab`, `compute_partition`,
`display_metadata`, `update_parameter_to_dset`, `__init__`, and the `MaskWidget.currentChanged`
handler.

## Goal

Two safety-net confirmation dialogs plus a visual emphasis tweak, so users don't
silently create masks against stale detector geometry or compute a partition while an
evaluated-but-unapplied mask is sitting around:

1. **Mask creation** (Evaluate or Apply, any mask tab): if metadata has been edited but
   not applied to the q/φ geometry, ask whether to update it first.
2. **Partition creation**: if a mask was evaluated but never applied, ask for
   confirmation before computing the partition.
3. **Update Metadata** and **Apply** buttons render **bold whenever they're enabled**,
   reinforcing the existing gray-when-disabled state as a stronger "this needs your
   attention" signal.

## Background — existing state signals (already implemented)

Both checks piggyback on state that already exists and is already correctly
maintained (see `d0c5262`, and the `btn_update_parameters` disable-on-launch fix from
earlier this session):

- `self.btn_update_parameters.isEnabled()` — `True` means metadata was edited
  (`update_parameter_to_dset`) and the qmap has not been recomputed from it yet
  (`display_metadata` clears it back to `False` once synced). No new state needed.
- `self.btn_mask_apply.isEnabled()` — `True` means `mask_evaluate_current_tab` produced
  a candidate mask on the current `MaskWidget` tab that hasn't been applied yet.
  Switching `MaskWidget` tabs already force-disables it (`currentChanged` handler), so
  whenever it reads `True`, `mask_apply_current_tab()` is guaranteed to apply the
  correct tab's targets. No new state needed.

## Decisions (settled with the user)

1. Stale-metadata check fires on **both** Evaluate and Apply.
2. Declining the metadata update **proceeds anyway** with the mask action (stale qmap)
   — the dialog is advisory, never blocking.
3. The check applies to **all** mask tabs (Blemish/Files, Draw, Binary, Manual,
   Outlier, Parametrization), not just the two that actually read q/φ (Outlier,
   Parametrization). Simpler rule, consistent behavior across tabs.
4. Unapplied-mask dialog offers three choices: **Apply & Continue**, **Continue
   Anyway**, **Cancel** (Cancel aborts `compute_partition` before it touches any
   partition parameters/spinboxes).
5. Bold styling tracks the enabled state dynamically (not a permanent style) — set in
   code alongside the existing enable/disable logic, not in `mask.ui`.

## A — Stale-metadata check

New helper, called at the top of `mask_evaluate_current_tab()` and
`mask_apply_current_tab()`, before their `for target in ...` loops:

```python
def _maybe_prompt_metadata_update(self):
    if not self.btn_update_parameters.isEnabled():
        return  # metadata already in sync
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Metadata Changed")
    box.setText(
        "Metadata has been edited but not applied to the q/φ geometry.\n"
        "Update metadata before creating this mask?"
    )
    update_btn = box.addButton("Update Metadata", QMessageBox.AcceptRole)
    box.addButton("Continue Without Updating", QMessageBox.RejectRole)
    box.setDefaultButton(update_btn)
    box.exec()
    if box.clickedButton() is update_btn:
        self.update_parameters()
```

`update_parameters()` already recomputes the qmap, refreshes the metadata tree, and
replots — after it runs, the subsequent mask evaluate/apply in the caller uses fresh
geometry. "Continue Without Updating" is a no-op (falls through).

**Known trade-off (flagged, not blocking):** because this fires on every Evaluate/Apply
click for every tab, iterating on mask parameters (e.g. adjusting a threshold
repeatedly) while metadata sits stale will re-prompt on every click until metadata is
updated. No "don't ask again this session" — can be added later if this proves
annoying in practice.

## B — Unapplied-mask check before computing a partition

New helper, called in `compute_partition()` immediately after the existing
`is_ready()` guard, before any partition kwargs/spinboxes are touched:

```python
def _maybe_prompt_unapplied_mask(self) -> bool:
    if not self.btn_mask_apply.isEnabled():
        return True  # nothing pending
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Unapplied Mask")
    box.setText(
        "A mask was evaluated but not applied to the working mask.\n"
        "Apply it before computing the partition?"
    )
    apply_btn = box.addButton("Apply && Continue", QMessageBox.AcceptRole)
    continue_btn = box.addButton("Continue Anyway", QMessageBox.DestructiveRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(apply_btn)
    box.exec()
    clicked = box.clickedButton()
    if clicked is apply_btn:
        self.mask_apply_current_tab()
        return True
    return clicked is continue_btn
```

`compute_partition()` gains one line at its top:

```python
def compute_partition(self):
    if not self.is_ready():
        return
    if not self._maybe_prompt_unapplied_mask():
        return
    ...
```

## C — Bold-when-enabled, consolidated into shared setters

The enable+tooltip logic for these two buttons is already duplicated across several
call sites (`__init__`, `display_metadata`, `update_parameter_to_dset`,
`mask_evaluate_current_tab`, `mask_apply_current_tab`, the `MaskWidget.currentChanged`
lambda). Bolting "also toggle bold" onto each site individually would add a fourth
duplicated concern and risk drift, so this consolidates all of them into two setters:

```python
def _set_update_parameters_state(self, stale: bool):
    self.btn_update_parameters.setEnabled(stale)
    font = self.btn_update_parameters.font()
    font.setBold(stale)
    self.btn_update_parameters.setFont(font)

def _set_apply_state(self, ready: bool):
    self.btn_mask_apply.setEnabled(ready)
    self.btn_mask_apply.setToolTip(_APPLY_TIP_READY if ready else _APPLY_TIP_DISABLED)
    font = self.btn_mask_apply.font()
    font.setBold(ready)
    self.btn_mask_apply.setFont(font)
```

Every existing `self.btn_update_parameters.setEnabled(...)` and
`self.btn_mask_apply.setEnabled(...)` call site is replaced with a call to the
matching setter:

- `__init__`: `_set_update_parameters_state(False)`, `_set_apply_state(False)`.
- `display_metadata`: `_set_update_parameters_state(False)`.
- `update_parameter_to_dset`: `_set_update_parameters_state(True)`.
- `mask_evaluate_current_tab`: `_set_apply_state(True)`.
- `mask_apply_current_tab`: `_set_apply_state(False)`.
- `MaskWidget.currentChanged` handler: replace the inline lambda tuple with a call to
  `_set_apply_state(False)` (via a small bound method instead of an inline lambda, so
  the setter is reused rather than re-duplicating the tooltip+enable pair).

No `.ui`/Designer changes — bold state is dynamic and belongs in code next to the
enable logic it now travels with.

## Testing

Headless, offscreen (`QT_QPA_PLATFORM=offscreen`), added to `tests/gui/test_main_window.py`:

- `_set_update_parameters_state` / `_set_apply_state`: enabled state and
  `font().bold()` move together at each transition (launch, after load, after a
  metadata edit, after evaluate, after apply, after tab switch).
- `_maybe_prompt_metadata_update`: with metadata stale, monkeypatch `QMessageBox.exec`
  to simulate each button click and assert `update_parameters` is called (or not)
  accordingly; with metadata in sync, assert no dialog is constructed.
- `_maybe_prompt_unapplied_mask`: same monkeypatch approach for all three buttons —
  assert `mask_apply_current_tab` is called only for "Apply & Continue", and the
  return value gates `compute_partition` correctly (`Cancel` → partition kwargs/spinboxes
  untouched).

Manual verification: launch the real GUI and walk both flows visually — bold text is
easy to assert headlessly but also worth eyeballing.

## Risks & mitigations

- **Dialog fatigue** from the all-tabs, both-triggers, proceed-anyway design (see
  trade-off note in section A) — accepted for now; revisit if it proves disruptive.
- **`QMessageBox` custom-button identity checks** (`clickedButton() is apply_btn`) are
  the standard PySide6 pattern already used nowhere else in this file, but are
  well-established Qt idiom — no ambiguity risk since buttons are freshly created
  per-call.
- **Interaction with `save_mask`**: `save_mask` can call `self.compute_partition()`
  internally (when `sm.new_partition is None`) — this now also goes through the
  unapplied-mask prompt, which is correct/desired (same guarantee should hold whether
  partition is computed explicitly or implicitly via Save).

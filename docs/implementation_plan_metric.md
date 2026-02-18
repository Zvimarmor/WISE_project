# Lab Editor & Metric Implementation Plan

## 1. New Editor Class: `EasyEdit/easyeditor/editors/wise_lab_editor.py`

We will subclass `BaseEditor` to add the "First Edit Retention" metric without modifying the core library.

```python
from .editor import BaseEditor
from ..evaluate import compute_edit_quality
# ... imports ...

class WISELabEditor(BaseEditor):
    def edit_requests(self, requests, sequential_edit=False, verbose=True, test_generation=False, **kwargs):
        # ... copy logic from BaseEditor.edit_requests ...
        
        # New Logic for Metric
        track_first_edit = kwargs.get('track_first_edit', False)
        first_edit_history = []
        
        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                
                # --- METRIC INJECTION ---
                if track_first_edit and len(requests) > 0 and (i % 10 == 0 or i == len(requests)-1):
                    # Evaluate Request[0]
                    res_0 = compute_edit_quality(
                        edited_model, 
                        self.model_name, 
                        self.hparams, 
                        self.tok, 
                        requests[0], 
                        self.hparams.device, 
                        eval_metric=kwargs.get('eval_metric', 'exact match'),
                        test_generation=test_generation
                    )
                    first_edit_history.append({
                        'step': i + 1,
                        'metrics': res_0
                    })
                # ------------------------
                
                # Update model
                # ... (standard update logic) ...
        
        # Save or Return history
        # We will modify the return to include this history
        return all_metrics, edited_model, weights_copy, first_edit_history
```

## 2. New Runner Script: `EasyEdit/examples/run_wise_lab_editing.py`

A cloned version of `run_wise_editing.py`:
1.  **Import**: `from easyeditor.editors.wise_lab_editor import WISELabEditor`
2.  **ArgParse**: Add `--track_first_edit` flag.
3.  **Instantiation**: Use `WISELabEditor` instead of `BaseEditor`.
4.  **Handling Return**: Unpack the 4th return value (`first_edit_history`) and save it to `outputs/.../first_edit_metrics.json`.

## 3. Update SLURM Script

1.  update `scripts/run_wise_lab.sh` to run `EasyEdit/examples/run_wise_lab_editing.py`.
2.  Add `--track_first_edit` to the arguments.
3.  **Environment**: Update `source ...` line with the path found (pending `find` command).

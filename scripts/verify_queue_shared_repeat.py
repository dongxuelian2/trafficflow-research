"""Verify full public-frame sampling and independent shared-model retraining."""
import hashlib
import json
from pathlib import Path
from trafficflow.queue import QueueModel
from trafficflow.runtime import sha256_file


if __name__=='__main__':
    output=Path('reports/research/C24_C29_shared_training_repeat.json')
    if output.exists():raise FileExistsError('Never overwrite repetition evidence')
    pairs=[('C24_D12_I5_N_aligned_onset','C27_D12_I5_N_onset_specialist'),
        ('C25_D7_I10_W_aligned_onset','C28_D7_I10_W_onset_specialist'),
        ('C26_D12_I5_N_winter_aligned_onset','C29_D12_I5_N_winter_onset_specialist')]
    rows=[]
    for first,second in pairs:
        a=json.loads(Path('reports/research',first+'.json').read_text())
        b=json.loads(Path('reports/research',second+'.json').read_text())
        if a['dates']!=b['dates']:raise ValueError('Training calendars differ')
        for part in ('fit','selection'):
            if a['samplers'][part]['windows']!=b['samplers'][part]['windows']:
                raise ValueError('Repeated event sampling changed')
        old_path=Path('artifacts/research',first,'at_T_600_63.pkl')
        new_path=Path('artifacts/research',second,'shared_threshold.pkl')
        old,new=map(QueueModel.load,(old_path,new_path))
        old_text=old.model.booster_.model_to_string()
        new_text=new.model.booster_.model_to_string()
        if old_text!=new_text or (old.thresholds,old.require_nonempty)!=(new.thresholds,new.require_nonempty):
            raise ValueError('Repeated fitted model or calibration settings differ')
        for ra,rb in zip(a['assessment'],b['assessment']):
            if ra['models']['at_T_600_63']['windows']!=rb['models']['shared_threshold']['windows']:
                raise ValueError('Repeated assessment window forecasts differ')
        rows.append({'original_report':first,'repeated_report':second,
            'original_model_sha256':sha256_file(old_path),'repeated_model_sha256':sha256_file(new_path),
            'identical_booster_sha256':hashlib.sha256(old_text.encode()).hexdigest(),
            'fit_and_selection_events_identical':True,'thresholds_identical':True,
            'per_assessment_window_IoU_identical':True})
    result={'pairs':rows,'repeat_scope':'Independent public-frame feature construction, event selection, LGB fitting and calibration. All per-window proxy IoUs agree.',
        'limitations':'Does not repeat seven-million-row formal submission generation; probabilities and latent truth are not audited by per-window IoU equality alone.',
        'formal_Queue':None}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('All three shared-model repetitions matched exactly')

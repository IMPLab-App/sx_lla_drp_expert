"""Freeze the user-authorized unconditional LU M2 and R/CD 54-run queue."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
from lla_experiment_utils import sha256
from run_lla_drp import verify_sources


def prepare(batch, destination):
    batch, destination = batch.resolve(), destination.resolve()
    verify_sources(batch)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ('configs', 'split_indices', 'original_indices', 'reports', 'logs', 'failed_attempts', 'tests'):
        (destination/name).mkdir()
    parent = json.loads((batch/'manifest.json').read_text())
    definitions = batch/'code_audit/rcd_existing_definitions.json'
    scenes = json.loads(definitions.read_text())['scenarios']
    expected = [f'r{i}' for i in range(1, 13)] + [f'cd{i}' for i in range(1, 7)]
    assert [r['scenario'] for r in scenes] == expected
    template = yaml.safe_load(Path(next(t for t in parent['tasks'] if t['version']=='M0')['config']).read_text())
    assert sha256(parent['data_file']) == parent['data_sha256']
    with np.load(parent['data_file'], allow_pickle=False) as data:
        labels, names = data['labels'].copy(), data['class_names'].copy()
    with np.load(template['fixed_split_file'], allow_pickle=False) as split:
        pool = split['train_pool'].copy()
        val, test = split['validation'].copy(), split['test'].copy()
    tasks = [dict(t, stage='lu_m2', enabled_after='all_full_six_complete_technical_checks_only')
             for t in parent['tasks'] if t['version']=='M2']
    provenance = []
    scene_keys = ('num_labels', 'ulb_num_labels', 'lb_imb_type', 'ulb_imb_type',
                  'lb_imb_ratio', 'ulb_imb_ratio', 'full_train_counts', 'labeled_counts')
    for index, scene in enumerate(scenes):
        source = Path(scene['source_config'])
        original = yaml.safe_load(source.read_text())
        ref = Path(scene['reference_indices'])
        source_data = Path(original['data_dir'])/Path(parent['data_file']).name
        assert sha256(source_data) == parent['data_sha256'], scene['scenario']
        assert original['full_train_counts'] == scene['full_counts']
        assert original['labeled_counts'] == scene['labeled_counts']
        with np.load(ref, allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        assert np.array_equal(arrays['validation'], val)
        assert np.array_equal(arrays['test'], test)
        for key, count in (('labeled', scene['labeled_counts']), ('unlabeled', scene['unlabeled_counts'])):
            assert np.bincount(labels[arrays[key]], minlength=len(names)).tolist() == count
            assert len(np.unique(arrays[key])) == len(arrays[key])
            assert np.isin(arrays[key], pool).all()
        assert not np.intersect1d(arrays['labeled'], arrays['unlabeled']).size
        assert (np.asarray(scene['labeled_counts']) + scene['unlabeled_counts']).tolist() == scene['full_counts']
        # Preserve the exact historical L/U/validation/test arrays and order.
        # The current validated loader additionally requires the seed-2 pool.
        original_copy = destination/'original_indices'/ref.name
        shutil.copy2(ref, original_copy)
        frozen = destination/'split_indices'/(scene['case_name']+'.npz')
        arrays.update(train_pool=pool, class_names=names)
        np.savez_compressed(frozen, **arrays)
        with np.load(frozen, allow_pickle=False) as check:
            for key in ('labeled', 'unlabeled', 'validation', 'test'):
                assert np.array_equal(check[key], arrays[key])
        group = 'r' if scene['scenario'].startswith('r') else 'cd'
        for version in ('M0', 'M1', 'M2'):
            cfg = dict(template)
            cfg.update({k: original[k] for k in scene_keys})
            cfg.update(save_name=scene['case_name'],
                save_dir=str(batch/(group+'_'+{'M0':'full', 'M1':'wo_lla_drp', 'M2':'wo_dea'}[version])),
                fixed_split_file=str(frozen), fixed_split_sha256=sha256(frozen),
                e3_adjust_mode='fixed' if version=='M1' else 'lla_hard',
                e3_drp_enabled=version!='M1', ablation_fixed_dea_weights=version=='M2',
                ablation_no_ddde_kl=True, ablation_no_mff=False, ablation_no_dea_aggregator=False,
                study_variant=version, stage='R_CD_three_model_ablation',
                inference_protocol={'M0':'dynamic_DEA_E1raw_E3lla', 'M1':'original_fixed_tau4_raw_fusion',
                                    'M2':'fixed_half_E1raw_E3lla'}[version],
                e3_unlabeled_selection='hard_095' if version=='M1' else 'continuous_DRP',
                resume=False, load_path=None, overwrite=False)
            path = destination/'configs'/(scene['case_name']+'__'+version+'.yaml')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            tasks.append(dict(scenario=scene['scenario'].upper(), case_name=scene['case_name'],
                version=version, stage='rcd', gpu=index%2, config=str(path), config_sha256=sha256(path),
                output=str(Path(cfg['save_dir'])/cfg['save_name']), seed=2,
                fixed_split_sha256=sha256(frozen), full_counts=scene['full_counts'],
                labeled_counts=scene['labeled_counts'], unlabeled_counts=scene['unlabeled_counts']))
        provenance.append(dict(scenario=scene['scenario'].upper(), case_name=scene['case_name'],
            source_config=str(source), source_config_sha256=sha256(source),
            source_indices=str(ref), source_indices_sha256=sha256(ref),
            archived_indices=str(original_copy), frozen_indices=str(frozen),
            frozen_indices_sha256=sha256(frozen),
            transformation='Added train_pool/class_names; original four sample arrays byte-identical and order preserved'))
    assert len(tasks)==60 and sum(t['stage']=='rcd' for t in tasks)==54
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), batch=str(batch), tasks=tasks,
        code_snapshot=parent['code_snapshot'], data_file=parent['data_file'], data_sha256=parent['data_sha256'],
        original_definitions=str(definitions), original_definitions_sha256=sha256(definitions),
        scenario_provenance=provenance, training_seed=2, seeds=1,
        stage_order=['finish_current_full_six', 'lu_m2_six', 'rcd_54'],
        authorization='无论新六任务结果如何，都要接着启动DEA消融、R/CD的54项，启动有问题的话，自行记录问题并解决',
        performance_gate_controls_scheduling=False, technical_checks_required=True,
        models=dict(M0='LLA+DRP+MFF+DEA', M1='fixed_tau4+MFF+DEA', M2='LLA+DRP+MFF+fixed_half_weights'),
        m2_fusion_training_loss='retained', baseline_reuse=False,
        baseline_reuse_reason='Historical R/CD checkpoints do not establish the identical validation-Macro-F1 selection protocol; retrain all 54',
        test_used_for_selection=False, original_preregistration_retained=True)
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (destination/'reports/AUTHORIZATION.md').write_text(
        '# Training authorization amendment\n\n'+manifest['authorization']+'\n\n'
        'This prospective scheduling amendment overrides performance gates only. The original gate remains available for analysis. '
        'All models still select checkpoints by validation Macro-F1; tests never control scheduling or parameters. '
        'M2 retains MFF and fusion auxiliary training, fixing expert weights at 0.5/0.5. '
        'One seed; no significance claim. Six LU M2 plus 54 fresh R/CD runs are queued after current Full six.\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a=p.parse_args();m=prepare(a.batch, a.destination)
    print(json.dumps(dict(prepared=len(m['tasks']),rcd=54,lu_m2=6)))

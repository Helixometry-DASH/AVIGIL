"""Every computational stage requires an explicit --execute opt-in."""
import argparse
import json
from pathlib import Path
from .io import ROOT, load_config, atomic_json

STAGES = ('prepare', 'extract', 'reference-inputs', 'fit-pca', 'fit-cca', 'train-genuine', 'cache-evidence',
          'fit-transport', 'pretrain-boundary', 'train-experts', 'train-crossmodal', 'calibrate', 'prepare-test', 'infer', 'evaluate')


def parser():
    p = argparse.ArgumentParser(description='ReBound: real-reference boundary transfer')
    p.add_argument('stage', choices=('preflight', 'plan') + STAGES)
    p.add_argument('--config', default=str(ROOT / 'configs' / 'server.json'))
    p.add_argument('--source', choices=('audio_source', 'visual_source', 'mixed_source'))
    p.add_argument('--target', choices=('A', 'V', 'MIXED'))
    p.add_argument('--calibration-only', action='store_true', help='Cache evidence only for held-out genuine training clips')
    p.add_argument('--execute', action='store_true', help='Explicitly start the selected computational stage')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--mount', help='Read-only Linux mount prefix, e.g. X:; preflight only')
    p.add_argument('--report', help='Optional preflight JSON destination')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    config = load_config(args.config)
    if args.stage == 'plan':
        print(json.dumps({'stages': list(STAGES), 'execution': 'No stage runs without --execute',
                          'training_sources': ['audio_source', 'visual_source'],
                          'test_protocol': 'Freeze all checkpoints and calibration before prepare-test/infer/evaluate'}, indent=2))
        return
    if args.stage == 'preflight':
        from .preflight import inspect
        report = inspect(config, args.mount)
        if args.report:
            atomic_json(args.report, report)
        print(json.dumps(report, indent=2))
        return
    if not args.execute:
        p.error('No job started. Review the plan/preflight; add --execute only when ready to start this stage.')
    if args.mount:
        p.error('--mount is supported only for read-only preflight; execute on the server')
    if args.calibration_only and (args.stage != 'cache-evidence' or not args.source):
        p.error('--calibration-only requires cache-evidence and an explicit --source')
    if args.stage in ('train-experts', 'train-crossmodal', 'calibrate', 'prepare-test', 'infer', 'evaluate') and not args.source:
        p.error('This stage requires --source')
    if args.stage in ('prepare-test', 'infer', 'evaluate') and not args.target:
        p.error('Final evaluation requires an explicit --target')
    from .runtime import Runtime
    runtime = Runtime(config)
    try:
        if args.stage == 'evaluate':
            if runtime.rank == 0:
                from .inference import evaluate
                print(json.dumps(evaluate(config, args.source, args.target), indent=2))
        elif args.stage == 'prepare':
            if runtime.rank == 0:
                from .protocol import prepare_manifests
                print(json.dumps(prepare_manifests(config), indent=2))
        elif args.stage == 'fit-pca':
            if runtime.rank == 0:
                from .pipeline import save_pca
                save_pca(config)
        elif args.stage == 'fit-cca':
            if runtime.rank == 0:
                from .pipeline import save_cca
                save_cca(config)
        elif args.stage in ('extract', 'reference-inputs', 'cache-evidence', 'prepare-test'):
            from .pipeline import preparation_rows, cache_evidence
            from .media import build_reference_inputs
            from .extract_batched import extract as extract_regions
            if args.stage == 'prepare-test':
                # Frozen calibrations are required before test feature computation.
                from .inference import Predictor
                Predictor(config, args.source, runtime.device)
                from .protocol import selected_rows
                rows = selected_rows(config, args.source, 'test', target=args.target, final_evaluation=True)
                extract_regions(config, rows, runtime.device, runtime.rank, runtime.world)
                runtime.barrier()
                build_reference_inputs(config, rows, runtime.rank, runtime.world)
                runtime.barrier()
                cache_evidence(config, rows, runtime)
            else:
                if args.calibration_only:
                    from .protocol import selected_rows, genuine_rows
                    rows = genuine_rows(selected_rows(config, args.source, 'train'), 'calibration')
                else:
                    rows = preparation_rows(config)
                if args.stage == 'extract':
                    extract_regions(config, rows, runtime.device, runtime.rank, runtime.world)
                elif args.stage == 'reference-inputs':
                    build_reference_inputs(config, rows, runtime.rank, runtime.world)
                else:
                    cache_evidence(config, rows, runtime)
        elif args.stage == 'calibrate':
            from .inference import calibrate
            calibrate(config, args.source, runtime)
        elif args.stage == 'infer':
            from .inference import infer
            infer(config, args.source, args.target, runtime)
        else:
            from .pipeline import train_stage
            stage = {'train-genuine': 'genuine', 'fit-transport': 'transport',
                     'pretrain-boundary': 'boundary', 'train-experts': 'experts',
                     'train-crossmodal': 'crossmodal'}[args.stage]
            train_stage(config, stage, runtime, args.source, args.resume)
        runtime.barrier()
    finally:
        runtime.close()


if __name__ == '__main__':
    main()

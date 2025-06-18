from pm2s import CRNNJointPM2S
import argparse
import torch
def main(args):

    pm2s_processor = CRNNJointPM2S(
        beat_pps_args = {
            'prob_thresh': 0.5,
            'penalty': 1.0,
            'merge_downbeats': False,
            'method': 'dp',
        },
        ticks_per_beat = 480,
        quantization = args.quantization,
        device = torch.device('cpu')
    )

    pm2s_processor.convert(
        args.perf_midi,
        args.score_midi,
        include_time_signature=args.time_signature,
        include_key_signature=args.key_signature,
        include_tempo_changes=args.tempo_changes
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--perf-midi', type=str, required=True)
    parser.add_argument('--score-midi', type=str, required=True)
    parser.add_argument('--quantization', type=int, default=32)
    parser.add_argument('--time-signature', type=bool, default=True)
    parser.add_argument('--key-signature', type=bool, default=True)
    parser.add_argument('--tempo-changes', type=bool, default=True)
    args = parser.parse_args()
    main(args)

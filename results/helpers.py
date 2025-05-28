
import numpy as np


KEYWORDS = ['125M-50k', '125M-200k', '760M-50k', '760M-200k']
VARIANTS = ['ln', 'dyisrusp', 'dyisru', 'dytsp', 'dyt']
STATS = ['mean', 'std', 'min', 'max', 'layers']


class Helpers:

    @staticmethod
    def steps_from_run_name(run_name: str) -> float:
        """
        Args:
            run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'

        Return:
            steps: e.g. 50000.0
        """
        if '-50k-' in run_name:
            return 50000.0
        elif '-200k-' in run_name:
            return 200000.0
        else:
            raise ValueError(f'ERROR! Could not parse run_name = {run_name}')


    @staticmethod
    def get_keyword_from_run_name(_run_name: str) -> str:
        """
        Args:
            _run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'

        Return:
            keyword: e.g. '125M-50k'
        """
        for keyword in KEYWORDS:
            if keyword in _run_name:
                return keyword
        raise ValueError(f'ERROR! could not find keyword in run_name = {_run_name}')

    @staticmethod
    def get_variant_from_run_name(_run_name: str) -> str:
        """
        Args:
            _run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'

        Return:
            variant: e.g. 'dyt'
        """
        for variant in VARIANTS:
            if variant in _run_name:
                return variant
        raise ValueError(f'ERROR! could not find variant in run_name = {_run_name}')

    @staticmethod
    def get_init_from_run_name(_run_name: str, variant: str) -> list[float]:
        """
        Args:
            _run_name: e.g. 'exp12B-125M-50k-dytx-i0p5x0p5-s1'
            variant: e.g. 'dyt'

        Return:
            init: e.g. [0.5, 0.5]
        """
        init_str = _run_name.split(variant)[-1].split('-i')[-1].split('-s')[0]
        init_strs = init_str.split('x')
        init_strs = [elem.replace('p', '.') for elem in init_strs]
        return [float(elem) for elem in init_strs]

    @classmethod
    def reformat_loss(cls, _loss: dict[str, float], diagonal: bool = True) -> dict:
        """
        Args:
            _loss:  {
                'exp12A-125M-50k-ln-i1p0-s1': 3.139458656311035, 
                'exp12B-125M-50k-dyt-i0p5-s1': 3.342463493347168,
            }
        Return:
            reformatted_loss: {
                '125M-50k': {
                    'ln': 3.14,
                    'dyt': {
                        0.5: 3.34,
                    }
                }
            }
        """
        output = {keyword: {'ln': None, 'dyt': {}, 'dyisrusp': {}, 'dytsp': {}, 'dyisru': {}} for keyword in KEYWORDS}
        for run_name, val_loss in _loss.items():
            keyword = cls.get_keyword_from_run_name(run_name)
            variant = cls.get_variant_from_run_name(run_name)
            init = cls.get_init_from_run_name(run_name, variant)
            if 0:
                print(run_name, keyword, variant, init, val_loss)
            if variant == 'ln':
                output[keyword][variant] = val_loss
            else:
                if diagonal is False:
                    if init[0] not in output[keyword][variant]:
                        output[keyword][variant][init[0]] = {}
                    output[keyword][variant][init[0]][init[1]] = val_loss
                else:
                    output[keyword][variant][init[0]] = val_loss

        # sort
        for keyword in output:
            for variant in output[keyword]:
                if variant != 'ln':
                    output[keyword][variant] = {
                        k: v for k, v in sorted(output[keyword][variant].items(), key=lambda item: item[0])
                    }
                    if diagonal is False:
                        for elem in output[keyword][variant]:
                            output[keyword][variant][elem] = {
                                k: v for k, v in sorted(output[keyword][variant][elem].items(), key=lambda item: item[0])
                            }

        return output

    @classmethod
    def reformat_alpha_beta(cls, _alpha_beta: dict[str, float], diagonal: bool = True) -> dict:
        """
        Args:
            _alpha_beta:  {
                'exp12B-125M-50k-dyt-i0p5x0p5-s1': {'mean': 3.342463493347168, 'std': [..]},
            }
        Return:
            reformatted_alpha_beta_mean: {
                '125M-50k': {
                    'dyt': {
                        0.5: {
                            0.5 : {
                                'mean': 3.34,
                                'std': ..,
                            }
                        }
                    }
                }
            }
        """
        output = {
            keyword: {'ln': None, 'dyt': {}, 'dyisrusp': {}, 'dytsp': {}, 'dyisru': {}}
            for keyword in KEYWORDS
        }
        for run_name, alpha_beta in _alpha_beta.items():
            keyword = cls.get_keyword_from_run_name(run_name)
            variant = cls.get_variant_from_run_name(run_name)
            init = cls.get_init_from_run_name(run_name, variant)
            if 0:
                print(run_name, keyword, variant, init, alpha_beta)
            if variant == 'ln':
                output[keyword][variant] = None
            else:
                for stat in STATS:
                    if stat not in output[keyword][variant]:
                        output[keyword][variant][stat] = {}
                    if diagonal is False:
                        if init[0] not in output[keyword][variant][stat]:
                            output[keyword][variant][stat][init[0]] = {}
                        output[keyword][variant][stat][init[0]][init[1]] = alpha_beta[stat]
                    else:
                        output[keyword][variant][stat][init[0]] = alpha_beta[stat]
        return output

    @classmethod
    def reformat_average_time_per_step(cls, _time_per_step: dict) -> dict[str, str]:
        runs = sorted(list(set(['-'.join(k.split('-s')[0].split('-')[:4]) for k in _time_per_step])))
        _avg_time_per_step = {}
        for run in runs:
            values = [v for k, v in _time_per_step.items() if run in k]
            _avg_time_per_step[run] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

        _avg_time_per_step_str = {k: f"{v['mean']:.4f} +- {v['std']:.4f}" for k, v in _avg_time_per_step.items()}
        _avg_time_per_step_str = {k: v.split(' +- 0.0000')[0] for k, v in _avg_time_per_step_str.items() if 'A' in k or 'B' in k or 'C' in k}
        return _avg_time_per_step_str
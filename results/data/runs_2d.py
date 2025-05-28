RUNS_2D = {
    # 'exp12B-125M-50k-dyt-i0p5x0p5-s1': '5mer7pfj',
    # 'exp12B-125M-50k-dyt-i0p5x0p69-s1': 'nmfzorri',
    # ADD run_name: run_id HERE
}

RUNS_2D_FILTERED = {
    k: v
    for k, v in RUNS_2D.items()
}

RUNS_2D_LOSS_MANUAL = {
    # exp35
    'exp35B-760M-200k-dyt-i0p69x0p5-s1': 2.8160,
    'exp35B-760M-200k-dyt-i1p0x0p5-s1': 2.8491,
    'exp35B-760M-200k-dyt-i1p0x0p69-s1': 2.8009,
    'exp35B-760M-200k-dyt-i2p0x0p5-s1': 2.9838,
    'exp35B-760M-200k-dyt-i2p0x0p69-s1': 4.5645,
    'exp35B-760M-200k-dyt-i2p0x1p0-s1': 2.7372,
    'exp35B-760M-200k-dyt-i3p0x0p5-s1': 5.2271,
    'exp35B-760M-200k-dyt-i3p0x0p69-s1': 4.9200,
    'exp35B-760M-200k-dyt-i3p0x1p0-s1': 4.9830,
    'exp35B-760M-200k-dyt-i3p0x2p0-s1': 5.0588,
    'exp35C-760M-200k-dyisrusp-i-2p0x-1p0-s1': 4.9988,
    'exp35C-760M-200k-dyisrusp-i-2p0x1p0-s1': 4.8990,
    'exp35C-760M-200k-dyisrusp-i-2p0x2p0-s1': 5.0274,
    'exp35C-760M-200k-dyisrusp-i-2p0x4p0-s1': 4.8844,
    'exp35C-760M-200k-dyisrusp-i-1p0x1p0-s1': 2.7693,
    'exp35C-760M-200k-dyisrusp-i-1p0x2p0-s1': 2.7693,
    'exp35C-760M-200k-dyisrusp-i-1p0x4p0-s1': 4.8173,
    'exp35C-760M-200k-dyisrusp-i1p0x2p0-s1': 2.8057,
    'exp35C-760M-200k-dyisrusp-i1p0x4p0-s1': 2.8639,
    'exp35C-760M-200k-dyisrusp-i2p0x4p0-s1': 4.8844,
}
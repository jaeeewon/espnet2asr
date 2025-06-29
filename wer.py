# "/home/<user>/espnet/egs2/l3das22/enh1/local/metric.py"

import jiwer

def wer(ans, inf):
    return jiwer.wer(ans, inf)
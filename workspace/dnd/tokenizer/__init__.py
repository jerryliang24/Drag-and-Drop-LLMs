import re
import warnings


def ignore_std_warnings(message, category, filename, lineno, file=None, line=None):
    if re.match(r"std\(\).*", str(message)):
        return  # all the warnings start with std() will be ignored.
    else:  # show other warnings
        warnings._showwarnmsg_impl(warnings.WarningMessage(message, category, filename, lineno))


warnings.showwarning = ignore_std_warnings


from .register import (
    Qwen253BVL_LoRA_Tokenizer2D,
    Qwen257BLoRA_Tokenizer2D,
    Qwen2505LoRA_Tokenizer2D,
    Qwen2515LoRA_Tokenizer2D,
)
from .tokenizer import TokenizerInterface

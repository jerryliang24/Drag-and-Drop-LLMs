cd ./workspace/main

python exploration/test_overhead/test_overhead_0.5B.py --eval_dataset fullARC-cBERT2000 --test_dataset ARC-c
python exploration/test_overhead/test_overhead_3B.py
python exploration/test_overhead/test_overhead_code_1.5B.py --eval_dataset full-Code1000 --test_dataset Code-74k-ShareGPT
python exploration/test_overhead/test_overhead_math_1.5B.py --eval_dataset full-Math1000 --test_dataset Competition_Math
python exploration/test_overhead/test_overhead_code_7B.py --eval_dataset full-Code1000 --test_dataset Code-74k-ShareGPT
python exploration/test_overhead/test_overhead_math_7B.py --eval_dataset full-Math4000 --test_dataset Competition_Math

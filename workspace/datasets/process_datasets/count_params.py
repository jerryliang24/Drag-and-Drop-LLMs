from safetensors.torch import safe_open

diction = safe_open("", framework="pt")
count = 0
for k in diction.keys():
    v = diction.get_tensor(k)
    print(v.shape)
    count += v.numel()
print(count)


def list_layer_keys(module, key_so_far="", max_depth=4):
    if key_so_far != "":
        print(f"{key_so_far}: {type(module)}")
        key_so_far = key_so_far + '.'

    if max_depth==0:
        return

    if len(module._modules) > 0:
        for new_key, submodule in module._modules.items():
            list_layer_keys(submodule, key_so_far + new_key, max_depth-1)
        print()
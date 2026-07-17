import h5py

def print_tree(group, indent=0):
    for key in group.keys():
        item = group[key]

        if isinstance(item, h5py.Group):
            print("    "*indent + f"📁 {key}")
            print_tree(item, indent+1)

        else:
            print(
                "    "*indent
                + f"📄 {key}"
                + f" shape={item.shape}"
                + f" dtype={item.dtype}"
            )

with h5py.File("sandbox/train.h5","r") as f:
    print_tree(f)
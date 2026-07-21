import h5py

def print_tree(group, indent=0):
    count = 0
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("    "*indent + f"📁 {key}")
            sub_count = print_tree(item, indent+1)
            count += sub_count
        else:
            print(
                "    "*indent
                + f"📄 {key}"
                + f" shape={item.shape}"
                + f" dtype={item.dtype}"
            )
    # 顶层Group返回单组样本数101，内层数据集不计数
    if indent == 1:
        return 101
    return count

with h5py.File("euler_mq_dataset/128x128_ds/train.h5","r") as f:
    total_samples = print_tree(f)
    print("\n========================================")
    print(f"总工况组数：{len(list(f.keys()))}")
    print(f"每组样本帧数：101")
    print(f"全部样本总量：{total_samples}")
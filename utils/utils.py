import pickle

def np_print(np_array):
    h, w = np_array.shape
    if h == 1 or w == 1:
        num_fmt = "{:.6f}"
    else:
        num_fmt = "{:.3f}"

    str_array = "[\n" + ",\n".join([
        "\t[" + ",\t".join([num_fmt.format(v).rjust(10, ' ') for v in row]) + "]"
        for row in np_array
    ]) + "\n]"
    ret = "np.array(" + str_array + ")"
    return ret

def numeric_sort(file_name):
    return int(file_name.split("_")[-1].split(".")[0])

def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.loads(f.read())
    return data
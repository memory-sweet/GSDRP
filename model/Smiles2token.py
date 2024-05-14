import torch

NewDic = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, ':': 5, '[': 6, ']': 7, '2': 8, '1': 9,
              'H': 10,
              'c': 11, 'C': 12, '3': 13, '(': 14, ')': 15, '4': 16, '5': 17, 'O': 18, '6': 19, '7': 20, '8': 21,
              '9': 22,
              '0': 23, '=': 24, 'N': 25, 'n': 26, 'F': 27, '-': 28, 'S': 29, '/': 30, 'Cl': 31, 's': 32, 'o': 33,
              '#': 34,
              '+': 35, 'Br': 36, '\\': 37, 'P': 38, 'I': 39, '-2': 40, '-3': 41, 'Si': 42, 'B': 43, '-4': 44}


def split(sm):
    """
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    """
    arr = []
    i = 0
    while i < len(sm) - 1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', 'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i] == '%':
            arr.append(sm[i:i + 3])
            i += 3
        elif sm[i] == 'C' and sm[i + 1] == 'l':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'C' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'C' and sm[i + 1] == 'u':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'N' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'N' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'R' and sm[i + 1] == 'b':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'R' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'X' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'L' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'l':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 's':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'g':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'u':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'M' and sm[i + 1] == 'g':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'M' and sm[i + 1] == 'n':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'T' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'Z' and sm[i + 1] == 'n':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 's' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 's' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 't' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'H' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '2':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '3':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '4':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '2':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '3':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '4':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'K' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'F' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm) - 1:
        arr.append(sm[i])
    return ' '.join(arr)

def get_inputs(sm):
    seq_len = 190
    sm = sm.split()
    if len(sm) > 218:
        # print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [NewDic.get(token, NewDic['<unk>']) for token in sm]
    ids = [NewDic['<sos>']] + ids + [NewDic['<eos>']]
    seg = [1] * len(ids)
    padding = [NewDic['<pad>']] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)
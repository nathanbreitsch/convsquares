from collections import deque

import torch


def marching_squares(x):
    offset_tensor = torch.tensor([case_to_offset(i) for i in range(0, 16)])

    degen_offset_tensor = torch.tensor([case_to_offset_degen(i) for i in range(0, 16)])
    with torch.no_grad():
        weight = torch.tensor([(1, 2), (4, 8)], dtype=torch.uint8, requires_grad=False)[
            None, None, :, :
        ]
        conv_out = torch.nn.functional.conv2d(
            x[None, None, :, :].to(dtype=torch.uint8),
            weight=weight,
            stride=(1, 1),
            padding=1,
        ).squeeze()
        isedge = (conv_out > 0) & (conv_out < 15)
        vertices = isedge.nonzero()
        type_indices = conv_out[isedge]
        offsets = offset_tensor[type_indices.tolist()]
        edges = vertices[:, None, :] + offsets

        # handle degenerate cases
        degen = (type_indices == 6) | (type_indices == 9)
        degen_type_indices = type_indices[degen]
        degen_vertices = vertices[degen]
        degen_offsets = degen_offset_tensor[degen_type_indices.tolist()]
        degen_edges = degen_vertices[:, None, :] + degen_offsets
        all_edges = torch.cat([edges, degen_edges], dim=0)
        contours = assemble_contours(all_edges)
        return [torch.tensor(c) for c in contours]


def case_to_offset(square_case):
    top = 0, 1 / 2
    bottom = 1, 1 / 2
    left = 1 / 2, 0
    right = 1 / 2, 1

    if square_case == 1:
        return (top, left)
    elif square_case == 2:
        return (right, top)
    elif square_case == 3:
        return (right, left)
    elif square_case == 4:
        return (left, bottom)
    elif square_case == 5:
        return (top, bottom)
    elif square_case == 6:
        return (left, top)
    elif square_case == 7:
        return (right, bottom)
    elif square_case == 8:
        return (bottom, right)
    elif square_case == 9:
        return (top, right)
    elif square_case == 10:
        return (bottom, top)
    elif square_case == 11:
        return (bottom, left)
    elif square_case == 12:
        return (left, right)
    elif square_case == 13:
        return (top, right)
    elif square_case == 14:
        return (left, top)
    else:
        return (top, top)


def case_to_offset_degen(square_case):
    top = 0, 1 / 2
    bottom = 1, 1 / 2
    left = 1 / 2, 0
    right = 1 / 2, 1

    if square_case == 6:
        return (right, bottom)
    elif square_case == 9:
        return (bottom, left)
    else:
        return (top, top)

# modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours_cy.pyx
def assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        from_point = tuple(from_point.tolist())
        to_point = tuple(to_point.tolist())
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            else:
                if tail_num > head_num:
                    head.extend(tail)
                    contours.pop(tail_num, None)
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:
                    tail.extendleft(reversed(head))
                    starts.pop(head[0], None)
                    contours.pop(head_num, None)
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)

    return [contour for _, contour in sorted(contours.items())]

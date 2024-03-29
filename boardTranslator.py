import itertools
import json
import numpy as np

WALL = '#'
EMPTY_CELL = '@'

BOARD_626099 = '[["#","#","(33)","(3)","(35)","#","#"],["#","(16)","@","@","@","(8)","#"],["#","(4-20)","@","@","@","@","#"],["(6)","@","@","(16-16)","@","@","#"],["(19)","@","@","@","@","#","#"],["#","(22)","@","@","@","#","#"],["#","#","#","#","#","#","#"]]'
BOARD_622655 = '[["#","(11)","(35)","#","#","(16)","(9)","(14)","#","#","#","#","(45)","(6)","#"],["(16)","@","@","(11)","(29-14)","@","@","@","(6)","#","#","(6)","@","@","#"],["(39)","@","@","@","@","@","@","@","@","#","(10)","(9-12)","@","@","#"],["#","(17-10)","@","@","@","(12)","(13)","@","@","(7-24)","@","@","@","#","#"],["(13)","@","@","(13)","@","@","#","#","(4-11)","@","@","@","@","(17)","#"],["(17)","@","@","(4-14)","@","@","(9)","(17-3)","@","@","#","(9)","@","@","#"],["#","(12)","@","@","(11-17)","@","@","@","@","#","#","(6-12)","@","@","#"],["#","(10)","(45-3)","@","@","(6-14)","@","@","#","(12)","(11-6)","@","@","(16)","#"],["(7)","@","@","(14-3)","@","@","#","#","(34)","@","@","@","@","@","#"],["(34)","@","@","@","@","@","#","(13)","(4-5)","@","@","(10-17)","@","@","#"],["#","(11-15)","@","@","#","#","(3-8)","@","@","(24-4)","@","@","(36)","#","#"],["(4)","@","@","#","#","(8-18)","@","@","@","@","(25-15)","@","@","(17)","#"],["(11)","@","@","(15)","(14-5)","@","@","#","(17)","@","@","(13)","@","@","#"],["#","(21)","@","@","@","@","(4)","(21)","(13)","@","@","(11-16)","@","@","#"],["#","(15-21)","@","@","@","(8)","@","@","(11)","(17-18)","@","@","@","(17)","#"],["(11)","@","@","#","#","(43)","@","@","@","@","@","@","@","@","#"],["(17)","@","@","#","#","#","(24)","@","@","@","#","(15)","@","@","#"],["#","#","#","#","#","#","#","#","#","#","#","#","#","#","#"]]'
BOARD_357465 = '[["#","#","(20)","(12)","(14)","#","(13)","(25)","#","#","#","(29)","(7)","#","(15)","(30)","#","(7)","(13)","#"],["#","(23-15)","@","@","@","(17)","@","@","#","(7)","(3-15)","@","@","(14)","@","@","(24-15)","@","@","#"],["(30)","@","@","@","@","(26-10)","@","@","(13-12)","@","@","@","@","(18)","@","@","@","@","@","#"],["(11)","@","@","(9)","@","@","(39-21)","@","@","@","@","@","(8)","(7-23)","@","@","@","(22)","#","#"],["(8)","@","@","(10)","(15)","@","@","@","@","(12)","(18)","@","@","@","(24)","@","@","@","(23)","#"],["#","(7)","@","@","(15-7)","@","@","(9-8)","@","@","(11)","(3)","@","@","(18)","(19)","@","@","@","#"],["#","#","(28-23)","@","@","@","@","@","(13)","@","@","(17)","(10)","@","@","(11)","(16-8)","@","@","#"],["#","(23-23)","@","@","@","(4-15)","@","@","(15)","(6-10)","@","@","#","(41-27)","@","@","@","@","@","#"],["(15)","@","@","#","(10)","@","@","(14-28)","@","@","@","@","(4-28)","@","@","@","@","(12)","(17)","#"],["(11)","@","@","#","(17)","@","@","@","@","@","(8)","(9-4)","@","@","(17)","#","(14)","@","@","#"],["(17)","@","@","(12)","(5)","(11-16)","@","@","(11)","(23-33)","@","@","@","@","@","#","(4)","@","@","#"],["#","(21)","(16-14)","@","@","@","@","(20)","@","@","@","@","(13-9)","@","@","(6)","(23-7)","@","@","#"],["(21)","@","@","@","@","@","(18)","(10)","@","@","(10)","(17)","@","@","(21-18)","@","@","@","#","#"],["(12)","@","@","(25)","(6)","@","@","(12)","(14)","@","@","(14-22)","@","@","@","@","@","(34)","#","#"],["(15)","@","@","@","(12)","(17)","@","@","(29)","(5)","@","@","(26-8)","@","@","(17)","@","@","(19)","#"],["#","(19)","@","@","@","(11-21)","@","@","@","(11)","(4-28)","@","@","@","@","(9)","(16)","@","@","#"],["#","(11)","(3-6)","@","@","@","#","(8-19)","@","@","@","@","@","(8-7)","@","@","(14-7)","@","@","#"],["(31)","@","@","@","@","@","(20)","@","@","@","@","(8)","@","@","(20)","@","@","@","@","#"],["(4)","@","@","(4)","@","@","(14)","@","@","#","#","(16)","@","@","(20)","@","@","@","#","#"],["#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#"]]'
BOARD_383659 = '[["#","#","(17)","(16)","#","#","(16)","(6)","#","#","(15)","(7)","#","(7)","(29)","#","(37)","(8)","#"],["#","(6-14)","@","@","#","(4)","@","@","(33)","(15)","@","@","(3-9)","@","@","(13)","@","@","#"],["(13)","@","@","@","(27)","(12-17)","@","@","@","(9-19)","@","@","@","@","@","(17-6)","@","@","#"],["(9)","@","@","(6-19)","@","@","@","(13-11)","@","@","@","(25)","@","@","@","@","@","@","#"],["(20)","@","@","@","@","@","(23)","@","@","@","(37)","(17)","(9)","(12)","@","@","@","(7)","#"],["#","(6)","(35-6)","@","@","(10)","(6-5)","@","@","(12-19)","@","@","@","(13)","(15-20)","@","@","@","#"],["(8)","@","@","(6)","@","@","@","(9-38)","@","@","@","@","@","@","@","(4)","@","@","#"],["(11)","@","@","(22-35)","@","@","@","@","@","@","@","(9-9)","@","@","@","(16-6)","@","@","#"],["(13)","@","@","@","(30)","(7)","@","@","@","(10-14)","@","@","#","(8-9)","@","@","(23)","(17)","#"],["#","(7-17)","@","@","@","(13)","(17)","#","(11-11)","@","@","@","(11-25)","@","@","@","@","@","#"],["(37)","@","@","@","@","@","@","(17-11)","@","@","@","(3-8)","@","@","@","(11-9)","@","@","#"],["(10)","@","@","(29)","@","@","@","@","@","(16)","@","@","@","#","(22)","@","@","@","#"],["(5)","@","@","(11)","@","@","(16)","@","@","#","(4)","@","@","#","(15)","@","@","#","#"],["#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#","#"]]'
BOARDS = [BOARD_622655, BOARD_357465, BOARD_383659, BOARD_626099]


def load_board_str(board_string):
    return np.matrix(json.loads(board_string))


def label_rows(board):
    board = board.copy()
    n, m = board.shape
    row_idx = -1
    col_idx = 0
    in_row = False
    rows_size = []
    for i in range(n):
        for j in range(m):
            if board[i, j] == EMPTY_CELL:
                if not in_row:
                    in_row = True
                    row_idx += 1
                    col_idx = 0

                board[i, j] = f'{row_idx}~{col_idx}'
                col_idx += 1

            else:
                if in_row:
                    rows_size.append(col_idx)
                    in_row = False

    return board, rows_size


def eval_cell(cell_str: str):
    if cell_str.startswith('('):
        cell_str = cell_str[1:-1]
        return tuple(cell_str.split('-'))

    if '~' in cell_str:
        idxs = cell_str.split('~')
        return tuple(int(idx) for idx in idxs)

    return cell_str


def extract_sum(board, is_transposed=False):
    sum_idx = 0 if is_transposed else -1
    n, m = board.shape

    sums = []
    for i in range(n):
        for j in range(m - 1):
            cell_val = eval_cell(board[i, j])
            next_cell_val = eval_cell(board[i, j + 1])
            if next_cell_val == EMPTY_CELL and isinstance(cell_val, tuple):
                sums.append(int(cell_val[sum_idx]))

    return sums


def extract_cols(board, labeled_board):
    board, labeled_board = board.T, labeled_board.T
    n, m = board.shape

    cols, cur_col = [], []
    for i in range(n):
        for j in range(m):
            if board[i, j] == EMPTY_CELL:
                cell_idx = eval_cell(labeled_board[i, j])
                cur_col.append(cell_idx)
            else:
                if cur_col:
                    cols.append(cur_col)
                    cur_col = []

    return cols


def extract_board_params(board):
    labeled_board, rows_size = label_rows(board)
    rows_sum = extract_sum(board)
    rows_opt = [get_parts(row_sum, row_size) for row_sum, row_size in zip(rows_sum, rows_size)]
    cols_sum = extract_sum(board.T, is_transposed=True)
    cols_map = extract_cols(board, labeled_board)

    return rows_size, rows_sum, rows_opt, cols_sum, cols_map


def get_board_parms_by_idx(i):
    board = load_board_str(BOARDS[i])
    return extract_board_params(board)


def get_parts(row_sum, row_size):
    possible_nums = range(1, 10)
    all_perm = itertools.combinations(possible_nums, row_size)
    parts = []
    for perm in all_perm:
        if sum(perm) == row_sum:
            parts.append(perm)
    return parts


if __name__ == '__main__':
    rows_size, rows_sum, rows_opt, cols_sum, cols_map = get_board_parms_by_idx(0)
    print('rows_size:', rows_size)
    print('rows_sum:', rows_sum)
    print('rows_opt:', rows_opt)
    print('cols_sum:', cols_sum)
    print('cols_map:', cols_map)

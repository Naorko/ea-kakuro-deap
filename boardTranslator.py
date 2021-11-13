import json
import numpy as np

WALL = '#'
EMPTY_CELL = '@'

board_626099 = '[["#","#","(33)","(3)","(35)","#","#"],["#","(16)","@","@","@","(8)","#"],["#","(4-20)","@","@","@","@","#"],["(6)","@","@","(16-16)","@","@","#"],["(19)","@","@","@","@","#","#"],["#","(22)","@","@","@","#","#"],["#","#","#","#","#","#","#"]]'


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
            next_cell_val = eval_cell(board[i, j+1])
            if next_cell_val == EMPTY_CELL and isinstance(cell_val, tuple):
                sums.append(cell_val[sum_idx])

    return sums


if __name__ == '__main__':
    board = load_board_str(board_626099)

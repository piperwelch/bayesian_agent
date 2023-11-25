from collections import deque

def is_valid_move(maze, row, col, visited):
    rows, cols = len(maze), len(maze[0])
    return 0 <= row < rows and 0 <= col < cols and maze[row][col] == 0 and not visited[row][col]

def shortest_path(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False] * cols for _ in range(rows)]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    visited[start[0]][start[1]] = True

    while queue:
        current_row, current_col, distance = queue.popleft()

        if (current_row, current_col) == end:
            return distance

        for dr, dc in directions:
            new_row, new_col = current_row + dr, current_col + dc

            if is_valid_move(maze, new_row, new_col, visited):
                queue.append((new_row, new_col, distance + 1))
                visited[new_row][new_col] = True

    return -1  # No path found

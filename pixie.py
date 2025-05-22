import requests


class GridBoard:
    def __init__(self, width, height):
        # Initialize a width x height grid with all pixels off (0 represents off)
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def clear(self):
        """Turn off all pixels in the grid."""
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = 0

    def draw_pixel(self, x, y, value=1):
        """Set a single pixel (x,y) to the given value (1 for on, 0 for off)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = value

    def draw_vertical_line(self, x, y1, y2, value=1):
        """Draw a vertical line at column x from row y1 to row y2 (inclusive)."""
        if x < 0 or x >= self.width:
            return  # x out of bounds
        # Ensure y1 <= y2 for iteration
        if y1 > y2:
            y1, y2 = y2, y1
        for y in range(max(0, y1), min(self.height, y2 + 1)):
            self.grid[y][x] = value

    def draw_horizontal_line(self, y, x1, x2, value=1):
        """Draw a horizontal line at row y from column x1 to x2 (inclusive)."""
        if y < 0 or y >= self.height:
            return  # y out of bounds
        # Ensure x1 <= x2 for iteration
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(max(0, x1), min(self.width, x2 + 1)):
            self.grid[y][x] = value

    def display(self):
        """Print the current grid to console using '#' for on and '.' for off."""
        for y in range(self.height):
            row_str = "".join('#' if pixel else '.' for pixel in self.grid[y])
            print(row_str)

    def process_command(self, command):
        """Parse a textual command and update the grid accordingly."""
        cmd = command.lower().strip()
        if cmd.startswith("turn on pixel at"):
            # e.g. "Turn on pixel at (5,5)"
            coords = cmd.split("at")[1]
            nums = [int(s) for s in coords.replace('(', '')
            .replace(')', '')
            .replace(',', ' ')
            .split()
                    if s.isdigit()]
            if len(nums) >= 2:
                x, y = nums[0], nums[1]
                self.draw_pixel(x, y, value=1)
        elif cmd.startswith("turn off pixel at"):
            # e.g. "Turn off pixel at (3,4)"
            coords = cmd.split("at")[1]
            nums = [int(s) for s in coords.replace('(', '')
            .replace(')', '')
            .replace(',', ' ')
            .split()
                    if s.isdigit()]
            if len(nums) >= 2:
                x, y = nums[0], nums[1]
                self.draw_pixel(x, y, value=0)
        elif cmd.startswith("draw a vertical line"):
            # e.g. "Draw a vertical line at x=2 from y=2 to y=6"
            try:
                at_index = cmd.index("x=")
                from_index = cmd.index("from y=")
            except ValueError:
                return  # if format not found, skip
            x_val_str = cmd[at_index + 2: from_index]  # between "x=" and " from y"
            x_val = int(''.join(ch for ch in x_val_str if ch.isdigit()))
            range_part = cmd[from_index:]
            nums = [int(s) for s in range_part.split() if s.isdigit()]
            if len(nums) >= 2:
                y1, y2 = nums[0], nums[1]
                self.draw_vertical_line(x_val, y1, y2, value=1)
        elif cmd.startswith("draw a horizontal line"):
            # e.g. "Draw a horizontal line at y=3 from x=4 to x=12"
            try:
                at_index = cmd.index("y=")
                from_index = cmd.index("from x=")
            except ValueError:
                return
            y_val_str = cmd[at_index + 2: from_index]
            y_val = int(''.join(ch for ch in y_val_str if ch.isdigit()))
            range_part = cmd[from_index:]
            nums = [int(s) for s in range_part.split() if s.isdigit()]
            if len(nums) >= 2:
                x1, x2 = nums[0], nums[1]
                self.draw_horizontal_line(y_val, x1, x2, value=1)
        elif cmd.startswith("clear"):
            self.clear()
        else:
            # For unrecognized commands, just ignore or handle differently
            pass


def query_ollama(prompt, model="long-gemma", host="http://localhost:11434"):
    """
    Send a prompt to Ollama running locally and return the text response.
    Adjust as needed for your Ollama setup or additional parameters.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        # Ollama returns a JSON array of tokens or a streaming response.
        # In the simplest case, you can parse it like below if the server
        # is configured to return a single final text. 
        data = response.json()
        # Depending on Ollama version, data might be a dict with "done" or "completion"
        # If it's token-by-token, you might need to accumulate. 
        if isinstance(data, list):
            # older streaming style: each item might have "token" or "done"
            text = "".join(item.get("token", "") for item in data if not item.get("done"))
            return text
        elif isinstance(data, dict) and "completion" in data:
            return data["completion"]
        else:
            return str(data)
    except requests.RequestException as e:
        print("Error querying Ollama:", e)
        return ""


if __name__ == "__main__":
    # Initialize our 16x16 grid
    grid = GridBoard(16, 16)

    # Example interaction loop:
    # We prompt Ollama for the next command, feed that command to our grid, then display.
    # In a real scenario, you might have your own conversation or logic to shape these prompts.

    # We'll do a few steps just for demonstration. Adjust the prompt so that the model
    # returns text like "Turn on pixel at (5,5)" or "Draw a horizontal line at y=3 from x=4 to x=12", etc.

    # 1) Provide a short system-level style context so the model knows what we want:
    system_context = (
        "You are controlling a 16x16 pixel board. "
        "You can issue commands like:\n"
        "- Turn on pixel at (x,y)\n"
        "- Turn off pixel at (x,y)\n"
        "- Draw a vertical line at x=N from y=A to y=B\n"
        "- Draw a horizontal line at y=N from x=A to x=B\n"
        "- Clear\n"
        "Only respond with a single command each time."
    )

    # You might store or maintain a conversation. For simplicity, we'll do a few
    # queries in a row, each time showing the updated grid.

    # Prompt #1
    prompt = system_context + "\n\nPlease issue your first command."
    response = query_ollama(prompt, model="gemma-long")

    print("\n----- LLM Response #1 -----")
    print(response)
    # Process the LLM's text as a command
    grid.process_command(response)
    print("\nGrid after command #1:")
    grid.display()

    # Prompt #2: show the updated grid in text, then ask for the next command
    # We’ll create a textual representation of the grid to feed back.
    current_grid = []
    for y in range(grid.height):
        row_str = "".join('#' if pixel else '.' for pixel in grid.grid[y])
        current_grid.append(row_str)
    grid_text = "\n".join(current_grid)

    prompt2 = (
            system_context
            + "\n\nCurrent grid state:\n"
            + grid_text
            + "\n\nPlease issue your next command."
    )
    response2 = query_ollama(prompt2, model="gemma-long")

    print("\n----- LLM Response #2 -----")
    print(response2)
    grid.process_command(response2)
    print("\nGrid after command #2:")
    grid.display()

    # Continue similarly for more commands...

    print("\nDone with demonstration!")
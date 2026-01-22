# Agent Development Guidelines

- **Language Policy**: Perform coding, comments, and README.md edits in **English**, but always report results in **Korean**.
- **Python Version & Style**:
  - **Target Python 3.11+**: Ensure the code is compatible with Python 3.11 or higher. Utilize modern APIs and syntax available in 3.11+ (e.g., improved `asyncio`, `TaskGroup`, `Self` type) to ensure the code is concise, readable, and efficient.
  - **Formatting & Sorting**: Use **Black** format for code styling and ensure imports are sorted consistently, following **isort** conventions.
  - Write Python docstrings in **Google style**.
    - Do **not** include type hints inside docstrings.
    - Put type hints **directly in parameter definitions**.
    - Omit the `Args` section when all arguments are trivial or self-explanatory.
- **Maintenance & Quality**:
  - When modifying code, review file header comments and update them if needed.
  - After edits, review comments in modified code and improve them as needed for clarity and readability.
  - Understand the overall logic of the plugin flow deeply; **avoid adding unnecessary defensive code** that might clutter the implementation.
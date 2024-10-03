# ANAI

ANAI is a Python package for interacting with the Anthropic API.

## Setup

1. Make sure you have [rye](https://rye-up.com/) installed.

2. Clone this repository and navigate to the project directory.

3. Run the following command to install dependencies:

   ```
   rye sync
   ```

4. Set up your environment variables:

   Create a `.env` file in the root directory of the project with the following content:

   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
   ```

   Replace `your_api_key_here` with your actual Anthropic API key.

## Running Tests

To run the tests, use pytest:

```
rye run pytest
```

To ensure the API is working, run the following command:

```
rye run pytest --live_api
```

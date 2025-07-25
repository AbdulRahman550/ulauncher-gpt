# [ULauncher](https://ulauncher.io/) Chat GPT Extension

> Disclaimer: Not affiliated with OpenAI

This extension enables you to use ChatGPT custom system prompts to create your own custom assistant

![Screen shot](images/screenshot.png)

Feel free to fork for a more domain specific assistant

## Configuration

This extension supports both OpenAI and Gemini APIs. You can select your preferred provider in the extension settings.

-   **API Provider**: Choose between "OpenAI" and "Gemini".
-   **OpenAI API Key**: Your API key for the OpenAI API.
-   **Gemini API Key**: Your API key for the Gemini API.
-   **OpenAI API Base URL**: The base URL for the OpenAI API. Defaults to `https://api.openai.com/v1`.
-   **OpenAI Model**: The model to use for OpenAI (e.g., `gpt-3.5-turbo`).
-   **Gemini Model**: The model to use for Gemini (e.g., `gemini-1.5-flash`).
-   **Terminal Prefix**: The prefix to trigger the terminal prompt (e.g., `@`).
-   **Terminal Prompt**: The system prompt to use for terminal commands.
-   **Assistant Prompt**: The system prompt to use for general assistant queries.

## Install

- Open **Ulauncher**
- Click on the **cog wheel** to open your preferences
- Click on the **EXTENSIONS** tab
- Click on **Add extension**
- Paste this repository's URL

## Troubleshooting

You can view Ulauncher logs by running `ulauncher --verbose` from your terminal. This is useful for debugging any issues with the extension.

import logging
from ulauncher.api.client.Extension import Extension
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.shared.event import KeywordQueryEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.action.CopyToClipboardAction import CopyToClipboardAction
from ulauncher.api.shared.action.DoNothingAction import DoNothingAction
import requests
import json

logger = logging.getLogger(__name__)
EXTENSION_ICON = 'images/icon.png'


def wrap_text(text, max_w):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        if len(current_line + word) <= max_w:
            current_line += ' ' + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    return '\n'.join(lines)


class GPTExtension(Extension):
    """
    Ulauncher extension to generate text using GPT-3
    """

    def __init__(self):
        super(GPTExtension, self).__init__()
        logger.info('GPT-3 extension started')
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener())


class KeywordQueryEventListener(EventListener):
    """
    Event listener for KeywordQueryEvent
    """

    def on_event(self, event, extension):
        logger.info('Processing user preferences')
        # Get user preferences
        try:
            api_provider = extension.preferences['api_provider']
            openai_api_key = extension.preferences['api_key']
            gemini_api_key = extension.preferences['gemini_api_key']
            base_url = extension.preferences['base_url']
            max_tokens = int(extension.preferences['max_tokens'])
            frequency_penalty = float(
                extension.preferences['frequency_penalty'])
            presence_penalty = float(
                extension.preferences['presence_penalty'])
            temperature = float(extension.preferences['temperature'])
            top_p = float(extension.preferences['top_p'])
            system_prompt = extension.preferences['system_prompt']
            line_wrap = int(extension.preferences['line_wrap'])
            model = extension.preferences['model']
        # pylint: disable=broad-except
        except Exception as err:
            logger.error('Failed to parse preferences: %s', str(err))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Failed to parse preferences: ' +
                                    str(err),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])

        # Get search term
        search_term = event.get_argument()
        logger.info('The search term is: %s', search_term)
        # Display blank prompt if user hasn't typed anything
        if not search_term:
            logger.info('Displaying blank prompt')
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Type in a prompt...',
                                    on_enter=DoNothingAction())
            ])

        if api_provider == 'OpenAI':
            return self.handle_openai_request(
                search_term, openai_api_key, base_url, model, system_prompt,
                temperature, max_tokens, top_p, frequency_penalty,
                presence_penalty, line_wrap)
        elif api_provider == 'Gemini':
            return self.handle_gemini_request(
                search_term, gemini_api_key, model, system_prompt,
                temperature, max_tokens, top_p, frequency_penalty,
                presence_penalty, line_wrap)

    def handle_openai_request(self, search_term, api_key, base_url, model,
                              system_prompt, temperature, max_tokens, top_p,
                              frequency_penalty, presence_penalty, line_wrap):
        headers = {
            'content-type': 'application/json',
            'Authorization': 'Bearer ' + api_key
        }
        body = {
            "messages": [{"role": "system","content": system_prompt},
                         {"role": "user", "content": search_term}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "model": model,
        }
        body = json.dumps(body)

        try:
            response = requests.post(
                f'{base_url}/chat/completions', headers=headers, data=body, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            choices = response_json['choices']
            items = []
            for choice in choices:
                message = choice['message']['content']
                message = wrap_text(message, line_wrap)
                items.append(ExtensionResultItem(icon=EXTENSION_ICON, name="Assistant", description=message,
                                                 on_enter=CopyToClipboardAction(message)))
            return RenderResultListAction(items)
        except requests.exceptions.RequestException as err:
            logger.error('Request failed: %s', str(err))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Request failed: ' + str(err),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])
        except Exception as err:
            logger.error('Failed to parse response: %s', str(err))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Failed to parse response: ' + str(err),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])

    def handle_gemini_request(self, search_term, api_key, model, system_prompt,
                              temperature, max_tokens, top_p,
                              frequency_penalty, presence_penalty, line_wrap):
        headers = {
            'content-type': 'application/json',
            'x-goog-api-key': api_key
        }
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": search_term}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            }
        }
        body = json.dumps(body)

        try:
            response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                headers=headers, data=body, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            items = []
            for candidate in response_json['candidates']:
                message = candidate['content']['parts'][0]['text']
                message = wrap_text(message, line_wrap)
                items.append(ExtensionResultItem(icon=EXTENSION_ICON, name="Assistant", description=message,
                                                 on_enter=CopyToClipboardAction(message)))
            return RenderResultListAction(items)
        except requests.exceptions.RequestException as err:
            logger.error('Request failed: %s', str(err))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Request failed: ' + str(err),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])
        except Exception as err:
            logger.error('Failed to parse response: %s', str(err))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Failed to parse response: ' + str(err),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])
        # pylint: disable=broad-except
        except Exception as err:
            logger.error('Failed to parse response: %s', str(response))
            return RenderResultListAction([
                ExtensionResultItem(icon=EXTENSION_ICON,
                                    name='Failed to parse response: ' +
                                    str(response),
                                    on_enter=CopyToClipboardAction(str(err)))
            ])

        try:
            item_string = ' | '.join([item.description for item in items])
            logger.info("Results: %s", item_string)
        except Exception as err:
            logger.error('Failed to log results: %s', str(err))
            logger.error('Results: %s', str(items))

        return RenderResultListAction(items)


if __name__ == '__main__':
    GPTExtension().run()

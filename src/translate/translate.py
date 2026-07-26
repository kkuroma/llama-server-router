import csv
from pathlib import Path

LANGUAGES_CSV = Path(__file__).parent / "languages.csv"
PROMPT_TEMPLATE = Path(__file__).parent / "prompt.txt"


class TranslationService:
    """
    Loads the language table and prompt template, and builds cache-friendly
    chat messages for translation requests
    """

    def __init__(self):
        self.languages: list[dict[str, str]] = []
        self.lang_map: dict[str, str] = {}
        self.prompt_template: str = ""
        self._load()

    def _load(self):
        """
        Populates the language list/map from the CSV and reads the prompt template

        Skips CSV rows with fewer than two columns and keys the map by language id
        """
        with open(LANGUAGES_CSV, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                lang_id = row[0].strip()
                lang_name = row[1].strip()
                self.languages.append({
                    "lang_id": lang_id,
                    "lang_name": lang_name,
                })
                self.lang_map[lang_id] = lang_name
        self.prompt_template = PROMPT_TEMPLATE.read_text()

    def getLanguages(self, lang: str | None = None) -> list[dict[str, str]]:
        """
        Returns the language table, optionally filtered by display name

        Args:
            lang (str | None): The language name to match exactly, or None for all

        Returns:
            The list of {lang_id, lang_name} entries that match
        """
        if lang:
            return [l for l in self.languages if l["lang_name"] == lang]
        return self.languages

    def buildMessages(
        self,
        source_id: str,
        target_id: str,
        text: str,
        additionals: str = "",
    ) -> list[dict[str, str]]:
        """
        Builds a 3-chunk message array optimized for cache_prompt reuse

        Chunk 1 (system) is the stable instruction from prompt.txt, chunk 2
        (system) holds any extra user requests, and chunk 3 (user) is the text

        Args:
            source_id (str)     : The source language id, must be in lang_map
            target_id (str)     : The target language id, must be in lang_map
            text (str)          : The text to translate
            additionals (str)   : Extra instructions appended as a system message

        Returns:
            The list of chat messages ready to send to the model
        """
        source_name = self.lang_map[source_id]
        target_name = self.lang_map[target_id]

        system_prompt = (
            self.prompt_template
            .replace("{SOURCE_LANG}", source_name)
            .replace("{SOURCE_CODE}", source_id)
            .replace("{TARGET_LANG}", target_name)
            .replace("{TARGET_CODE}", target_id)
            .replace("{TEXT}", "")
            .rstrip()
        )

        messages = [{"role": "system", "content": system_prompt}]
        if additionals and additionals.strip():
            messages.append({"role": "system", "content": additionals.strip()})
        messages.append({"role": "user", "content": text})
        return messages

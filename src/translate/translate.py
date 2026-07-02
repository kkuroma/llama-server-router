import csv
from pathlib import Path

LANGUAGES_CSV = Path(__file__).parent / "languages.csv"
PROMPT_TEMPLATE = Path(__file__).parent / "prompt.txt"


class TranslationService:
    def __init__(self):
        self.languages: list[dict] = []
        self.lang_map: dict[str, str] = {}
        self.prompt_template: str = ""
        self._load()

    def _load(self):
        with open(LANGUAGES_CSV, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                lang_id = row[0].strip()
                lang_name = row[1].strip()
                self.languages.append({"lang_id": lang_id, "lang_name": lang_name})
                self.lang_map[lang_id] = lang_name
        self.prompt_template = PROMPT_TEMPLATE.read_text()

    def get_languages(self, lang: str | None = None) -> list[dict]:
        if lang:
            return [l for l in self.languages if l["lang_name"] == lang]
        return self.languages

    def build_messages(
        self,
        source_id: str,
        target_id: str,
        text: str,
        additionals: str = "",
    ) -> list[dict]:
        """Build message array with 3 chunks for cache_prompt optimization.

        Chunk 1 (system): stable translation instruction from prompt.txt
        Chunk 2 (system): additional user requests (if any)
        Chunk 3 (user):   the actual text to translate
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

"""OpenAI 호환 chat 클라이언트 — 의존성은 stdlib 만.

왜 새로 짜는가: 렌더러는 env `vista4d` 에 있고 `openai`/`any_llm` 은 `lbm`/`GenDoP` 에만 있다.
루프는 렌더↔VLM 을 10회 이상 교차하므로 env 를 오갈 수 없고, 설치는 과거에 거부된 적이 있다.
`/v1/chat/completions` 는 그냥 JSON POST 라 `urllib.request` 로 충분하다.

JSON 복구 4단계는 LBM `Director/director_engine_llm.py:187-260` 의 **로직만** 이식했다
(`_strip_code_fence` → `_extract_first_json_object` → `_parse_json_candidates` →
`_repair_json_response`). 원본 `call_json_response`(:281)는 `reasoning_effort=` 를 무조건
넘겨서 그걸 모르는 모델이 400 을 내므로 안 쓴다.

**스키마 위반에 temperature 를 올리지 않는다.** 규칙을 어긴 건 샘플링 문제가 아니라 이해 문제라
같은 온도로 `## VALIDATION_ERRORS` 를 붙여 다시 묻는 게 맞다 (최대 `max_repairs` 회).

서버 띄우기: `screen -dmS vlm bash exec/serve_qwen3vl.sh` (모델 로딩 ~3분)

예시:
    python lbm/vlm.py --ping
    python lbm/vlm.py --prompt "Reply with {\\"ok\\": true}" --image out/camel/board/board_candidates.png
"""
import base64
import json
import re
import time
import urllib.error
import urllib.request
from os import path

DEFAULT_API_BASE = "http://127.0.0.1:22002/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def image_data_url(image_path: str):
    """로컬 이미지 1장 → `data:image/png;base64,...`. LBM `image_path_to_data_url` 과 같은 포맷.

    파일 경로(`file://`)로 넘기면 서버의 `--allowed-local-media-path` 에 걸리므로 data URL 로 싣는다.
    """
    mime = MIME.get(path.splitext(image_path)[1].lower(), "image/png")
    with open(image_path, "rb") as file:
        return f"data:{mime};base64," + base64.b64encode(file.read()).decode("ascii")


def strip_code_fence(text: str):
    """```json ... ``` 한 겹을 벗긴다."""
    stripped = text.strip()
    matched = re.match(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", stripped)
    return matched.group(1).strip() if matched else stripped


def extract_first_json_object(text: str):
    """디코드 가능한 첫 JSON object/array 부분문자열. 없으면 빈 문자열."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index:index + end].strip()
    return ""


def parse_json_candidates(text: str):
    """(payload, error). raw → fence 제거 → 부분문자열 순으로 관대하게 시도한다."""
    raw = text.strip()
    candidates = [raw] if raw else []
    stripped = strip_code_fence(raw)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    extracted = extract_first_json_object(stripped or raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    error = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            error = f"invalid_json: {exc}"
            continue
        if isinstance(payload, dict):
            return payload, None
        error = f"invalid_json_type: expected object got {type(payload).__name__}"
    return None, error or "invalid_json: no_decodable_json_object_found"


class VLMClient:
    """`POST {api_base}/chat/completions` 한 개만 감싼다. 모든 턴을 `trace` 에 쌓는다."""

    def __init__(self, api_base: str = DEFAULT_API_BASE, model: str = DEFAULT_MODEL,
                 api_key: str = "EMPTY", temperature: float = 0.1, max_tokens: int = 2048,
                 timeout: float = 180.0, retries: int = 3):
        self.api_base = api_base.rstrip("/")
        self.model, self.api_key = model, api_key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout, self.retries = timeout, retries
        self.trace = []

    # ---------------------------------------------------------------- transport
    def _post(self, endpoint: str, payload: dict):
        request = urllib.request.Request(
            f"{self.api_base}{endpoint}", method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        # 지수 backoff. 서버가 로딩 중이면 connection refused 가 잠깐 난다.
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:500]
                if attempt == self.retries - 1:
                    raise RuntimeError(f"HTTP {exc.code} from {self.api_base}: {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"{self.api_base} 에 못 붙었다: {exc}. "
                                       "서버가 떠 있나? `bash exec/serve_qwen3vl.sh`") from exc
            time.sleep(2.0 * (2 ** attempt))
        raise AssertionError("unreachable")

    def models(self):
        """`GET /v1/models` — 서버가 살아 있는지 + served-model-name 확인용."""
        with urllib.request.urlopen(f"{self.api_base}/models", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    # ---------------------------------------------------------------- chat
    def chat(self, prompt: str, images=(), system: str | None = None, label: str = "",
             video: str | None = None, video_fps: float | None = None):
        """(text, meta). 이미지는 data URL 로 실린다. 텍스트가 **이미지 뒤**에 온다.

        순서를 이렇게 두는 이유: contract 텍스트가 "이 board 에서 골라라"라고 지시하는데
        지시가 먼저 오면 모델이 이미지를 보기 전에 답을 정해버리는 경향이 있다.
        """
        content = [{"type": "image_url", "image_url": {"url": image_data_url(str(image))}}
                   for image in images]
        # D280 (R41). 영상 입력 — mp4 를 data URL 로 싣고, 샘플링 fps 는 Qwen3-VL 프로세서에
        # `mm_processor_kwargs` 로 넘긴다. 안 주면 content/payload 가 예전과 글자 그대로 같다.
        if video is not None:
            with open(video, "rb") as file:
                url = "data:video/mp4;base64," + base64.b64encode(file.read()).decode("ascii")
            content.append({"type": "video_url", "video_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": content}]
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        if video is not None and video_fps is not None:
            payload["mm_processor_kwargs"] = {"fps": float(video_fps)}

        started = time.time()
        response = self._post("/chat/completions", payload)
        text = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage", {})
        meta = {"label": label, "seconds": round(time.time() - started, 2),
                "images": [str(image) for image in images],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": response["choices"][0].get("finish_reason")}
        self.trace.append({**meta, "prompt": prompt, "system": system, "raw": text})
        return text, meta

    def chat_json(self, prompt: str, images=(), system: str | None = None,
                  validate=None, max_repairs: int = 3, label: str = "",
                  echo_previous: bool = True, repair_hint: str = "",
                  hint_first: bool = False):
        """스키마를 만족하는 dict 를 얻을 때까지 되묻는다. (payload, info).

        `validate(payload) -> list[str]` 이 위반 목록을 돌려주면 그걸 그대로 붙여 재질의한다.
        `payload is None` 이면 소진된 것이고, 호출자는 결정론적 fallback 으로 내려가야 한다
        (§B6: **파이프라인은 절대 hard-fail 하지 않는다**).

        `echo_previous=False` 면 재질의에 `## PREVIOUS_RESPONSE` 를 **안 붙인다**.
        2026-08-25 실측(TRUMANS action tagging 37창): 자기 답을 다시 보여주면 Qwen3-VL 이 그걸
        **바이트 단위로 그대로 베껴** 재질의가 통째로 무효가 된다 — 3개 창이 4턴 내내 동일한
        137/158/116 토큰을 뱉고 fallback 으로 떨어졌다. temperature 를 0.7 로 올려도 2/3 은
        그대로였다(그래서 원인은 샘플링이 아니다). 자기 답을 빼고 위반만 알려주면 3/3 통과.
        기본값은 기존 동작 유지 — `loop.py` 쪽 재현성을 건드리지 않기 위함.

        `repair_hint` 는 재질의에만 덧붙는 지시. 위반이 특정 필드에 몰릴 때 그 필드를 어떻게
        고쳐야 하는지 알려주는 용도다 (위반 목록만으로는 "그럼 뭐라고 쓰냐"가 안 나온다).

        `hint_first=True` 면 `## VALIDATION_ERRORS` + `repair_hint` 를 원 프롬프트 **앞**에 둔다.
        원 프롬프트가 길어지면(씬 접지 블록 3종 = 약 600 토큰) 뒤에 붙인 힌트가 묻힌다 —
        접지를 켠 실행에서 w18 이 4턴 내내 같은 `body_facing` 위반을 반복하고 fallback 으로
        떨어졌는데, 접지 없이 같은 힌트로는 통과했던 창이다. 기본값 False = 기존 동작.
        """
        turn_prompt, repairs, errors = prompt, 0, []
        while True:
            text, meta = self.chat(turn_prompt, images=images, system=system,
                                   label=f"{label}#{repairs}")
            payload, parse_error = parse_json_candidates(text)
            violations = [parse_error] if parse_error else \
                (list(validate(payload)) if validate else [])
            if not violations:
                return payload, {"repairs": repairs, "turns": repairs + 1,
                                 "errors": errors, "last_meta": meta}
            errors.extend(violations)
            repairs += 1
            if repairs > max_repairs:
                return None, {"repairs": repairs - 1, "turns": repairs,
                              "errors": errors, "last_meta": meta,
                              "exhausted": True}
            # temperature 는 올리지 않는다 — 규칙 위반은 샘플링 문제가 아니다.
            #    parse_error 일 때는 깨진 원문 자체가 고칠 대상이라 echo_previous 와 무관하게 보여준다.
            previous = (f"\n\n## PREVIOUS_RESPONSE\n{text}"
                        if (echo_previous or parse_error) else "")
            block = ("## VALIDATION_ERRORS (your previous attempt was rejected)\n"
                     + "\n".join(f"- {v}" for v in violations)
                     + (f"\n\n{repair_hint}" if repair_hint else ""))
            tail = "\n\nReturn corrected raw JSON only. No markdown fences."
            turn_prompt = (f"{block}\n\n{prompt}{previous}{tail}" if hint_first
                           else f"{prompt}{previous}\n\n{block}{tail}")

    def save_trace(self, folder: str):
        """턴마다 `turn_<nn>.json`. prompt / 이미지 경로 / raw 응답 / 소요시간 / 토큰."""
        from os import makedirs
        makedirs(folder, exist_ok=True)
        for index, turn in enumerate(self.trace):
            with open(path.join(folder, f"turn_{index:02d}.json"), "w", encoding="utf-8") as file:
                json.dump(turn, file, ensure_ascii=False, indent=1)
        return len(self.trace)


def main(args):
    client = VLMClient(api_base=args.api_base, model=args.model, temperature=args.temperature)
    if args.ping:
        served = client.models()
        for entry in served.get("data", []):
            print(f"{'served':<12}{entry['id']}  ctx {entry.get('max_model_len')}")
        return
    images = [args.image] if args.image else []
    if args.json:
        payload, info = client.chat_json(args.prompt, images=images)
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"\n{'repairs':<12}{info['repairs']}  errors {info['errors']}")
    else:
        text, meta = client.chat(args.prompt, images=images)
        print(text)
        print(f"\n{'seconds':<12}{meta['seconds']}  "
              f"tokens {meta['prompt_tokens']}->{meta['completion_tokens']}  "
              f"finish {meta['finish_reason']}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--api_base", default=DEFAULT_API_BASE, type=str)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str)
    parser.add_argument("--prompt", default="Reply with the JSON object {\"ok\": true}.", type=str)
    parser.add_argument("--image", default=None, type=str)   # 1장만 (스모크용)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--ping", action="store_true", help="/v1/models 만 찍고 종료")
    parser.add_argument("--json", action="store_true", help="chat_json 경로로 호출")
    main(parser.parse_args())

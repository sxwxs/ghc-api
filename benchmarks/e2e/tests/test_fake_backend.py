import json

from benchmarks.e2e.fake_backend.app import app


def _events(response):
    events = []
    for line in response.get_data(as_text=True).splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_models_advertise_only_synthetic_models():
    client = app.test_client()
    data = client.get("/models").get_json()
    assert [model["id"] for model in data["data"]] == ["fake-opus", "fake-gpt"]
    assert "/v1/messages" in data["data"][0]["supported_endpoints"]
    assert "/responses" in data["data"][1]["supported_endpoints"]


def test_opus_stream_has_thinking_text_tool_and_usage():
    client = app.test_client()
    response = client.post("/v1/messages", json={
        "model": "fake-opus",
        "stream": True,
        "metadata": {"ghc_benchmark": {"profile": "full", "text_bytes": 128, "text_chunks": 4}},
    })
    events = _events(response)
    event_types = [event["type"] for event in events]
    block_types = [event["content_block"]["type"] for event in events if event["type"] == "content_block_start"]
    assert event_types[0] == "message_start"
    assert event_types[-2:] == ["message_delta", "message_stop"]
    assert block_types == ["thinking", "text", "tool_use"]
    assert events[-2]["usage"]["cache_read_input_tokens"] > 0
    assert "copilot_usage" in events[-2]


def test_gpt_stream_has_full_responses_event_family():
    client = app.test_client()
    response = client.post("/v1/responses", json={
        "model": "fake-gpt",
        "stream": True,
        "metadata": {"ghc_benchmark": {"profile": "full", "text_bytes": 128, "text_chunks": 4}},
    })
    events = _events(response)
    event_types = {event["type"] for event in events}
    assert {
        "response.created",
        "response.in_progress",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.web_search_call.completed",
        "response.output_text.annotation.added",
        "response.completed",
    } <= event_types
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["usage"]["input_tokens_details"]["cached_tokens"] > 0
    assert completed["copilot_usage"]["total_nano_aiu"] == 0


def test_non_streaming_responses_are_protocol_objects():
    client = app.test_client()
    opus = client.post("/v1/messages", json={"model": "fake-opus", "stream": False}).get_json()
    gpt = client.post("/v1/responses", json={"model": "fake-gpt", "stream": False}).get_json()
    assert opus["type"] == "message"
    assert {part["type"] for part in opus["content"]} == {"thinking", "text", "tool_use"}
    assert gpt["object"] == "response"
    assert gpt["status"] == "completed"
    assert {item["type"] for item in gpt["output"]} >= {"reasoning", "function_call", "message"}


def test_backend_never_echoes_request_content_or_real_urls():
    secret = "PRIVATE_DUMP_SENTINEL_7f993f0a"
    client = app.test_client()
    for path, model, body_field in [
        ("/v1/messages", "fake-opus", "messages"),
        ("/v1/responses", "fake-gpt", "input"),
    ]:
        response = client.post(path, json={
            "model": model,
            "stream": True,
            body_field: [{"role": "user", "content": secret}],
            "metadata": {"ghc_benchmark": {"profile": "full"}},
        })
        body = response.get_data(as_text=True)
        assert secret not in body
        assert "github.com" not in body
        assert "Authorization" not in body
        if "https://" in body:
            assert "https://perf-fixture.invalid/" in body

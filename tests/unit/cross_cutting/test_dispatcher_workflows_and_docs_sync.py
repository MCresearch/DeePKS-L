"""Tests for workflow-based dispatcher routing and docs sync."""

from pathlib import Path


def test_dispatcher_routes_test_through_workflow_wrapper(monkeypatch):
    from deepks.config.dispatcher import dispatch_command

    seen = {}

    def fake_run_test_workflow(config):
        seen['config'] = config
        return {'ok': 'test'}

    monkeypatch.setattr('deepks.workflows.test.run_test_workflow', fake_run_test_workflow)

    runtime_config = {
        'type': 'test',
        'test_param': {
            'type': 'test',
            'runtime': {'device': 'cpu', 'verbose': 2},
            'data': {'test': ['sys-a']},
            'ml': {'checkpoint': {'file': 'model.pth'}},
        },
    }

    result = dispatch_command(runtime_config)

    assert result == {'ok': 'test'}
    assert seen['config']['data']['test'] == ['sys-a']
    assert seen['config']['runtime']['verbose'] == 2
    assert seen['config']['type'] == 'test'


def test_dispatcher_routes_stats_through_workflow_wrapper(monkeypatch):
    from deepks.config.dispatcher import dispatch_command

    seen = {}

    def fake_run_stats_workflow(config):
        seen['config'] = config
        return {'ok': 'stats'}

    monkeypatch.setattr('deepks.workflows.stats.run_stats_workflow', fake_run_stats_workflow)

    runtime_config = {
        'type': 'stats',
        'stats_param': {
            'type': 'stats',
            'verbose': 1,
            'systems': ['sys-a'],
            'dump_dir': 'dump',
        },
    }

    result = dispatch_command(runtime_config)

    assert result == {'ok': 'stats'}
    assert seen['config']['systems'] == ['sys-a']
    assert seen['config']['dump_dir'] == 'dump'
    assert seen['config']['verbose'] == 1
    assert seen['config']['type'] == 'stats'


def test_sync_input_parameter_docs_writes_rendered_content(tmp_path):
    from deepks.config import render_input_parameter_doc
    from deepks.tools.sync_input_parameter_docs import sync_input_parameter_docs

    output = tmp_path / 'input-parameter.md'
    expected = render_input_parameter_doc()

    written_path = sync_input_parameter_docs(str(output))

    assert written_path == str(output)
    assert output.read_text(encoding='utf-8') == expected
    assert output.read_text(encoding='utf-8').startswith('# DeePKS input parameter reference\n')
    assert '| `runtime` | `dict` | `all` |' in expected
    assert '| `physics.representation` | `string | dict` | `train, test, iterate` |' in expected


def test_configure_line_buffered_stdio_uses_safe_reconfigure(monkeypatch):
    from deepks.main import _configure_line_buffered_stdio

    calls = []

    class DummyStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

        def fileno(self):
            raise AssertionError("fileno() should not be used")

    monkeypatch.setattr('sys.stdout', DummyStream())
    monkeypatch.setattr('sys.stderr', DummyStream())

    _configure_line_buffered_stdio()

    assert calls == [{'line_buffering': True}, {'line_buffering': True}]


def test_main_returns_zero_on_success(monkeypatch, tmp_path):
    import sys
    from deepks.main import main

    config = tmp_path / 'input.yaml'
    config.write_text('type: test\nsystems_test:\n  - sys1\nmodel_file: model.pth\n', encoding='utf-8')

    monkeypatch.setattr('deepks.config.load_runtime_config', lambda path: {'type': 'test'})
    monkeypatch.setattr('deepks.config.dispatcher.dispatch_command', lambda runtime: {'ok': True})

    argv = sys.argv
    try:
        sys.argv = ['deepks', str(config)]
        assert main() == 0
    finally:
        sys.argv = argv

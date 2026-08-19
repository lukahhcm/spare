import copy

from lagent.actions import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from agentlego.parsers import DefaultParser
from agentlego.types import AudioIO, File, ImageIO
from ..base import BaseTool


def convert_type(t):
    if t in [str, ImageIO, AudioIO, File]:
        return 'STRING'
    elif t is int:
        return 'NUMBER'
    elif t is float:
        return 'FLOAT'
    elif t is bool:
        return 'BOOLEAN'
    return 'STRING'


class LagentTool(BaseAction):
    """A wrapper to align with the interface of Lagent tools."""

    def __init__(self, tool: BaseTool):
        tool = copy.copy(tool)
        tool.set_parser(DefaultParser)  # Use string input & output
        self.tool = tool

        parameters = []
        required = []
        for p in tool.inputs:
            parameters.append(
                dict(
                    name=p.name,
                    description=p.description,
                    type=convert_type(p.type),
                ))
            if not p.optional:
                required.append(p.name)

        self._is_toolkit = False
        super().__init__(
            description=dict(
                name=tool.name,
                description=tool.toolmeta.description,
                parameters=parameters,
                required=required,
            ),
            enable=True,
        )
        # Keep the benchmark tool's native name.  The generic wrapper class
        # name (``LagentTool``) would make every remote action collide and be
        # reported as InvalidAction by the official GTA runner.
        self._name = tool.name

    def run(self, *args, **kwargs) -> ActionReturn:
        if args and not kwargs:
            import json
            command = args[0]
            if isinstance(command, str):
                # ReAct's native protocol appends the previous tool response
                # as ``Response:...``.  That suffix is protocol text, not a
                # tool argument.  Remove it before applying the benchmark's
                # per-tool schema.
                raw = command.strip()
                if '\nResponse:' in raw:
                    raw = raw.split('\nResponse:', 1)[0].rstrip()
                try:
                    parsed = json.loads(raw)
                    kwargs = parsed if isinstance(parsed, dict) else {'text': parsed}
                except Exception:
                    kwargs = {'text': raw}

        # Some model generations wrap the schema payload once more as
        # {"text": "{...json...}"}.  Unwrap only when the inner object is
        # valid JSON and its keys match this tool's declared inputs.  This is
        # a lossless format repair; it never invents missing values.
        import json
        if set(kwargs) == {'text'} and isinstance(kwargs.get('text'), str):
            nested = kwargs['text'].strip()
            if '\nResponse:' in nested:
                nested = nested.split('\nResponse:', 1)[0].rstrip()
            try:
                parsed = json.loads(nested)
            except Exception:
                parsed = None
            input_names = {p.name for p in self.tool.inputs}
            if isinstance(parsed, dict) and set(parsed).issubset(input_names):
                kwargs = parsed

        # For scalar tools (Calculator/Solver/Plot), the native argument name
        # is benchmark-defined rather than the generic ``text`` alias.
        if set(kwargs) == {'text'}:
            scalar_inputs = [p.name for p in self.tool.inputs]
            if len(scalar_inputs) == 1:
                kwargs = {scalar_inputs[0]: kwargs['text']}

        try:
            outputs = self.tool(**kwargs)
            results = []

            if not isinstance(outputs, tuple):
                outputs = [outputs]

            for out, p in zip(outputs, self.tool.outputs):
                if p.type is ImageIO:
                    results.append(dict(type='image', content=out))
                elif p.type is AudioIO:
                    results.append(dict(type='audio', content=out))
                elif p.type is File:
                    results.append(dict(type='file', content=out))
                else:
                    results.append(dict(type='text', content=str(out)))

            return ActionReturn(
                type=self.name,
                args=kwargs,
                result=results,
            )
        except Exception as e:
            return ActionReturn(
                type=self.name,
                errmsg=repr(e),
                args=kwargs,
                state=ActionStatusCode.API_ERROR,
            )

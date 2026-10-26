from copy import deepcopy
import types
import json
import ast
import base64
import math
import mimetypes
import re
from typing import List, Tuple

from mmengine.registry import Registry

REGISTRY = Registry('helper')

try:
    import lagent
    import agentlego
except ImportError:
    lagent = None
    agentlego = None


def _patch_lagent_action_executor_split():
    """Patch lagent ActionExecutor parsing to be resilient to tool names containing dots.

    Upstream lagent parses toolkit calls like "<toolkit>.<api>" via `name.split('.')`,
    which crashes when `<toolkit>` itself contains dots (e.g. "Solver.v2.solve").

    Strategy:
      1) If `name` matches an action name exactly, treat it as a normal action (api='run').
      2) Otherwise, if it contains dots, split from the right once: `rsplit('.', 1)`.

    This keeps backward compatibility while preventing `ValueError: too many values to unpack`.
    """
    if lagent is None:
        return

    try:
        from lagent.actions.action_executor import ActionExecutor as _ActionExecutor
        from lagent.schema import ActionValidCode
    except Exception:
        return

    if getattr(_ActionExecutor, '_opencompass_safe_split_patched', False):
        return

    def _safe_call(self, name: str, command: str):
        if name in getattr(self, 'actions', {}):
            action_name, api_name = name, 'run'
        elif '.' in name:
            action_name, api_name = name.rsplit('.', 1)
        else:
            action_name, api_name = name, 'run'

        if not self.is_valid(action_name):
            if name == self.no_action.name:
                action_return = self.no_action(command)
            elif name == self.finish_action.name:
                action_return = self.finish_action(command)
            else:
                action_return = self.invalid_action(command)
        else:
            # Current lagent executes BaseAction through ``run``.  Calling the
            # object itself (the old API) silently turns valid AgentLego tools
            # into InvalidAction; preserve the native command string here.
            # LagentTool expects schema fields as keyword arguments; passing
            # a dict positionally silently drops them and yields empty HTTP
            # bodies for RemoteTool calls.
            if isinstance(command, dict):
                action_return = self.actions[action_name].run(**command)
            else:
                action_return = self.actions[action_name].run(command)
            action_return.valid = ActionValidCode.OPEN
        return action_return

    _ActionExecutor.__call__ = _safe_call
    setattr(_ActionExecutor, '_opencompass_safe_split_patched', True)

class DummyTool(agentlego.tools.BaseTool):

    def __init__(self, toolmeta):
        self.toolmeta = agentlego.schema.ToolMeta.from_json_dict(toolmeta)
        self.set_parser(agentlego.parsers.DefaultParser)
        self._is_setup = False

    def apply(self, *args, **kwargs):
        return 'Dummy Result'


class SummaryTheTask(agentlego.tools.BaseTool):
    """Native V3 memory action; it returns the model-provided summary verbatim."""
    default_desc = ('Summarize the current task, verified tool observations, '
                    'uncertainty, and the next action. This is a memory action, not a final answer.')

    def apply(self, content: str) -> str:
        return content

def dummy_action_executor(tools):
    return lagent.ActionExecutor(
        actions=[DummyTool(cfg).to_lagent() for cfg in tools])


def model_adapter(model):
    """Modify the generate method to accept and return single item."""
    if getattr(model, '_generate_is_wrapped', False):
        # Avoid wrap twice.
        return model

    from opencompass.utils import PromptList

    def chat(self, inputs, *args, **kwargs):
        prompt = PromptList()
        for item in inputs:
            msg = {'prompt': item['content']}
            if item['role'] == 'user':
                msg['role'] = 'HUMAN'
            elif item['role'] == 'assistant':
                msg['role'] = 'BOT'
            elif item['role'] == 'system':
                msg['role'] = 'SYSTEM'
            prompt.append(msg)
        return self.generate([prompt], *args, **kwargs)[0]

    model.chat = types.MethodType(chat, model)
    setattr(model, '_generate_is_wrapped', True)
    return model


def react_style_history(history, files, protocol) -> List[dict]:
    from lagent.schema import ActionReturn
    inner_steps = []
    if files:
        image_parts = []
        text_paths = []
        for file in files:
            path = str(file.get('path', '')) if isinstance(file, dict) else str(file)
            mime, _ = mimetypes.guess_type(path)
            if mime and mime.startswith('image/'):
                try:
                    with open(path, 'rb') as fp:
                        raw = base64.b64encode(fp.read()).decode('ascii')
                    image_parts.append({'type': 'image_url',
                                        'image_url': {'url': f'data:{mime};base64,{raw}'}})
                except OSError:
                    text_paths.append(path)
            else:
                text_paths.append(path)
        text = 'The related files are at ' + ', '.join(f'`{p}`' for p in text_paths)
        if image_parts:
            image_parts.insert(0, {'type': 'text', 'text': text +
                                   '\nThe image files above are the visual inputs for this task.'})
            inner_steps.append(dict(role='user', content=image_parts))
        elif text_paths:
            inner_steps.append(dict(role='system', content=text))
    for step in history:
        if not isinstance(step, dict):
            continue
        if step['role'] == 'user':
            inner_steps.append(dict(role='user', content=step['content']))
        elif step['role'] == 'assistant' and step.get('tool_calls'):
            name = step['tool_calls'][0]['function']['name']
            args = step['tool_calls'][0]['function']['arguments']
            context_reasoning = str(step.get('context_reasoning') or '').strip()
            reasoning_prefix = (
                f'<think>\n{context_reasoning}\n</think>\n'
                if context_reasoning else ''
            )
            response = reasoning_prefix + "{action}{name}\n{action_input}{args}".format(
                action=protocol.action['begin'],
                name=name,
                action_input=protocol.action_input['begin'],
                args=json.dumps(args),
            )
            inner_steps.append(dict(role='assistant', content=response))
        elif step['role'] == 'tool' and step.get('content'):
            # action= ActionReturn(result=[dict(type='text', content=step['content'])])
            inner_steps.append(dict(role='system', content=str(step['content'])))
        elif step['role'] == 'assistant' and step.get('content'):
            inner_steps.append(dict(role='assistant', content=step['content']))
    return inner_steps


def _clean_action_input(name, value):
    """Remove common model-generated prose contamination before execution."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                value = parsed
        except Exception:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    value = parsed
            except Exception:
                pass
        if isinstance(value, str):
            # Lagent's parser may return the raw Action Input string rather
            # than a mapping. Convert it to the tool's expected single field.
            if name == 'Calculator':
                m = re.search(r'expression\s*=\s*(.*)', value, re.S | re.I)
                expr = (m.group(1) if m else value).splitlines()[0].strip().strip('`')
                expr = re.sub(r'[$€£¥]\s*(?=\d)', '', expr)
                return {'expression': expr}
            if name == 'OCR':
                m = re.search(r'(/[^\s`\"\']+\.(?:png|jpe?g|webp))', value, re.I)
                return {'image': m.group(1) if m else value.strip()}
            if name == 'DrawBox':
                value = {'text': value}
    if not isinstance(value, dict):
        return value
    value = dict(value)
    if name == 'OCR' and isinstance(value.get('image'), str):
        m = re.search(r'(/[^\s`\"\']+\.(?:png|jpe?g|webp))', value['image'], re.I)
        if m:
            value['image'] = m.group(1)
    if name == 'Calculator' and isinstance(value.get('expression'), str):
        expr = value['expression']
        try:
            nested = json.loads(expr)
            if isinstance(nested, dict) and 'expression' in nested:
                expr = str(nested['expression'])
        except Exception:
            pass
        m = re.search(r'expression\s*=\s*([^\n`]+)', expr, re.I)
        if m:
            expr = m.group(1)
        expr = expr.splitlines()[0].strip().strip('`')
        # Nemotron sometimes writes arithmetic as ``2*$104``.  Removing a
        # currency glyph immediately before a numeric literal is a
        # deterministic syntax repair; it does not change the quantity.
        value['expression'] = re.sub(r'[$€£¥]\s*(?=\d)', '', expr)
    if name == 'DrawBox' and isinstance(value.get('text'), str):
        raw = value['text']
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            try:
                nested = json.loads(m.group(0))
                if isinstance(nested, dict):
                    value = nested
            except Exception:
                pass
        if set(value) == {'text'}:
            # Recover the native GTA DrawBox triplet from the common ReAct
            # form ``Action Input: image, bbox, annotation``.  Do not guess
            # any missing field.
            triplet = re.search(
                r'(?:Action Input:\s*)?'
                r'(/[^\n,`"\']+\.(?:png|jpe?g|webp))\s*,\s*'
                r'(\([^)\n]+\)|\[[^\]\n]+\])'
                r'(?:\s*,\s*([^\n]+))?',
                raw,
                re.I,
            )
            if triplet:
                value = {
                    'image': triplet.group(1),
                    'bbox': triplet.group(2),
                }
                annotation = str(triplet.group(3) or '').strip()
                if annotation:
                    value['annotation'] = annotation
    return value


def _recover_embedded_action(name, value, valid_names):
    """Recover an unambiguous JSON tool envelope emitted as InvalidAction."""
    if name != 'InvalidAction' or not isinstance(value, str):
        return name, value
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', value):
        try:
            payload, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        tool_name = str(payload.get('tool') or payload.get('name') or '')
        if tool_name not in valid_names:
            continue
        arguments = payload.get('arguments')
        if arguments is None:
            arguments = {
                key: item for key, item in payload.items()
                if key not in {'tool', 'name'}
            }
        if isinstance(arguments, dict):
            return tool_name, arguments
    return name, value


def _recover_nemotron_final(response,
                            thought,
                            action_input=None,
                            force_final=False):
    """Return a final answer only when Nemotron's intent is unambiguous."""
    candidates = [
        str(thought or ''),
        str(response or ''),
        str(action_input or ''),
    ]
    for text in candidates:
        match = re.search(r'(?:^|\n)\s*Response\s*:\s*(.+)', text, re.I | re.S)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'</?finalanswer>\s*', '', answer, flags=re.I)
            if answer:
                return answer.strip()
    if force_final:
        text = str(thought or response or '').strip()
        text = re.sub(r'^\s*Thought\s*:\s*', '', text, flags=re.I)
        if text and not re.search(r'(?:^|\n)\s*Action\s*:', text, re.I):
            return text
    return ''


def _v5_prunable_reasoning_segment_count(history):
    """Count persisted reasoning segments that V5 could summarize.

    V5's warm-up gate is about having prior trace content to compress, not
    about whether an external action succeeded (or was external at all).
    The hidden summary controller is therefore exposed after one completed
    assistant action step with non-empty reasoning.
    """
    count = 0
    for step in history:
        if not isinstance(step, dict) or step.get('role') != 'assistant':
            continue
        if (step.get('tool_calls')
                and str(step.get('context_reasoning') or '').strip()):
            count += 1
    return count


def _v5_unprocessed_reasoning_segment_count(history):
    """Count action reasoning not yet considered by a V5 summary pass."""
    return sum(
        1
        for step in history
        if (isinstance(step, dict)
            and step.get('role') == 'assistant'
            and step.get('tool_calls')
            and str(step.get('context_reasoning') or '').strip()
            and not step.get('v5_summary_processed'))
    )


def _canonical_tool_result(result):
    """Return a stable Lagent/OpenCompass observation schema.

    Remote AgentLego tools legitimately return strings, dictionaries, lists
    of content blocks, or image objects.  Older rollout code passed those
    forms through verbatim, so a single string result could later be treated
    as a content block and crash the inferencer.  Keep the executor output
    faithful while converting it to the one schema consumed by the history
    builder: a non-empty list of ``{type, content}`` dictionaries.
    """
    blocks = []

    def add(value):
        if value is None:
            return
        if isinstance(value, dict):
            kind = value.get('type')
            content = value.get('content', value.get('text', value))
            if kind in ('text', 'image'):
                blocks.append({'type': kind, 'content': str(content)})
            else:
                blocks.append({'type': 'text', 'content': json.dumps(value, ensure_ascii=False, default=str)})
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        # AgentLego ImageIO and similar objects expose a local output path.
        path = getattr(value, 'path', None) or getattr(value, 'file_path', None)
        if path:
            blocks.append({'type': 'image', 'content': str(path)})
        else:
            blocks.append({'type': 'text', 'content': str(value)})

    add(result)
    return blocks or [{'type': 'text', 'content': '[Tool returned no observation.]'}]


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """
    is_api = True

    def __init__(self,
                 agent_type,
                 llm,
                 actions=None,
                 tool_server=None,
                 tool_meta=None,
                 protocol=None,
                 prune_mode=None,
                 reasoning_output=True,
                 **kwargs):
        _patch_lagent_action_executor_split()
        raw_llm_cfg = deepcopy(llm) if isinstance(llm, dict) else {}
        built_llm = REGISTRY.build(llm)
        resolved_model_name = str(
            getattr(built_llm, 'path', '') or raw_llm_cfg.get('path') or '')
        llm = model_adapter(built_llm)
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}

        tools = {}
        if actions is not None:
            for action in actions:
                action = REGISTRY.build(action)
                if isinstance(action, agentlego.tools.BaseTool):
                    action = action.to_lagent()
                tools[action.name] = action
        if tool_server is not None:
            from agentlego.tools.remote import RemoteTool
            for tool in RemoteTool.from_server(tool_server):
                tools[tool.name] = tool.to_lagent()
        if tool_meta is not None:
            metas = json.load(open(tool_meta, 'r'))
            for meta in metas.values():
                tool = DummyTool(meta).to_lagent()
                tools.setdefault(tool.name, tool)

        self.tools = tools
        self.prune_mode = prune_mode
        self._model_name = resolved_model_name
        # GTA keeps the model's native/internal thinking disabled.  This flag
        # only controls whether the parsed, user-visible thought is preserved
        # in the runner output.
        self.reasoning_output = reasoning_output
        self._v5_hidden_summary = ''
        self._v5_final_summary_used = False
        self._v5_events = []
        self._v5_controller_events = []
        self._v5_scorer = None
        if prune_mode == 'v5':
            from reasoning_prune.openai_scorer import OpenAICompletionScorer
            endpoint = str(raw_llm_cfg.get('openai_api_base') or '')
            base_url = endpoint.removesuffix('/chat/completions').rstrip('/')
            self._v5_scorer = OpenAICompletionScorer(
                base_url=base_url,
                model=str(raw_llm_cfg.get('path') or ''),
                api_key=str(raw_llm_cfg.get('key') or 'EMPTY'),
                top_logprobs=20,
            )
        # V3 models summary as an explicit memory action. V5 does not: its
        # summary is controller-side auxiliary reasoning and must never enter
        # the model-visible tool contract or native tool trajectory.
        if prune_mode == 'v3':
            summary = SummaryTheTask()
            summary.name = 'summarize_the_task'
            lag_summary = summary.to_lagent()
            lag_summary._name = 'summarize_the_task'
            self.tools[summary.name] = lag_summary

        if protocol is not None:
            protocol = REGISTRY.build(protocol)
            agent_cfg['protocol'] = protocol

        from lagent import BaseAgent, ActionExecutor
        agent_cfg['action_executor'] = ActionExecutor(tools.values())
        self.agent: BaseAgent = REGISTRY.build(agent_cfg)

    def reset(self):
        pass

    def gt_response(self, prompt):
        if 'CIReAct' in str(self.agent.__class__):
            gold = prompt
            prompt = f"""{self.agent._protocol.action['begin']} IPythonInterpreter
{self.agent._protocol.action_input['begin']} ```python\n{gold}\n```\n"""  # noqa
            action_input = dict(
                command=f"""```python\n{gold}\n```\n""",
                timeout=120,
            )
            response = self.agent._action_executor('IPythonInterpreter',
                                                   action_input)
            gt_response = dict(role='assistant', content=prompt)
            system_response = dict(
                role='system',
                content=self.agent._protocol.format_response(response))
            return [gt_response, system_response]
        else:
            gt_response = dict(role='assistant', content=prompt)
            return [gt_response]

    @property
    def template_parser(self):
        return self.agent._llm.template_parser

    @template_parser.setter
    def template_parser(self, value):
        self.agent._llm.template_parser = value

    def next_step(self, history, resources=None, stop=False):
        from lagent.schema import ActionReturn
        tools = []
        files = []
        if resources is not None:
            tools = [
                self.tools[item['name']] for item in resources
                if item['type'] == 'tool'
            ]
            files = [item for item in resources if item['type'] == 'file']
        # GTA V3 hook: only an actual model-emitted summary marker permits
        # history compaction. No marker means the native history is untouched.
        if self.prune_mode == 'v3' and any(
                (isinstance(item, dict) and
                 ('summarize_the_task' in str(item.get('content', '')) or
                 any(isinstance(c, dict) and
                     c.get('function', {}).get('name') == 'summarize_the_task'
                     for c in item.get('tool_calls', []))))
                for item in history
                if isinstance(item, dict) and item.get('role') == 'assistant'):
            history = history[-6:]

        action_executor = lagent.ActionExecutor(actions=tools)
        summary_enabled = self.prune_mode == 'v3'
        if summary_enabled and 'summarize_the_task' in self.tools:
            tools = list(tools) + [self.tools['summarize_the_task']]
            action_executor = lagent.ActionExecutor(actions=tools)
        if stop:
            history = history + [{'role': 'user', 'content': 'Please summarize the chat history and give a final answer. Do not call any tools.'}]
        history = react_style_history(history, files, self.agent._protocol)
        tool_contract = json.dumps(action_executor.get_actions_info(), ensure_ascii=False, indent=2)
        history.insert(0, dict(role='system', content=(
            'Before every action and final answer, write concrete reasoning '
            'grounded in the user request and observations returned by real '
            'tools. For a final response, keep this reasoning separate from '
            'the answer. Do not invent observations or claim a tool ran when '
            'it did not. The available tools and their exact argument '
            'contracts are:\n' + tool_contract + '\n'
            'When calling a tool, use its exact name and JSON keyword '
            'arguments; never use Markdown code fences or an invented text '
            'argument.')))
        if self.prune_mode == 'v3':
            history.insert(1, dict(role='system', content=(
                'Only call it with a faithful summary of verified observations and next action; '
                'do not use it as a final answer.')))
        used_v5_hint = self.prune_mode == 'v5' and bool(self._v5_hidden_summary)
        if self.prune_mode == 'v5' and self._v5_hidden_summary:
            history.insert(1, dict(role='system', content=(
                'Hidden task-state summary for the next decision only. Do not '
                'quote it or treat it as a prior visible trace:\n'
                + self._v5_hidden_summary)))
        prompt = self.agent._protocol.format(chat_history=[],
                                             inner_step=history,
                                             action_executor=action_executor)
        response = self.agent._llm.chat(prompt)
        if used_v5_hint:
            # The auxiliary summary is intentionally a one-decision hint, not
            # persistent trace content.
            self._v5_hidden_summary = ''
        thought, action_name, action_input = self.agent._protocol.parse(
            response, action_executor)
        adapter_repairs = []
        if 'nemotron' in self._model_name.lower():
            valid_names = set(action_executor.action_names())
            sentinel_names = {
                action_executor.no_action.name,
                action_executor.invalid_action.name,
                action_executor.finish_action.name,
            }
            parser_invalid = (
                action_name == action_executor.invalid_action.name
                or action_name not in valid_names | sentinel_names
            )
            recovered_name, recovered_input = _recover_embedded_action(
                (action_executor.invalid_action.name
                 if parser_invalid else action_name),
                action_input,
                valid_names)
            if (recovered_name, recovered_input) != (action_name, action_input):
                if recovered_name in valid_names:
                    adapter_repairs.append('embedded_action_json')
                action_name, action_input = recovered_name, recovered_input
            elif parser_invalid:
                action_name = action_executor.invalid_action.name
            if action_name in {
                    action_executor.no_action.name,
                    action_executor.invalid_action.name,
            }:
                recovered_answer = _recover_nemotron_final(
                    response,
                    thought,
                    action_input=action_input,
                    force_final=stop)
                if recovered_answer:
                    adapter_repairs.append('response_as_final')
                    action_name = action_executor.finish_action.name
                    action_input = recovered_answer
            if action_name in {
                    action_executor.no_action.name,
                    action_executor.invalid_action.name,
            } and not stop:
                return {
                    'role': 'assistant',
                    'content': '',
                    'reasoning': str(thought or '').strip(),
                    'thought': str(thought or '').strip(),
                    '_adapter_retry': (
                        'Your previous response could not be parsed. Reply '
                        'using exactly one of these forms: `Action: <tool '
                        'name>` followed by `Action Input: <JSON object>`, or '
                        '`Final Answer: <answer>`. Do not use `Response:`.'
                    ),
                }
        cleaned_action_input = (
            action_input
            if action_name == action_executor.finish_action.name
            else _clean_action_input(action_name, action_input)
        )
        if cleaned_action_input != action_input:
            adapter_repairs.append(f'{action_name}_arguments')
        action_input = cleaned_action_input
        action: ActionReturn = action_executor(action_name, action_input)

        # No-tools control must not report parser-generated NoAction/Invalid
        # actions as tool calls.  Preserve the model's raw response and mark
        # the format failure, but expose no callable tool event to the
        # evaluator.
        if self.prune_mode == 'no_tools' and action.type != action_executor.finish_action.name:
            return dict(role='assistant', content=str(response),
                        reasoning=str(thought or '').strip(),
                        thought=str(thought or '').strip(),
                        error=dict(type='NO_TOOLS_ACTION_BLOCKED',
                                   msg='Tool calls are disabled for this control.'))

        if action.type == action_executor.finish_action.name:
            # lagent>=0.1.2 removed ActionReturn.format_result(); preserve
            # the official runner's finish semantics with its actual result.
            result = action.result
            if isinstance(result, list):
                result = '\n'.join(str(x.get('content', x)) if isinstance(x, dict) else str(x)
                                    for x in result)
            content = str(result)
            thought_text = str(thought or '').strip()
            if self.reasoning_output and thought_text:
                content = (f'<think>\n{thought_text}\n</think>\n'
                           f'<answer>\n{content}\n</answer>')
            msg = dict(role='assistant', content=content,
                       reasoning=thought_text, thought=thought_text)
            if adapter_repairs:
                msg['adapter_repairs'] = adapter_repairs
            return msg
        else:
            thought_text = str(thought or '').strip()
            msg = {'role': 'assistant'}
            if adapter_repairs:
                msg['adapter_repairs'] = adapter_repairs
            if self.reasoning_output:
                msg['reasoning'] = thought_text
                msg['thought'] = thought_text
            if self.prune_mode == 'visual_evidence_only':
                from reasoning_prune.policy import structured_evidence
                evidence = structured_evidence(thought_text)
                msg['original_reasoning'] = thought_text
                msg['reasoning'] = evidence
                msg['thought'] = evidence
                msg['context_reasoning'] = evidence
            elif self.prune_mode == 'v5':
                msg['context_reasoning'] = thought_text
            elif self.prune_mode is None:
                # The unpruned tools baseline must replay the complete
                # reasoning that preceded each tool call.  The history
                # renderer intentionally reads ``context_reasoning`` rather
                # than the audit/output-only ``reasoning`` field so pruned
                # modes can control exactly what survives in model context.
                msg['context_reasoning'] = thought_text
            # Keep the schema-cleaned arguments that were actually sent to
            # the executor; ActionReturn.args may retain the parser's raw,
            # prose-contaminated payload.
            args = action_input
            if action.errmsg is not None:
                msg['error'] = dict(type=action.state.name, msg=action.errmsg)
                # Handle fallback args
                if isinstance(args, dict):
                    args = args.get('inputs', args)
            function = dict(name=action.type, arguments=args)
            msg['tool_calls'] = [dict(type='function', function=function)]
            # The action has already been executed by ActionExecutor.  Keep
            # its real result attached so the outer multi-turn loop can feed
            # the observation back to the model without teacher forcing.
            # Always expose the same content-block shape to both the next
            # model turn and the evaluator, including failed tool calls.
            msg['tool_result'] = _canonical_tool_result(action.result)
            return msg

    def chat(self, query, memory=None, resources=None):
        tools = []
        files = []
        if resources is not None:
            tools = {
                item['name']: self.tools[item['name']]
                for item in resources if item['type'] == 'tool'
            }
            files = [item for item in resources if item['type'] == 'file']

        # Explicit no-tool baseline: the model receives the user request and
        # files, but no callable actions.  This is a separate control from
        # the normal tool-enabled baseline.
        if self.prune_mode == 'no_tools':
            tools = {}

        action_executor = self.agent._action_executor
        action_executor.actions = tools
        if memory is None:
            memory = []
            if files:
                prompt = 'The related files are at ' + ', '.join(
                    f'`{file["path"]}`' for file in files)
                memory.append(dict(role='user', content=prompt))
        memory.append(dict(role='user', content=query))

        # Use the OpenCompass-compatible chat path (next_step) instead of
        # lagent.ReAct.chat, whose generate_from_template expects a string and
        # rejects the structured multimodal history.  This also gives every
        # mode a real tool loop with real observations.
        self._v5_hidden_summary = ''
        self._v5_final_summary_used = False
        self._v5_events = []
        self._v5_controller_events = []
        adapter_retry_events = 0
        steps = []
        for turn in range(int(getattr(self.agent, 'max_turn', 10))):
            stop_now = turn >= int(getattr(self.agent, 'max_turn', 10)) - 1
            msg = self.next_step(memory, resources=resources, stop=stop_now)
            retry_prompt = (
                msg.pop('_adapter_retry', None)
                if isinstance(msg, dict) else None
            )
            if retry_prompt:
                adapter_retry_events += 1
                memory.append(dict(role='user', content=retry_prompt))
                continue
            tool_result = msg.pop('tool_result', None) if isinstance(msg, dict) else None
            tool_name = ''
            tool_args = {}
            if isinstance(msg, dict) and msg.get('tool_calls'):
                function = msg['tool_calls'][0].get('function', {})
                tool_name = str(function.get('name') or '')
                tool_args = function.get('arguments') or {}
            if (self.prune_mode == 'v5'
                    and not msg.get('tool_calls')
                    and not self._v5_final_summary_used
                    and _v5_unprocessed_reasoning_segment_count(memory) > 0):
                # Give the auxiliary controller one final autonomous trigger
                # opportunity. The tentative answer is replaced only when the
                # model elects to summarize.
                self._v5_final_summary_used = True
                summary = self._maybe_generate_v5_aux_summary(
                    memory, phase='before_final')
                if summary:
                    self._apply_v5_summary(
                        memory, summary, phase='before_final')
                    self._v5_hidden_summary = summary
                    msg = self.next_step(
                        memory, resources=resources, stop=True)
                    tool_result = (
                        msg.pop('tool_result', None)
                        if isinstance(msg, dict) else None
                    )
            steps.append(msg)
            memory.append(msg)
            if 'tool_calls' not in msg:
                break
            tool_result = _canonical_tool_result(tool_result)
            tool_text = '\n'.join(str(x['content']) for x in tool_result)
            memory.append(dict(role='tool', content=tool_text))
            steps.append(dict(role='tool', content=tool_result))
            if (self.prune_mode == 'v5'
                    and _v5_unprocessed_reasoning_segment_count(memory) > 0):
                summary = self._maybe_generate_v5_aux_summary(
                    memory, phase='after_action')
                if summary:
                    self._apply_v5_summary(
                        memory, summary, phase='after_action')
                    self._v5_hidden_summary = summary
        if self.prune_mode == 'delete_reasoning':
            # Reasoning-removal ablation: preserve actual tool calls/results
            # and final answers, but remove assistant thought-only records.
            steps = [s for s in steps if not (
                isinstance(s, dict) and s.get('meta', {}).get('type') == 'thought')]
        if self.prune_mode == 'v5' and steps:
            for step in reversed(steps):
                if isinstance(step, dict) and step.get('role') == 'assistant':
                    step['v5_summary_events'] = deepcopy(self._v5_events)
                    step['v5_summary_controller_events'] = deepcopy(
                        self._v5_controller_events)
                    break
        if adapter_retry_events and steps:
            for step in reversed(steps):
                if isinstance(step, dict) and step.get('role') == 'assistant':
                    step['adapter_format_retries'] = adapter_retry_events
                    break
        return steps, memory

    def _maybe_generate_v5_aux_summary(self, memory, phase):
        """Let the model autonomously trigger hidden summary generation."""
        state = []
        for step in memory:
            if not isinstance(step, dict):
                continue
            role = str(step.get('role') or '')
            if role == 'user':
                content = step.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(
                        str(item.get('text', ''))
                        for item in content
                        if isinstance(item, dict) and item.get('type') == 'text'
                    )
                state.append(f'User: {content}')
            elif role == 'assistant' and step.get('tool_calls'):
                reasoning = str(step.get('context_reasoning') or '').strip()
                function = step['tool_calls'][0].get('function', {})
                state.append(
                    'Assistant reasoning: ' + reasoning + '\n'
                    'Action: ' + str(function.get('name') or '') + ' '
                    + json.dumps(function.get('arguments') or {},
                                 ensure_ascii=False)
                )
            elif role == 'tool':
                state.append('Observation: ' + str(step.get('content') or ''))
        prompt = [
            {
                'role': 'system',
                'content': (
                    'Decide autonomously whether prior action reasoning should '
                    'be compressed now. Return exactly SKIP if no summary is '
                    'needed. Otherwise return SUMMARY: followed by a concise '
                    'auxiliary task-state summary preserving only verified '
                    'visual evidence, tool observations, unresolved '
                    'uncertainty, and the next decision. Do not emit or '
                    'recommend a tool call.'
                ),
            },
            {'role': 'user', 'content': '\n\n'.join(state)},
        ]
        try:
            raw = str(
                self.agent._llm.chat(prompt, max_out_len=256)
            ).strip()
            markers = list(re.finditer(
                r'(?im)^\s*(SKIP|SUMMARY\s*:)',
                raw,
            ))
            if not markers or markers[0].group(1).upper() == 'SKIP':
                self._v5_controller_events.append({
                    'phase': phase,
                    'summary_triggered': False,
                    'decision_format_valid': bool(markers),
                })
                return ''
            summary = raw[markers[0].end():].strip()
            summary = re.sub(
                r'\s*</(?:think|answer)>\s*$',
                '',
                summary,
                flags=re.I,
            ).strip()
            triggered = bool(summary)
            self._v5_controller_events.append({
                'phase': phase,
                'summary_triggered': triggered,
                'decision_format_valid': True,
            })
            return summary
        except Exception as exc:
            self._v5_controller_events.append({
                'phase': phase,
                'summary_triggered': False,
                'decision_format_valid': False,
                'error': f'{type(exc).__name__}: {exc}',
            })
            return ''

    def _apply_v5_summary(self, memory, summary, phase='unspecified'):
        """Use hidden-summary reverse KL to rewrite earlier reasoning.

        High-KL segments retain only structured visual evidence; low-KL
        segments lose all reasoning. Native action blocks and observations
        remain untouched.
        """
        from reasoning_prune.policy import structured_evidence

        # Persist only the pruning audit, never the auxiliary summary text.
        event = {
            'auxiliary_summary_generated': bool(summary),
            'hidden_from_trace': True,
            'trigger_phase': phase,
            'decisions': [],
        }
        if not summary or self._v5_scorer is None:
            event['skipped_reason'] = 'missing_summary_or_scorer'
            self._v5_events.append(event)
            return
        student_prompt = (
            'Use the system instructions and solve the task. Continue the '
            'existing native tool trajectory.\n'
        )
        teacher_prompt = (
            student_prompt + '\nSummary of prior task state:\n' + summary + '\n'
        )
        for index, step in enumerate(memory):
            if not isinstance(step, dict) or step.get('role') != 'assistant':
                continue
            reasoning = str(step.get('context_reasoning') or '').strip()
            if (not reasoning
                    or not step.get('tool_calls')
                    or step.get('v5_summary_processed')):
                continue
            try:
                student = self._v5_scorer.score(student_prompt, reasoning)
                teacher = self._v5_scorer.score(teacher_prompt, reasoning)
                if len(student) != len(teacher) or not student:
                    raise RuntimeError('incompatible_logprob_positions')
                raw = []
                for student_pos, teacher_pos in zip(student, teacher):
                    value = 0.0
                    for token, student_lp in student_pos.items():
                        probability = min(1.0, math.exp(float(student_lp)))
                        value += probability * (
                            float(student_lp)
                            - float(teacher_pos.get(token, -30.0))
                        )
                    raw.append(max(0.0, value))
                low, high = min(raw), max(raw)
                normalized = (
                    [0.0] * len(raw)
                    if high == low
                    else [(value - low) / (high - low) for value in raw]
                )
                high_count = sum(value > 0.2 for value in normalized)
                retain_visual = high_count >= 2
                replacement = (
                    structured_evidence(reasoning) if retain_visual else ''
                )
                step['context_reasoning'] = replacement
                step['reasoning'] = replacement
                step['thought'] = replacement
                step.setdefault('v5_original_reasoning', reasoning)
                step['v5_summary_processed'] = True
                event['decisions'].append({
                    'memory_index': index,
                    'high_kl_tokens': high_count,
                    'token_count': len(normalized),
                    'retain_visual_evidence': retain_visual,
                    'replacement': replacement,
                })
            except Exception as exc:
                event.setdefault('errors', []).append({
                    'memory_index': index,
                    'error': f'{type(exc).__name__}: {exc}',
                })
        self._v5_events.append(event)



FORCE_STOP_PROMPT_EN = (
    """You should directly give results based on history information."""  # noqa
)

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

Begin!
"""  # noqa

PYTHON_INTERPRETER_DESCRIPTION = """\
It can run a Python code. The code must be a valid code that contains only python method, and the method' name must be 'solution' and returns a dict, which key is variable name. The libraries I recommend are sympy and scipy. the format is:
```python
# import packages
import xxx
def solution():
    # initialize some variables
    variable_names_with_real_meaning = xxx
    # middle steps
    mid_variable = func(mid_variable)
    # final answer
    final_answer = func(mid_variable)
    return final_answer
```"""  # noqa


class CodeAgent(LagentAgent):
    """Code Agent wrapper for Lagent."""

    def __init__(self, llm, **kwargs):
        from lagent import PythonInterpreter, ReAct
        from lagent.agents.react import ReActProtocol

        agent_type = kwargs.pop('agent_type', ReAct)
        max_turn = kwargs.pop('max_turn', 3)
        actions = kwargs.pop(
            'actions',
            [
                dict(type=PythonInterpreter,
                     description=PYTHON_INTERPRETER_DESCRIPTION),
            ],
        )
        protocol = kwargs.pop(
            'protocol',
            dict(
                type=ReActProtocol,
                call_protocol=FEWSHOT_INSTRUCTION,
                force_stop=FORCE_STOP_PROMPT_EN,
                finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
            ),
        )
        super().__init__(agent_type=agent_type,
                         llm=llm,
                         actions=actions,
                         protocol=protocol,
                         max_turn=max_turn,
                         **kwargs)

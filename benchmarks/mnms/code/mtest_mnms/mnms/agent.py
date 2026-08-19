import os
import re

from autogen.agentchat import Agent, UserProxyAgent
from typing import Dict, Optional, Union

class MnmsUserAgent(UserProxyAgent):

    def __init__(
            self,
            name,
            prompt_generator, 
            feedback_generator,
            parser,
            verifier,
            executor,
            **config,
    ):
        super().__init__(name, **config)

        self.current_plan = []
        self.prompt_generator = prompt_generator
        self.feedback_generator = feedback_generator
        self.verifier = verifier
        self.parser = parser
        self.executor = executor
        self.current_task_id = None
        self.feedback_types = []
        self._v3_summary_calls = 0
        self._v3_summary_assistant_mark = 0

    def sender_hits_max_reply(self, sender: Agent):
        return self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply

    def receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])
        self._process_received_message(message, sender, silent)
        
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_error_message = parsed_results['message']
        parsed_status = parsed_results['status']
        parsed_error_code = parsed_results['error_code']

        # Terminate planning if parsing fails and termination check returns true
        # Otherwise proceed to parsing feedback or verification/execution
        if not parsed_status and self._is_termination_msg(message):
            return
        
        if parsed_status: 
            # if parsing succeeds, update the current plan to be the newly parsed one
            self.current_plan = parsed_content

            if self.verifier:
                verified_results = self.verifier.verify(parsed_content)
                verified_error_message = verified_results['message']
                verified_status = verified_results['status']
                verified_error_code = verified_results['error_code']

                if verified_status: 
                    # if verification succeeds
                    if self.executor:
                        # go to execution stage if there is an executor module
                        executed_results = self.executor.execute(self.current_task_id, parsed_content)
                        executed_error_message = executed_results['message']
                        executed_status = executed_results['status']

                        # send the message from running the execution module
                        reply = self.feedback_generator.get_prompt("execution", executed_status, executed_error_message)
                        if executed_status:
                            self.send(reply, sender, request_reply=False) # no need to reply when the global plan is executed correctly
                            self._consecutive_auto_reply_counter[sender.name] = 0
                            return
                        else:
                            # always check the counter before requesting any reply
                            if self.sender_hits_max_reply(sender): 
                                # reset the consecutive_auto_reply_counter
                                self._consecutive_auto_reply_counter[sender.name] = 0
                                return
                        
                            self._consecutive_auto_reply_counter[sender.name] += 1
                            self.feedback_types.append("execution")
                            self.send(reply, sender, request_reply=True)
                    else:
                        # if no exeuction module, send the message from verifier
                        reply = self.feedback_generator.get_prompt("verification", verified_status, verified_error_message, verified_error_code)
                        self.send(reply, sender, request_reply=False) # request_reply=false because verification is successful
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                else:
                    if self.sender_hits_max_reply(sender):
                        # reset the consecutive_auto_reply_counter
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                    
                    # if verification fails, construct a feedback message from the error code and message of the verifier
                    # send the feedback message, and request a reply
                    self._consecutive_auto_reply_counter[sender.name] += 1
                    reply = self.feedback_generator.get_prompt("verification", verified_status, verified_error_message, verified_error_code)
                    self.feedback_types.append("verification")
                    self.send(reply, sender, request_reply=True)
            else:
                if self.executor: 
                    # if verifier is NOT specified but executor is, run the execution module
                    executed_results = self.executor.execute(self.current_task_id, parsed_content)
                    executed_error_message = executed_results['message']
                    executed_status = executed_results['status']

                    # send the message from running the execution module
                    reply = self.feedback_generator.get_prompt("execution", executed_status, executed_error_message)
                    if executed_status:
                        self.send(reply, sender, request_reply=False)
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                    else:
                        if self.sender_hits_max_reply(sender):
                            # reset the consecutive_auto_reply_counter
                            self._consecutive_auto_reply_counter[sender.name] = 0
                            return
                        
                        self._consecutive_auto_reply_counter[sender.name] += 1
                        self.feedback_types.append("execution")
                        self.send(reply, sender, request_reply=True)
                else:
                    # if parsing succeeds, and neither verifier nor executor is specified, simply return
                    reply = self.feedback_generator.get_prompt("parsing", parsed_status, parsed_error_message, parsed_error_code)
                    self.send(reply, sender, request_reply=False)
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
        else:
            if self.sender_hits_max_reply(sender):
                # reset the consecutive_auto_reply_counter
                self._consecutive_auto_reply_counter[sender.name] = 0
                return

            # if parsing fails, construct a feedback message from the error code and message of the parser
            # send the feedback message, and request a reply
            self._consecutive_auto_reply_counter[sender.name] += 1
            reply = self.feedback_generator.get_prompt("parsing", parsed_status, parsed_error_message, parsed_error_code)
            self.feedback_types.append("parsing")
            self.send(reply, sender, request_reply=True)

    def generate_init_message(self, query):
        content = self.prompt_generator.get_prompt_for_curr_query(query)
        return content

    def initiate_chat(self, assistant, message, task_id, log_prompt_only=False):
        self.current_task_id = task_id
        self.current_plan = []
        self.feedback_types = []
        self._v3_summary_calls = 0
        self._v3_summary_assistant_mark = 0
        initial_message = self.generate_init_message(message)
        if log_prompt_only:
            print(initial_message)
        else:
            assistant.receive(initial_message, self, request_reply=True)
            
            # get the plan
            plan = self.current_plan
            print("\nFINAL plan:", plan)
        


class MnmsUserAgentLocal(MnmsUserAgent):
    def _v3_enabled(self):
        explicit_v3 = os.environ.get("MNMS_V3_PRUNE", "").lower() in {"1", "true", "yes"}
        configured_gate = (
            os.environ.get("MNMS_ENABLE_SUMMARY_TOOL", "0").lower() in {"1", "true", "yes"}
            and os.environ.get("SUMMARY_KL_ENABLE", "0").lower() in {"1", "true", "yes"}
        )
        return explicit_v3 or configured_gate

    def _v3_history_candidates(self, sender):
        history = sender._oai_messages.get(self, [])
        assistant_texts = [
            str(item.get("content") or "")
            for item in history
            if item.get("role") == "assistant" and item.get("content")
        ]
        recent = assistant_texts[self._v3_summary_assistant_mark:]
        return recent, sum(max(1, len(text) // 4) for text in recent)

    def _v3_summary_due(self, sender):
        if not self._v3_enabled() or self._v3_summary_calls:
            return False
        min_tools = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_REAL_TOOLS", "2"))
        min_tokens = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_REASONING_TOKENS", "80"))
        min_candidates = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_CANDIDATES", "2"))
        candidates, token_estimate = self._v3_history_candidates(sender)
        real_tools = len([node for node in self.current_plan if node.get("name") != "summarize_the_task"])
        return (
            real_tools >= min_tools
            and token_estimate >= min_tokens
            and len(candidates) >= min_candidates
        )

    def _v3_summary_due_message(self, sender):
        candidates, token_estimate = self._v3_history_candidates(sender)
        real_tools = len([node for node in self.current_plan if node.get("name") != "summarize_the_task"])
        return (
            "SUMMARY DUE: the gated memory policy is now satisfied. You have completed "
            f"{real_tools} real tool calls and accumulated about {token_estimate} unsummarized reasoning tokens "
            f"across {len(candidates)} candidate steps. Your next action MUST be exactly one "
            "summarize_the_task call with a faithful summary of the request, tool calls, observations, "
            "known facts, uncertainty, and next action. This summary is zero-cost and does not count "
            "as a real planning step."
        )

    def _v3_summary_action(self, sender, message, node):
        """Apply V3 MNMS pruning: retain native ACTION blocks only after a real summary action."""
        if not self._v3_enabled():
            return False
        if not isinstance(node, dict) or node.get("name") != "summarize_the_task":
            return False
        if not self._v3_summary_due(sender):
            self.send(
                "SUMMARY DEFERRED: the gated policy is not satisfied yet. Continue with the next real "
                "tool action and wait until the controller explicitly sends SUMMARY DUE.",
                sender,
                request_reply=True,
            )
            return True
        summary = str((node.get("args") or {}).get("content") or "").strip()
        if not summary:
            self._v3_summary_calls += 1
            return True
        try:
            from summary_kl_core import log_summarization_kl
            history = sender._oai_messages.get(self, [])
            assistant_slots = [i for i, item in enumerate(history) if item.get("role") == "assistant" and item.get("content")]
            prior = [str(history[i]["content"]) for i in assistant_slots][self._v3_summary_assistant_mark:]
            event = log_summarization_kl(
                system_prompt="MNMS native AutoGen step-by-step planning.", summary_text=summary,
                prior_assistant_raw_texts=prior, current_pre_tool_text=str(message),
                model=os.environ.get("SUMMARY_KL_MODEL"), task_id=self.current_task_id,
                output_dir=os.environ.get("SUMMARY_KL_OUTPUT_DIR"),
            )
            local_slots = assistant_slots[self._v3_summary_assistant_mark:]
            prune_indices = (event or {}).get("prune_indices", [])
            discard_no_prune = os.environ.get("MNMS_SUMMARY_GATE_DISCARD_NO_PRUNE", "1").lower() in {"1", "true", "yes"}
            for local_idx in prune_indices:
                if 0 <= local_idx < len(local_slots):
                    old = str(history[local_slots[local_idx]]["content"])
                    action = re.search(r"ACTION\s+\d+\s*:\s*(\{[\s\S]*\})", old)
                    history[local_slots[local_idx]]["content"] = action.group(0) if action else ""
            self._v3_summary_calls += 1
            self._v3_summary_assistant_mark = len(assistant_slots)
            if discard_no_prune and not prune_indices:
                summary = "No redundant reasoning was identified by KL; retain the existing context and continue."
        except Exception as exc:
            print(f"summary KL unavailable for {self.current_task_id}: {exc}")
            self._v3_summary_calls += 1
        # This is a memory operation rather than a task action: it must not
        # consume the native step-by-step reply budget or enter current_plan.
        self.send(
            "OBSERVATION: Zero-cost internal summary:\n"
            + summary
            + "\nContinue with the next THOUGHT and ACTION, or TERMINATE if complete.",
            sender,
            request_reply=True,
        )
        return True

    def receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])
        self._process_received_message(message, sender, silent)
        
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        parsed_error_message = parsed_results['message']
        parsed_error_code = parsed_results['error_code']

        # Terminate planning if parsing fails and termination check returns true
        # Otherwise proceed to parsing feedback or verification/execution
        if not parsed_status and self._is_termination_msg(message):
            return

        if parsed_status:
            # if parsing succeeds
            if self._v3_summary_action(sender, message, parsed_content):
                return
            if self._v3_summary_due(sender):
                self.send(self._v3_summary_due_message(sender), sender, request_reply=True)
                return
            if self.verifier:
                # verify a local step
                verified_results = self.verifier.verify_single_node(parsed_content)
                verified_error_message = verified_results['message']
                verified_status = verified_results['status']
                verified_error_code = verified_results['error_code']

                if verified_status: 
                    # if verification succeeds
                    if self.executor:
                        # go to execution stage if there is an executor module
                        executed_results = self.executor.execute(self.current_task_id, parsed_content)
                        executed_error_message = executed_results['message']
                        executed_status = executed_results['status']
                        # if this step executes successfully, adds it to the current plan
                        if executed_status:
                            self.current_plan.append(parsed_content)

                        if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                            # reset the consecutive_auto_reply_counter
                            self._consecutive_auto_reply_counter[sender.name] = 0
                            return
                        
                        if self._v3_summary_due(sender):
                            self._consecutive_auto_reply_counter[sender.name] += 1
                            self.send(self._v3_summary_due_message(sender), sender, request_reply=True)
                            return
                        # send the message from running the execution module
                        self._consecutive_auto_reply_counter[sender.name] += 1
                        reply = self.feedback_generator.get_prompt("execution", executed_status, executed_error_message) 
                        if not executed_status:
                            self.feedback_types.append("execution")
                        # always request a reply for 1) generating the next step if execution succeeds
                        # or 2) fixing this current step if execution fails
                        self.send(reply, sender, request_reply=True)
                    else:
                        # if verification succeeds and no exeuctor is specified, consider it a correct step and add it to plan
                        self.current_plan.append(parsed_content)

                        if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                            # reset the consecutive_auto_reply_counter
                            self._consecutive_auto_reply_counter[sender.name] = 0
                            return
                        
                        self._consecutive_auto_reply_counter[sender.name] += 1
                        # if no exeuction module, send the message from verifier
                        reply = self.feedback_generator.get_prompt("verification", verified_status, verified_error_message, verified_error_code) 
                        # request a reply for generating the next step
                        self.send(reply, sender, request_reply=True)
                else:
                    if self.sender_hits_max_reply(sender):
                        # reset the consecutive_auto_reply_counter
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                    
                    # if verification fails, construct a feedback message from the error code and message of the verifier
                    # send the feedback message, and request a reply to fix the current step
                    self._consecutive_auto_reply_counter[sender.name] += 1
                    reply = self.feedback_generator.get_prompt("verification", verified_status, verified_error_message, verified_error_code)
                    self.feedback_types.append("verification")
                    self.send(reply, sender, request_reply=True)
            else:
                if self.executor:
                    # if verifier is NOT specified but executor is, run the execution module
                    executed_results = self.executor.execute(self.current_task_id, parsed_content)
                    executed_error_message = executed_results['message']
                    executed_status = executed_results['status']
                    # if this step executes successfully, adds it to the current plan
                    if executed_status:
                        self.current_plan.append(parsed_content)

                    if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                        # reset the consecutive_auto_reply_counter
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                    
                    # send the message from running the execution module
                    self._consecutive_auto_reply_counter[sender.name] += 1
                    reply = self.feedback_generator.get_prompt("execution", executed_status, executed_error_message) 
                    if not executed_status:
                        self.feedback_types.append("execution")
                    # always request a reply for 1) generating the next step if execution succeeds
                    # or 2) fixing this current step if execution fails
                    self.send(reply, sender, request_reply=True)
                else:
                    # if parsing succeeds, and neither verifier nor executor is specified, add current step to plan
                    # request a reply for generating the next step 
                    self.current_plan.append(parsed_content)

                    if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                        # reset the consecutive_auto_reply_counter
                        self._consecutive_auto_reply_counter[sender.name] = 0
                        return
                    
                    self._consecutive_auto_reply_counter[sender.name] += 1
                    reply = self.feedback_generator.get_prompt("parsing", parsed_status, parsed_error_message, parsed_error_code)
                    self.send(reply, sender, request_reply=True)
        else:
            if self.sender_hits_max_reply(sender):
                # reset the consecutive_auto_reply_counter
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            
            # if parsing fails, construct a feedback message from the error code and message of the parser
            # send the feedback message, and request a reply
            self._consecutive_auto_reply_counter[sender.name] += 1
            reply = self.feedback_generator.get_prompt("parsing", parsed_status, parsed_error_message, parsed_error_code)
            self.feedback_types.append("parsing")
            self.send(reply, sender, request_reply=True)

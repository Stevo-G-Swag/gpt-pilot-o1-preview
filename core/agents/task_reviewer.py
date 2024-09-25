from core.agents.base import BaseAgent
from core.agents.convo import AgentConvo
from core.agents.response import AgentResponse
from core.log import get_logger
import difflib

log = get_logger(__name__)


class TaskReviewer(BaseAgent):
    agent_type = "task-reviewer"
    display_name = "Task Reviewer"

    async def run(self) -> AgentResponse:
        response = await self.review_code_changes()
        self.next_state.complete_step()
        return response

    async def review_code_changes(self) -> AgentResponse:
        """
        Review all the code changes during the current task.
        """

        log.debug(f"Reviewing code changes for task {self.current_state.current_task['description']}")
        all_feedbacks = [
            iteration["user_feedback"].replace("```", "").strip()
            for iteration in self.current_state.iterations
            if iteration.get("user_feedback")
        ]
        bug_hunter_instructions = [
            iteration["bug_hunting_cycles"][-1]["human_readable_instructions"].replace("```", "").strip()
            for iteration in self.current_state.iterations
            if iteration.get("bug_hunting_cycles")
        ]

        files_before_modification = self.current_state.modified_files

        # Generate diffs for modified files
        file_diffs = []
        for file in self.current_state.files:
            if file.path in files_before_modification:
                old_content = files_before_modification[file.path]
                new_content = file.content.content
                diff = difflib.unified_diff(
                    old_content.splitlines(),
                    new_content.splitlines(),
                    fromfile=f"{file.path} (before)",
                    tofile=f"{file.path} (after)",
                    lineterm=""
                )
                diff_text = "\n".join(list(diff))
                file_diffs.append((file.path, diff_text))

        llm = self.get_llm()
        convo = AgentConvo(self).template(
            "review_task",
            all_feedbacks=all_feedbacks,
            file_diffs=file_diffs,
            bug_hunter_instructions=bug_hunter_instructions,
        )
        llm_response: str = await llm(convo)

        if "done" in llm_response.strip().lower()[-20:]:
            return AgentResponse.done(self)
        else:
            return AgentResponse.task_review_feedback(self, llm_response)


"""Spec-only prompt registry for future final LinkedIn post flow agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_access import ROLE_CANDIDATE_WRITER
from services.packaging.linkedin_post_flow_access import ROLE_QUALITY_EVALUATOR
from services.packaging.linkedin_post_flow_access import get_access_contract


PROMPT_STATUS_BASELINE = "baseline"
PROMPT_STATUS_EXPERIMENTAL = "experimental"
PROMPT_STATUS_ACTIVE = "active"
PROMPT_STATUS_DEPRECATED = "deprecated"

FINAL_POST_PROMPT_STATUSES = (
    PROMPT_STATUS_BASELINE,
    PROMPT_STATUS_EXPERIMENTAL,
    PROMPT_STATUS_ACTIVE,
    PROMPT_STATUS_DEPRECATED,
)

MODEL_ROLE_CANDIDATE_WRITER_PRIMARY = "final_post_candidate_writer_primary"
MODEL_ROLE_QUALITY_EVALUATOR_PRIMARY = "final_post_quality_evaluator_primary"
MODEL_ROLE_REPAIR_PRIMARY = "final_post_repair_primary"
MODEL_ROLE_FACTUALITY_REVIEWER_PRIMARY = "final_post_factuality_reviewer_primary"

PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF = "final_post_candidate_from_brief"
PROMPT_FINAL_POST_QUALITY_EVALUATOR = "final_post_quality_evaluator"


@dataclass(frozen=True)
class FinalPostPromptContract:
    prompt_name: str
    prompt_version: str
    prompt_path: str
    agent_role: str
    access_mode: str
    input_contract: str
    output_contract: str
    model_role: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
            "agent_role": self.agent_role,
            "access_mode": self.access_mode,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "model_role": self.model_role,
            "status": self.status,
        }


FINAL_POST_PROMPT_REGISTRY = (
    FinalPostPromptContract(
        prompt_name=PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
        agent_role=ROLE_CANDIDATE_WRITER,
        access_mode=get_access_contract(ROLE_CANDIDATE_WRITER).access_mode,
        input_contract=(
            "PostBrief + AngleDecision + AngleDecision.authorial_voice_directive + selected evidence"
        ),
        output_contract="FinalPostPayload",
        model_role=MODEL_ROLE_CANDIDATE_WRITER_PRIMARY,
        status=PROMPT_STATUS_BASELINE,
    ),
    FinalPostPromptContract(
        prompt_name=PROMPT_FINAL_POST_QUALITY_EVALUATOR,
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_quality_evaluator.txt",
        agent_role=ROLE_QUALITY_EVALUATOR,
        access_mode=get_access_contract(ROLE_QUALITY_EVALUATOR).access_mode,
        input_contract="PostEditorialInput",
        output_contract="QualityReviewResult",
        model_role=MODEL_ROLE_QUALITY_EVALUATOR_PRIMARY,
        status=PROMPT_STATUS_EXPERIMENTAL,
    ),
)


def get_prompt_contract(prompt_name: str) -> FinalPostPromptContract:
    for contract in FINAL_POST_PROMPT_REGISTRY:
        if contract.prompt_name == prompt_name:
            return contract
    raise KeyError(prompt_name)


def list_prompt_contracts() -> tuple[FinalPostPromptContract, ...]:
    return FINAL_POST_PROMPT_REGISTRY


def prompt_contract_to_prompt_metadata(contract: FinalPostPromptContract) -> PromptMetadata:
    return PromptMetadata(
        prompt_name=contract.prompt_name,
        prompt_version=contract.prompt_version,
        prompt_path=contract.prompt_path,
    )

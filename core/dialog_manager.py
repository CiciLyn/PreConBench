"""
dialog management module for medical consultation simulation.
This module provides functionality for:
1. Managing doctor-patient dialog
2. Handling medical record generation
3. Tracking conversation history
4. Managing token usage
"""

from typing import Dict, Optional, Tuple, List
import time
import pandas as pd
from dataclasses import dataclass
from langchain_community.chat_message_histories import ChatMessageHistory
from retry import retry

from core.logger import logger
from core.models import GPTModel


@dataclass
class DialogResult:
    """Container for dialog processing results.

    Attributes:
        dialog: The complete dialog text
        medical_record: The generated medical record
        tokens_dict: Dictionary tracking token usage
        patient_think: Patient's thought process
    """

    dialog: str
    medical_record: str
    tokens_dict: Dict[str, int]
    patient_think: str


class DialogManager:
    """Manages the dialog between doctor and patient models.

    This class handles:
    1. Initialization of dialog participants
    2. Conversation flow management
    3. Medical record generation
    4. Token usage tracking
    """

    def __init__(
        self,
        patient_llm: GPTModel,
        doctor_llm: GPTModel,
        patient_system_prompt_template: str,
        patient_slow_think_prompt_template: str,
        doctor_system_prompt_template: str,
        doctor_slow_think_prompt_template: str,
        final_doctor_record_prompt_template: str,
        max_turn_num: int,
        patient_info: Dict[str, str],
    ):
        """Initialize the dialog manager.

        Args:
            patient_llm: Patient language model
            doctor_llm: Doctor language model
            patient_system_prompt_template: Template for patient system prompt
            patient_slow_think_prompt_template: Template for patient slow think prompt
            doctor_system_prompt_template: Template for doctor system prompt
            doctor_slow_think_prompt_template: Template for doctor slow think prompt
            final_doctor_record_prompt_template: Template for medical record generation
            max_turn_num: Maximum number of dialog turns
            patient_info: Dictionary containing patient information
        """
        # Initialize patient information
        self.patient_record_info = self._format_patient_record_info(patient_info)
        self.patient_info = self._format_patient_basic_info(patient_info)

        # Initialize prompts
        self.patient_system_prompt = self._format_patient_system_prompt(
            patient_system_prompt_template, patient_info
        )
        self.doctor_system_prompt = self._format_doctor_system_prompt(
            doctor_system_prompt_template, patient_info, max_turn_num
        )
        self.patient_slow_think_prompt = self._format_patient_slow_think_prompt(
            patient_slow_think_prompt_template, patient_info
        )
        self.doctor_slow_think_prompt = doctor_slow_think_prompt_template


        # Initialize models and parameters
        self.patient_llm = patient_llm
        self.doctor_llm = doctor_llm
        self.max_turn_num = max_turn_num
        self.final_doctor_record_prompt = final_doctor_record_prompt_template

        # Initialize conversation history
        self.patient_memory = ChatMessageHistory()
        self.doctor_memory = ChatMessageHistory()

    def _format_patient_record_info(self, patient_info: Dict[str, str]) -> str:
        """Format patient's record information."""
        return (
            f"{patient_info['主诉']}\n"
            f"{patient_info['现病史']}\n"
            f"{patient_info['既往史']}"
        )

    def _format_patient_basic_info(self, patient_info: Dict[str, str]) -> str:
        """Format patient's basic information."""
        occupation = patient_info.get("病人职业", "无")
        if pd.isna(occupation) or not occupation:
            occupation = "无"

        return f"""
病人的信息如下：
## 病人基本信息 ##
姓名：{patient_info.get('病人姓名', '王青')}
性别：{patient_info['病人性别']}
年龄：{patient_info['病人年龄']}
职业：{occupation}
科室信息：{patient_info['科室']}
"""

    def _format_patient_system_prompt(
        self, template: str, patient_info: Dict[str, str]
    ) -> str:
        """Format patient system prompt."""
        occupation = patient_info.get("病人职业", "无")

        if pd.isna(occupation) or not occupation:
            occupation = "无"

        return template.format(
            gender=patient_info["病人性别"],
            age=str(patient_info["病人年龄"]),
            occupation=occupation,
            name=patient_info.get("病人姓名", "王青"),
            personality=patient_info["character"],
            info=self.patient_record_info,
        )

    def _format_doctor_system_prompt(
        self, template: str, patient_info: Dict[str, str], max_turns: int
    ) -> str:
        """Format doctor system prompt."""
        occupation = patient_info.get("病人职业", "无")

        if pd.isna(occupation) or not occupation:
            occupation = "无"

        return template.format(
            department=patient_info["科室"],
            gender=patient_info["病人性别"],
            age=str(patient_info["病人年龄"]),
            occupation=occupation,
            name=patient_info.get("病人姓名", "王青"),
            max_turns=max_turns,
        )

    def _format_patient_slow_think_prompt(self, template: str, patient_info: Dict[str, str]) -> str:
        """Format patient slow think prompt."""
        return template.format(
            personality=patient_info["character"],
            info=self.patient_record_info,
        )
    def _format_doctor_slow_think_prompt(self, template: str, dialog_history_str: str) -> str:
        """Format doctor slow think prompt."""
        return template.format(
            dialog_history=dialog_history_str
        )

    def safe_add_message(
        self, memory: ChatMessageHistory, role: str, content: str
    ) -> bool:
        """Safely add a message to the conversation history.

        Args:
            memory: Chat message history
            role: Message role (user/assistant/system)
            content: Message content

        Returns:
            bool: True if message was added successfully
        """
        try:
            if not memory.messages:
                memory.add_message({"role": role, "content": content})
                return True

            current_indices = len(memory.messages)
            if memory.messages[0]["role"] == "system":
                current_indices -= 1

            if role == "user":
                assert current_indices % 2 == 0
            elif role == "assistant":
                assert current_indices % 2 == 1
            else:
                raise ValueError(f"Invalid role: {role}")

            memory.add_message({"role": role, "content": content})
            return True
        except Exception as e:
            raise ValueError(f"Failed to add message: {str(e)}")

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def generate_dialog(self, verbose: bool = False) -> DialogResult:
        """Generate a complete dialog between doctor and patient.

        Args:
            verbose: Whether to log detailed information

        Returns:
            DialogResult containing the dialog and related information
        """
        tokens_dict = {
            "patient_in": 0,
            "patient_out": 0,
            "doctor_in": 0,
            "doctor_out": 0,
        }

        try:
            # Initialize conversation
            self._initialize_conversation()

            doctor_response, doctor_tokens = self._get_doctor_response(
                self.doctor_memory.messages, verbose
            )
            self.safe_add_message(self.doctor_memory, "assistant", doctor_response)
            self.safe_add_message(self.patient_memory, "user", doctor_response)
            tokens_dict["doctor_in"] += doctor_tokens[0]
            tokens_dict["doctor_out"] += doctor_tokens[1]
            if verbose:
                logger.info(f"doctor response: {doctor_response}")

            # Generate dialog turns
            for turn in range(self.max_turn_num):
                try:
                    # Get patient response
                    patient_response, patient_tokens = self._get_patient_response(
                        self.patient_memory.messages, verbose
                    )
                    dialog_process_warning = f'\n\n对话第{turn+1}轮结束。开始第{turn+2}轮对话: \n\n'
                    patient_response = patient_response + dialog_process_warning
                    if not patient_response:
                        raise ValueError("Empty patient response")
                    self.safe_add_message(
                        self.patient_memory, "assistant", patient_response
                    )
                    self.safe_add_message(self.doctor_memory, "user", patient_response)
                    if verbose:
                        logger.info(f"patient response: {patient_response}")

                    # Get doctor response
                    doctor_response, doctor_tokens = self._get_doctor_response(
                        self.doctor_memory.messages, verbose
                    )
                    if not doctor_response:
                        raise ValueError("Empty doctor response")
                    self.safe_add_message(
                        self.doctor_memory, "assistant", doctor_response
                    )
                    self.safe_add_message(self.patient_memory, "user", doctor_response)
                    if verbose:
                        logger.info(f"doctor response: {doctor_response}")

                    # Update token counts
                    self._update_token_counts(
                        tokens_dict, patient_tokens, doctor_tokens
                    )

                    # Check for dialog completion
                    if "[预问诊单]" in doctor_response:
                        return self._generate_final_result(tokens_dict)

                except Exception as e:
                    logger.error(f"dialog turn {turn + 1} failed: {str(e)}")
                    return self._create_error_result(tokens_dict)

            # if max turn reached, generate final medical record. 

            return self._generate_final_medical_record(tokens_dict)

        except Exception as e:
            logger.error(f"dialog generation failed: {str(e)}")
            return self._create_error_result(tokens_dict)

    def _initialize_conversation(self) -> None:
        """Initialize the conversation with system prompts."""
        self.safe_add_message(self.doctor_memory, "system", self.doctor_system_prompt)
        self.safe_add_message(
            self.doctor_memory,
            "user",
            f"{self.patient_info}\n目前对话尚未开始，请开启你在对话中的首轮提问",
        )
        self.safe_add_message(self.patient_memory, "system", self.patient_system_prompt)

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def _get_patient_response(
        self, messages: List[Dict], verbose: bool
    ) -> Tuple[Optional[str], Tuple[int, int]]:
        """Get response from patient model."""
        start_time = time.time()
        response, in_tokens, out_tokens = self.patient_llm.chat(messages, self.patient_slow_think_prompt)
        response = self._post_process_patient_response(response)
        if verbose:
            logger.info(f"Patient API call time: {time.time() - start_time:.2f}s")
        return response, (in_tokens, out_tokens)

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def _get_doctor_response(
        self, messages: List[Dict], verbose: bool
    ) -> Tuple[Optional[str], Tuple[int, int]]:
        """Get response from doctor model."""
        start_time = time.time()
        dialog_history_str = "\n".join([msg["content"] for msg in messages[1:]])
        doctor_slow_think_prompt = self._format_doctor_slow_think_prompt(
            self.doctor_slow_think_prompt, dialog_history_str
        )
        response, in_tokens, out_tokens = self.doctor_llm.chat(messages, doctor_slow_think_prompt)
        response = self._post_process_doctor_response(response)
        if verbose:
            logger.info(f"Doctor API call time: {time.time() - start_time:.2f}s")
        return response, (in_tokens, out_tokens)

    def _update_token_counts(
        self,
        tokens_dict: Dict[str, int],
        patient_tokens: Tuple[int, int],
        doctor_tokens: Tuple[int, int],
    ) -> None:
        """Update token usage counts."""
        tokens_dict["patient_in"] += patient_tokens[0]
        tokens_dict["patient_out"] += patient_tokens[1]
        tokens_dict["doctor_in"] += doctor_tokens[0]
        tokens_dict["doctor_out"] += doctor_tokens[1]

    def _generate_final_result(self, tokens_dict: Dict[str, int]) -> DialogResult:
        """Generate final result when dialog is complete."""
        medical_record = self.get_medical_record()
        final_dialog = self.get_final_dialog()
        patient_think = self.get_final_dialog_with_think()

        return DialogResult(
            dialog=final_dialog,
            medical_record=medical_record,
            tokens_dict=tokens_dict,
            patient_think=patient_think,
        )

    def _create_error_result(self, tokens_dict: Dict[str, int]) -> DialogResult:
        """Create error result when dialog fails."""
        return DialogResult(
            dialog="None",
            medical_record="None",
            tokens_dict=tokens_dict,
            patient_think="None",
        )

    def _generate_final_medical_record(
        self, tokens_dict: Dict[str, int], verbose: bool = False
    ) -> DialogResult:
        """Generate final medical record when max turns reached."""
        self.safe_add_message(
            self.doctor_memory, "user", self.final_doctor_record_prompt
        )
        doctor_response, doctor_in_token, doctor_out_token = self.doctor_llm.chat(
            self.doctor_memory.messages
        )
        if verbose:
            logger.info(f"doctor response: {doctor_response}")
        self.safe_add_message(self.doctor_memory, "assistant", doctor_response)
        tokens_dict["doctor_in"] += doctor_in_token
        tokens_dict["doctor_out"] += doctor_out_token

        return DialogResult(
            dialog=self.get_final_dialog(),
            medical_record=self.get_medical_record(),
            tokens_dict=tokens_dict,
            patient_think="None",
        )

    def get_medical_record(self) -> str:
        """Extract medical record from dialog history."""
        try:
            dialog_str = "\n".join(
                [msg["content"] for msg in self.doctor_memory.messages[1:]]
            )
            # Extract content between [预问诊单]: and the end
            if "[预问诊单]:" in dialog_str:
                record_content = dialog_str.rsplit("[预问诊单]:", 1)[1].strip()
                # Remove any content after the record if it exists
                if "-----End of Record-----" in record_content:
                    record_content = record_content.split("-----End of Record-----", 1)[
                        0
                    ].strip()
                return record_content
            return "None"
        except Exception as e:
            logger.error(f"Failed to get medical record: {str(e)}")
            return "None"

    def get_final_dialog(self) -> str:
        """Format final dialog from conversation history."""
        messages = self.doctor_memory.messages[1:]
        dialog = []

        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                # Skip the initial patient info message
                if "病人的信息如下" in content:
                    continue
                prefix = "" if "病人" in content else "[患者]: "
                dialog.append(f"{prefix}{content}")
            elif msg["role"] == "assistant":
                prefix = "" if "医生" in content else "[医生]: "
                dialog.append(f"{prefix}{content}")

        # Get only the dialog part before [预问诊单]
        dialog_no_record = "\n".join(dialog).rsplit("[预问诊单]", 1)[0]
        return dialog_no_record

    def get_final_dialog_with_think(self) -> str:
        """Get dialog with patient's thought process."""
        return "\n".join([msg["content"] for msg in self.patient_memory.messages])

    def _post_process_doctor_response(self, response: str) -> str:
        """Post-process doctor's response."""
        if "[病人]" in response:
            return response.rsplit("[病人]", 1)[0].strip()
        return response.strip()

    def _post_process_patient_response(self, response: str) -> str:
        """Post-process doctor's response."""
        # Filter out the doctor dialogue performance part
        if "[医生对话表现]:" in response:
            response = "[病人]: " + response.split("[病人]:", 1)[1].strip()
        elif '[病人]' in response:
            response = '[病人]' + response.rsplit('[病人]', 1)[1].strip()
        elif '[病人]' not in response:
            response = '[病人]: ' + response.strip()
        return response.strip()

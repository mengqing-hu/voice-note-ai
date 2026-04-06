"""
utils/llm_client.py
LLM 客户端封装：支持多个 LLM 提供商（Gemini, Groq）
- 自动重试、异常处理、token 用量记录
"""

import time
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    多模型 LLM 客户端封装。
    同时支持 Gemini 和 Groq，自动检测提供商。
    统一处理：配置初始化、重试逻辑、异常捕获、用量统计。

    Usage:
        from src.utils.llm_client import LLMClient

        # 使用 Groq（推荐，速度更快）
        client = LLMClient(api_key="gsk_xxx...", provider="groq", model="mixtral-8x7b-32768")
        
        # 或使用 Gemini
        client = LLMClient(api_key="your_key", provider="gemini", model="gemini-2.0-flash")
        
        response = client.chat("请总结以下内容：...")
        print(response)
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "groq",
        model: str = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Args:
            api_key:     API Key（Groq 或 Gemini）
            provider:    "groq" 或 "gemini"，默认 groq
            model:       模型名称。不指定则使用提供商默认：
                        - groq: "mixtral-8x7b-32768"
                        - gemini: "gemini-2.0-flash"
            max_retries: 失败后最多重试次数
            retry_delay: 每次重试前等待秒数
        """
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 设置默认模型
        if model is None:
            model = "mixtral-8x7b-32768" if self.provider == "groq" else "gemini-2.0-flash"
        
        self.model_name = model

        # 初始化对应提供商的客户端
        if self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=api_key)
            logger.info(f"✓ Groq 客户端初始化完成，模型: {model}")
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info(f"✓ Gemini 客户端初始化完成，模型: {model}")
        else:
            raise ValueError(f"不支持的提供商: {provider}。支持: groq, gemini")

        # 用量统计（本次运行累计）
        self._total_calls = 0
        self._total_input_chars = 0
        self._total_output_chars = 0

    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        """
        发送单轮对话请求，返回模型回复文本。
        内置重试逻辑，失败后等待后重试。

        Args:
            prompt:      完整 prompt 文本
            temperature: 温度，0.0~1.0，提取结构化信息建议用低温（0.1~0.3）

        Returns:
            模型回复的文本内容

        Raises:
            RuntimeError: 重试次数耗尽后仍失败

        Usage:
            result = client.chat(prompt_text)
        """
        if self.provider == "groq":
            return self._chat_groq(prompt, temperature)
        elif self.provider == "gemini":
            return self._chat_gemini(prompt, temperature)

    def _chat_groq(self, prompt: str, temperature: float) -> str:
        """Groq API 调用"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"调用 Groq - {self.model_name}（第 {attempt} 次）输入: {len(prompt)} 字")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                )

                output = response.choices[0].message.content

                # 统计用量
                self._total_calls += 1
                self._total_input_chars += len(prompt)
                self._total_output_chars += len(output)

                logger.info(f"调用成功，输出: {len(output)} 字")
                return output

            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # 检查是否是配额超出错误
                is_quota_error = any([
                    "429" in error_str,
                    "quota" in error_str.lower(),
                    "rate limit" in error_str.lower(),
                ])
                
                # 配额错误立即抛出，无需重试
                if is_quota_error:
                    logger.error(f"API 配额超出，无需重试: {e}")
                    raise RuntimeError(f"API 配额已用尽: {error_str}") from e
                
                logger.warning(f"调用失败（第 {attempt} 次）: {e}")
                if attempt < self.max_retries:
                    logger.info(f"等待 {self.retry_delay}s 后重试...")
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"LLM 调用失败，已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    def _chat_gemini(self, prompt: str, temperature: float) -> str:
        """Gemini API 调用"""
        import google.generativeai as genai
        
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"调用 {self.model_name}（第 {attempt} 次）输入: {len(prompt)} 字")

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                    ),
                )
                
                # 检查响应是否被安全过滤器阻止
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    raise RuntimeError(f"请求被安全过滤器阻止: {response.prompt_feedback.block_reason}")
                
                # 检查候选响应是否被阻止
                if response.candidates and response.candidates[0].finish_reason and \
                   str(response.candidates[0].finish_reason) == "SAFETY":
                    safety_ratings = response.candidates[0].safety_ratings if hasattr(response.candidates[0], 'safety_ratings') else []
                    raise RuntimeError(f"响应被安全过滤器阻止。安全评分: {safety_ratings}")
                
                # 尝试获取文本
                try:
                    output = response.text
                except (ValueError, AttributeError) as e:
                    raise RuntimeError(f"无法获取响应文本。可能被内容过滤器阻止: {e}")
                
                if not output or output.strip() == "":
                    raise RuntimeError("LLM 返回空响应")

                # 统计用量
                self._total_calls += 1
                self._total_input_chars += len(prompt)
                self._total_output_chars += len(output)

                logger.info(f"调用成功，输出: {len(output)} 字")
                return output

            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # 检查是否是配额超出错误
                is_quota_error = any([
                    "429" in error_str,
                    "exceed" in error_str.lower(),
                    "quota" in error_str.lower(),
                    "exceeded" in error_str.lower(),
                    "rate limit" in error_str.lower(),
                    "free tier" in error_str.lower(),
                ])
                
                # 配额错误立即抛出，无需重试
                if is_quota_error:
                    logger.error(f"API 配额超出，无需重试: {e}")
                    raise RuntimeError(f"API 配额已用尽: {error_str}") from e
                
                logger.warning(f"调用失败（第 {attempt} 次）: {e}")
                if attempt < self.max_retries:
                    logger.info(f"等待 {self.retry_delay}s 后重试...")
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"LLM 调用失败，已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    def chat_with_system(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """
        带 system prompt 的对话（通过拼接实现，Gemini 原生不区分 system/user）。

        Usage:
            result = client.chat_with_system(
                system_prompt="你是一个专业的面试分析师...",
                user_prompt=dialog_text,
            )
        """
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.chat(full_prompt, temperature=temperature)

    def get_usage_stats(self) -> dict:
        """
        获取本次运行的累计调用统计。

        Returns:
            {"total_calls": int, "total_input_chars": int, "total_output_chars": int}
        """
        return {
            "total_calls": self._total_calls,
            "total_input_chars": self._total_input_chars,
            "total_output_chars": self._total_output_chars,
        }

    def log_usage(self) -> None:
        """打印累计用量到日志"""
        stats = self.get_usage_stats()
        logger.info(
            f"LLM 用量统计 | 调用次数: {stats['total_calls']} | "
            f"输入: {stats['total_input_chars']} 字 | "
            f"输出: {stats['total_output_chars']} 字"
        )


def build_client_from_env() -> LLMClient:
    """
    从环境变量读取 API Key，快速构建 LLMClient。
    需要 .env 中设置 GEMINI_API_KEY。

    Usage:
        from src.utils.llm_client import build_client_from_env
        client = build_client_from_env()
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("未找到 GEMINI_API_KEY，请检查 .env 文件")

    return LLMClient(api_key=api_key)
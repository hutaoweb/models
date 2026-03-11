import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("inference.py")
SPEC = importlib.util.spec_from_file_location("rella_inference", MODULE_PATH)
inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference)


class GetPromptTestCase(unittest.TestCase):
    def test_english_input_uses_english_default_system_prompt(self):
        prompt = inference.get_prompt({"input": "You are helpful.", "output": "Yes."})

        self.assertTrue(prompt.startswith("<s>[INST]<<SYS>>\n"))
        self.assertIn("You are a helpful, respectful and honest assistant.", prompt)
        self.assertNotIn("你是一个乐于助人", prompt)
        self.assertIn("You are helpful.", prompt)
        self.assertTrue(prompt.endswith("[/INST] Yes."))

    def test_chinese_input_uses_chinese_default_system_prompt(self):
        prompt = inference.get_prompt({"input": "你是", "output": "是。"})

        self.assertTrue(prompt.startswith("<s>[INST]<<SYS>>\n"))
        self.assertIn("你是一个乐于助人、尊重他人且诚实的助手。", prompt)
        self.assertNotIn("You are a helpful, respectful and honest assistant.", prompt)
        self.assertIn("你是", prompt)
        self.assertTrue(prompt.endswith("[/INST] 是。"))

    def test_custom_system_prompt_overrides_language_detection(self):
        prompt = inference.get_prompt(
            {
                "input": "你是",
                "output": "是。",
                "system_prompt": "自定义系统提示。",
            }
        )

        self.assertIn("自定义系统提示。", prompt)
        self.assertNotIn("你是一个乐于助人、尊重他人且诚实的助手。", prompt)


if __name__ == "__main__":
    unittest.main()
